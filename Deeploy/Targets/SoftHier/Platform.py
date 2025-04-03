# ----------------------------------------------------------------------
#
# File: SoftHierPlatform.py
#
# Last edited: 03.04.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author:
# - Bowen Wang <bowwang@iis.ee.ethz.ch>, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import numpy as np

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicAddBindings, BasicConv1DBinding, BasicConv2DBindings, \
    BasicDebugPrintBindings, BasicDivBindings, BasicDWConv1DBinding, BasicDWConv2DBinding, BasicGatherBindings, \
    BasicGELUBindings, BasicLayerNormBindings, BasicMulBindings, BasicPad1DBindings, BasicPad2DBindings, \
    BasicReduceMeanBindings, BasicReduceSumBindings, BasicReshapeBindings, BasicRQIntegerDivBinding, \
    BasicRQSGELUBinding, BasicSliceBindings, BasicSoftmaxBindings, BasicTransposeBindings, DummyBinding
from Deeploy.Targets.Generic.Layers import AddLayer, ConvLayer, DebugPrintLayer, DivLayer, GatherLayer, GELULayer, \
    GEMMLayer, ITAMaxLayer, LayerNormLayer, MatMulLayer, MaxPoolLayer, MHSALayer, MulLayer, PadLayer, ReduceMeanLayer, \
    ReduceSumLayer, RequantShiftLayer, ReshapeLayer, RQGEMMLayer, RQIntegerDivLayer, RQMatMulLayer, RQSiGELULayer, \
    SliceLayer, SoftmaxLayer, TransposeLayer
from Deeploy.Targets.Generic.Parsers import AddParser, DebugParser, DummyParser, FlattenParser, GatherParser, \
    GELUParser, GenericConv1DParser, GenericConv2DParser, GenericDWConv1DParser, GenericDWConv2DParser, \
    GenericGEMMParser, GenericMaxPool2DParser, IntegerDivParser, ITAMaxParser, MatMulParser, MulParser, Pad1DParser, \
    Pad2DParser, ReduceMeanParser, ReduceSumParser, RequantShiftParser, ReshapeParser, RQGEMMParser, \
    RQIntegerDivParser, RQMatMulParser, RQSiGELUParser, SliceParser, TransposeParser, UnsqueezeParser, \
    iLayerNormParser, iSoftmaxParser
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import ExtractPaddingFromConvPass, \
    ExtractPaddingFromPoolPass, MatMulAddMergePass, MergeConstAddAndRequantPass, SplitAddPass, iGELURequantMergePass
    
from Deeploy.Targets.SoftHier.Bindings  import SoftHierGEMMBinding_8_8_32_32
# from Deeploy.Targets.SoftHier.Parsers   import <add parsers here>
from Deeploy.Targets.Softhier.Templates import AllocateTemplate, FreeTemplate
# from Deeploy.Targets.SoftHier.TopologyOptimizationPasses.Passes import <add Passes here>

# Fallback bindings from the generic platform
# (they support a wider range of attribute values)
GenericConv1D_Mapper = NodeMapper(GenericConv1DParser(), [BasicConv1DBinding])
GenericDWConv1D_Mapper = NodeMapper(GenericDWConv1DParser(), [BasicDWConv1DBinding])
GenericConv2D_Mapper = NodeMapper(GenericConv2DParser(), BasicConv2DBindings)
GenericDWConv2D_Mapper = NodeMapper(GenericDWConv2DParser(), [BasicDWConv2DBinding])

GenericConv_Mappers = [GenericConv2D_Mapper, GenericDWConv2D_Mapper, GenericConv1D_Mapper, GenericDWConv1D_Mapper]

# Basic bindings
Add_Mapper = NodeMapper(AddParser(), BasicAddBindings)

# MemPool specific bindings
GEMM_Mapper = NodeMapper(GenericGEMMParser(), [SoftHierGEMMBinding_8_8_32_32])

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

SoftHierlMapping = {
    'Gemm': GEMMLayer([GEMM_Mapper]),
}


class SoftHierVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.SoftHierInitTemplate
    allocTemplate = AllocateTemplate.SoftHierAllocateTemplate
    deallocTemplate = FreeTemplate.SoftHierLocalTemplate


class SoftHierTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.SoftHierInitTemplate
    allocTemplate = AllocateTemplate.SoftHierAllocateTemplate
    deallocTemplate = FreeTemplate.SoftHierLocalTemplate


class SoftHierConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.SoftHierGlobalInitTemplate
    allocTemplate = AllocateTemplate.SoftHierGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.SoftHierGlobalTemplate

    def _bufferRepresentation(self) -> Dict:
        retDict = super()._bufferRepresentation()
        # WIESEP: Workaround for banshee simulations.
        # Due to problems wrongly copied bytes, we want array sized a multiple of 4
        bytes = np.prod(self.shape) * (self._type.typeWidth // 8)
        if bytes % 4 != 0:
            bytes = 4 * int((bytes / 4 + 1))
        size = (bytes * 8) // self._type.typeWidth
        retDict['size'] = int(size)
        return retDict


class SoftHierStructBuffer(StructBuffer):

    initTemplate = AllocateTemplate.SoftHierStructInitTemplate
    allocTemplate = AllocateTemplate.SoftHierStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


SoftHierOptimizer = TopologyOptimizer([
    # MatMulAddMergePass(),
    # DebugPrintPass(r'.*[Mm]at[Mm]ul.*', position = 'after'),
])

includeList = ["DeeployMath.h", "runtime.h", "synchronization.h"]


class SoftHierEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = SoftHierlMapping, initCode: str = "", includeList = includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class SoftHierPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [SoftHierEngine("MemPool")],
                 variableBuffer = SoftHierVariableBuffer,
                 constantBuffer = SoftHierConstantBuffer,
                 structBuffer = SoftHierStructBuffer,
                 transientBuffer = SoftHierTransientBuffer):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_dma_pattern.h"
#include "flex_cluster_arch.h"

// Deeploy-generated
#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"


int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    flex_alloc_init();
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
    flex_intra_cluster_sync();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/
    uint32_t CID = flex_get_cluster_id();//Get cluster ID

    if(CID==0){ // only allow cluster 0 to work  
        if (flex_is_dm_core()) { // allow core 0 to init network and dma
            printf("Initializing network...\n");
            InitNetwork(0, 1);

            printf("Allocated >>> in0: 0x%8x, in1: 0x%8x\n", DeeployNetwork_inputs[0], DeeployNetwork_inputs[1]);
            printf("HBM       >>> in0: 0x%8x, in1: 0x%8x\n", (uint32_t)testInputVector[0], (uint32_t)testInputVector[1]);
            printf("size      >>> in0: 0x%8x, in1: 0x%8x\n", (uint32_t)DeeployNetwork_inputs_bytes[0], (uint32_t)DeeployNetwork_inputs_bytes[1]);

            for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
                void *addr = testInputVector[buf];
                //Trigger DMA transaction: move from HBM to L1
                uint64_t mask = 0x00000000ffffffff;
                // printf("%p\n", DeeployNetwork_inputs[buf]);
                // printf("%p\n", &DeeployNetwork_input_1);
                uint64_t masked_addr = (uint64_t)addr & mask;

                uint32_t tmp = ((uint32_t *)masked_addr)[0];
                printf("tmp = %x\n", tmp);

                flex_dma_async_1d(      DeeployNetwork_inputs[buf],
                                        masked_addr, 
                                        DeeployNetwork_inputs_bytes[buf]);
                // printf("%p\n", DeeployNetwork_inputs[buf]);
                // printf("%p\n", DeeployNetwork_input_1);
                //Wait all DMA transaction done
                flex_dma_async_wait_all();
                // printf("Done - %d\n", buf);
                // printf("out: %x, in0: %x, in1: %x\n", DeeployNetwork_output_0, DeeployNetwork_input_0, DeeployNetwork_input_1);
            }
            
        }
        flex_intra_cluster_sync();//Cluster barrier

        if (flex_is_first_core()) { // allow core 0 to compute
            printf("Running network...\r\n");
            // printf("out: %x, in0: %x, in1: %p\n", DeeployNetwork_output_0, DeeployNetwork_input_0, DeeployNetwork_input_1);
            // printf("out: %x, in0: %x, in1: %x\n", DeeployNetwork_output_0, DeeployNetwork_input_0, DeeployNetwork_input_1[0]);

            RunNetwork(0, 1);

        }
        flex_intra_cluster_sync();//Cluster barrier

        

        // verification
        int32_t tot_err = 0;
        uint32_t tot = 0;
        OUTPUTTYPE diff;
        OUTPUTTYPE expected, actual;

        if (flex_is_first_core()){
            for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
                tot += DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE);
                for (uint32_t i = 0; i < DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE); i++) {
                    expected = ((OUTPUTTYPE *)testOutputVector[buf])[i];
                    actual = ((OUTPUTTYPE *)DeeployNetwork_outputs[buf])[i];
                    diff = expected - actual;
                    if (diff != 0){
                        tot_err += 1;
                        printf("Expected: %4d  ", expected);
                        printf("Actual: %4d  ", actual);
                        printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
                    }
                }
            }
            printf("Errors: %d out of %d \r\n", tot_err, tot);
        }
        flex_intra_cluster_sync();//Cluster barrier
    }
    
    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
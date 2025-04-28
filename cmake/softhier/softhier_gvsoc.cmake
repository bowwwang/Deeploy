macro(add_gvsoc_emulation name)
  # Full path to the binary to simulate
  set(BINARY_PATH ${CMAKE_BINARY_DIR}/bin/${name})

  # Path to the gvsoc binary
  # set(GVSOC_EXECUTABLE /scratch2/bowwang/tmp_gvsoc_rebase/gvsoc/install/bin/gvsoc)

  # bowwang: adapted to docker
  set(GVSOC_EXECUTABLE $ENV{SOFTHIER_INSTALL_DIR}/install/bin/gvsoc)

  add_custom_target(gvsoc_${name}
    DEPENDS ${name}
    COMMAND ${GVSOC_EXECUTABLE}
            --target=pulp.chips.flex_cluster.flex_cluster
            --binary ${BINARY_PATH}
            run
            --trace=/chip/cluster_0/pe0
    COMMENT "Simulating deeploytest with GVSOC"
    USES_TERMINAL
    VERBATIM
  )
endmacro()
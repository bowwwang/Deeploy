macro(add_gvsoc_emulation name)
  # Full path to the binary to simulate
  set(BINARY_PATH ${CMAKE_BINARY_DIR}/bin/${name})

  # Path to the gvsoc binary
  set(GVSOC_EXECUTABLE /scratch2/bowwang/tmp_gvsoc_rebase/gvsoc/install/bin/gvsoc)

  add_custom_target(gvsoc_${name}
    DEPENDS ${name}
    COMMAND ${GVSOC_EXECUTABLE}
            --target=pulp.chips.flex_cluster.flex_cluster
            --binary ${BINARY_PATH}
            run
    COMMENT "Simulating deeploytest with GVSOC"
    USES_TERMINAL
    VERBATIM
  )
endmacro()
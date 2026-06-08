if (USE_HIP)
    message(STATUS "Generating sls_gpu (HIP)")
    file(GLOB HIP_SOURCES "./*.cu")
    set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library( sls_gpu ${HIP_SOURCES})
    set_target_properties( sls_gpu PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    target_compile_features( sls_gpu PRIVATE cxx_std_17)
    target_include_directories( sls_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src/lib)
    # Force-include the compat header before GLM so the GLM device-qualifier
    # override (see cuda_to_hip.h) is in effect in every HIP translation unit.
    target_compile_options( sls_gpu PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/cuda_to_hip.h)
    target_link_libraries( sls_gpu core)
else (USE_HIP)
    message(STATUS "CUDA found, generating sls_gpu")
    file(GLOB CUDA_SOURCES
        "./*"
        )
    #cuda_compile(SLS_OBJ ${CUDA_SOURCES})
    cuda_add_library( sls_gpu ${CUDA_SOURCES})
    target_link_libraries( sls_gpu core)
endif (USE_HIP)

aux_source_directory(. SOURCES)
add_executable(SLS App.cpp)
target_link_libraries( SLS core)

IF(${CUDA_FOUND})
    list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__ -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
    cuda_add_executable(SLS_GPU ./App_CUDA.cu)
    target_link_libraries(SLS_GPU core sls_gpu)
    set_target_properties ( SLS_GPU
        PROPERTIES
        #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        #LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
ENDIF(${CUDA_FOUND})

add_executable(SLS_CALIB App_Calib.cpp)
target_link_libraries(SLS_CALIB core sls_calib)

add_executable(SLS_GRAYCODE App_Graycode.cpp)
target_link_libraries(SLS_GRAYCODE core sls_graycode)

set_target_properties ( SLS
    PROPERTIES
    #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    #LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
set_target_properties ( SLS_CALIB
    PROPERTIES
    #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    #LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
set_target_properties ( SLS_GRAYCODE
    PROPERTIES
    #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    #LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

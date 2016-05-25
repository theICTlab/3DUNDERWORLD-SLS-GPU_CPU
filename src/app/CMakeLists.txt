aux_source_directory(. SOURCES)
add_executable(SLS App.cpp)
target_link_libraries( SLS core)

IF(${CUDA_FOUND})
    list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__ -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
    cuda_add_executable(SLS_GPU ./App_CUDA.cu)
    target_link_libraries(SLS_GPU core sls_gpu)
ENDIF(${CUDA_FOUND})

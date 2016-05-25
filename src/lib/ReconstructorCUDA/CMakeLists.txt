message(STATUS "CUDA found, generating sls_gpu")
file(GLOB CUDA_SOURCES
    "./*"
    )
list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__ -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
#cuda_compile(SLS_OBJ ${CUDA_SOURCES})
cuda_add_library( sls_gpu ${CUDA_SOURCES})


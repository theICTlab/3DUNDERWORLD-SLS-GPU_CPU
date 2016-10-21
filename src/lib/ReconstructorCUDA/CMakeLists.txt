message(STATUS "CUDA found, generating sls_gpu")
file(GLOB CUDA_SOURCES
    "./*"
    )
#cuda_compile(SLS_OBJ ${CUDA_SOURCES})
cuda_add_library( sls_gpu ${CUDA_SOURCES})


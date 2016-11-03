/*
 * CUDA error handling class
 */
#pragma once
#include <cuda_runtime.h>
#include <core/Log.hpp>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        LOG::writeLogErr("GPUassert: %s %s: %d\n", cudaGetErrorString(code), file, line);
        LOG::writeLogErr("Error Code: %d\n", code);
        if (abort) exit(code);
    }
}

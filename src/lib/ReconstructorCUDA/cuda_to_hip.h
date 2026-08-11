// CUDA-to-HIP compatibility shim. The single file that knows about HIP.
// On AMD (USE_HIP) it aliases the small CUDA surface this project uses to the
// HIP runtime; on NVIDIA it is a plain include of the CUDA runtime. Every other
// source keeps the CUDA spelling (cudaXxx, atomicInc, ...).
#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// Include libc string/memory decls BEFORE the HIP runtime so host memcpy/memset
// (used inside device helpers in this project) keep their libc decls; once
// <hip/hip_runtime.h> is in scope these names also gain __device__ overloads.
#include <cstring>
#include <cstdlib>

#include <hip/hip_runtime.h>

// GLM 0.9.9 only emits __host__ __device__ on its math functions (dot, length,
// normalize, mat*vec, ...) when it detects the NVIDIA CUDA compiler. It keys
// that on __CUDACC__ + CUDA_VERSION (glm/simd/platform.h), which hipcc/clang
// does not define, so by default every glm:: function is host-only and the
// device kernels here fail to compile. GLM is included (transitively via
// core/*.h and <glm/glm.hpp>) only AFTER this header, which is force-included
// first on the HIP targets, and AFTER <hip/hip_runtime.h> above is fully parsed.
// Defining these here steers GLM into its CUDA code path so its qualifier macro
// GLM_CUDA_FUNC_DEF expands to `__device__ __host__`, matching how a CUDA build
// of this project gets device-callable glm. GLM_FORCE_CUDA stops GLM from
// pulling in <cuda.h> to discover CUDA_VERSION.
#if !defined(__CUDACC__)
#define __CUDACC__ 1
#endif
#if !defined(CUDA_VERSION)
#define CUDA_VERSION 8000
#endif
#define GLM_FORCE_CUDA

// Error / status
#define cudaError_t            hipError_t
#define cudaSuccess            hipSuccess
#define cudaGetErrorString     hipGetErrorString
#define cudaPeekAtLastError    hipPeekAtLastError

// Memory
#define cudaMalloc             hipMalloc
#define cudaFree               hipFree
#define cudaMemcpy             hipMemcpy
#define cudaMemset             hipMemset
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

// Events (used for profiling timing only)
#define cudaEvent_t            hipEvent_t
#define cudaEventCreate        hipEventCreate
#define cudaEventRecord        hipEventRecord
#define cudaEventSynchronize   hipEventSynchronize
#define cudaEventElapsedTime   hipEventElapsedTime

// Abort the kernel. `trap;` is an NVPTX instruction with no amdgcn spelling;
// clang provides __builtin_trap() in device code, which nvcc does not, so each
// backend needs its own and neither compiles under the other.
#define gpuTrap()              __builtin_trap()

#else

#include <cuda_runtime.h>

#define gpuTrap()              asm("trap;")

#endif

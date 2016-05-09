#include "Dynamic_bits.cuh"
#include <core/Dynamic_Bitset.h>
#include <cassert>
#include <fstream>
namespace SLS
{
bool Dynamic_Bitset_Array::writeElemToPGM( std::string fileName, size_t elemIdx, const size_t &w, const size_t &h, bool transpose)
{
    const size_t numBytes = (bitsPerElem)/BITS_PER_BYTE;
    unsigned char *startBit = &bits[numBytes*elemIdx];
    unsigned char bits_h[numBytes];
    gpuErrchk(cudaMemcpy(bits_h, startBit, numBytes, cudaMemcpyDeviceToHost));
    Dynamic_Bitset bitset_h(numBytes, bits_h);
    bitset_h.writeToPGM(fileName, w, h, transpose);
    return true;
}

bool Dynamic_Bitset_Array::writeToPGM( std::string fileName, size_t w, size_t h, bool transposed, unsigned int maxValue)
{
    if ( w*h != numElem) 
    {
        LOG::writeLogErr("Misaligned between number of elements and output image resolution\n");
        return false;   // Simple assertion
    }
    if (maxValue == 0)
        // maxValue is by default equal to the maximum number of these bit can repesent. 
        maxValue = (long unsigned int)(1<<(bitsPerElem+1))-1;
    unsigned int *uintArray_d = nullptr;
    unsigned int *uintArray_h = nullptr;
    gpuErrchk (cudaMalloc ((void**)&uintArray_d,  sizeof(uint)*numElem));

    Kernel::toNormalizedUintArray<<<200,200>>>( getGPUOBJ(), 255, maxValue, uintArray_d);

    gpuErrchk (cudaPeekAtLastError());
    uintArray_h = new unsigned int[numElem];
    gpuErrchk ( cudaMemcpy ( uintArray_h, uintArray_d, sizeof(uint)*numElem, cudaMemcpyDeviceToHost));
    gpuErrchk (cudaFree (uintArray_d));


    FILE *pFp = std::fopen(fileName.c_str(), "wb");

    if (!pFp)
    {
        LOG::writeLogErr("Unable to open %s to write PGM\n", fileName.c_str());
        delete []uintArray_h;
        return false;
    }

    std::fprintf(pFp, "P5\n%zu\n%zu\n%d\n", w, h, 255);
    for (size_t i = 0; i < h*w; i++)
    {
        unsigned char val = (transposed?uintArray_h[(i%w)*h+i/w]:uintArray_h[i]);
        putc(val,pFp);
    }
    delete [] uintArray_h;
    return true;

}
bool Dynamic_Bitset_Array::writeToPPM( std::string fileName, size_t w, size_t h, bool transposed, unsigned int maxValue)
{    
    if ( w*h != numElem) 
    {
        LOG::writeLogErr("Misaligned between number of elements and output image resolution\n");
        return false;   // Simple assertion
    }
    if (maxValue == 0)
        // maxValue is by default equal to the maximum number of these bit can repesent. 
        maxValue = (long unsigned int)(1<<(bitsPerElem+1))-1;
    unsigned int *uintArray_d = nullptr;
    unsigned int *uintArray_h = nullptr;
    gpuErrchk (cudaMalloc ((void**)&uintArray_d,  sizeof(uint)*numElem));

    Kernel::toNormalizedUintArray<<<200,200>>>( getGPUOBJ(), (1<<3*8)-1, maxValue, uintArray_d);

    gpuErrchk (cudaPeekAtLastError());
    uintArray_h = new unsigned int[numElem];
    gpuErrchk ( cudaMemcpy ( uintArray_h, uintArray_d, sizeof(uint)*numElem, cudaMemcpyDeviceToHost));
    gpuErrchk (cudaFree (uintArray_d));


    FILE *pFp = std::fopen(fileName.c_str(), "wb");

    if (!pFp)
    {
        LOG::writeLogErr("Unable to open %s to write PPM\n", fileName.c_str());
        delete []uintArray_h;
        return false;
    }

    // Write to RGB ppm
    // http://paulbourke.net/dataformats/ppm/
    std::fprintf(pFp, "P6\n%zu\n%zu\n%d\n", w, h, 255);
    for (size_t i = 0; i < h*w; i++)
    {
        unsigned int val = (transposed?uintArray_h[(i%w)*h+i/w]:uintArray_h[i]);

        // convert first 3 bytes to RGB
        putc((unsigned char)((val>>16) & 0xFF),pFp);
        putc((unsigned char)((val>>8) & 0xFF),pFp);
        putc((unsigned char)((val>>0) & 0xFF),pFp);
    }
    delete [] uintArray_h;
    return true;
}

namespace Kernel
{
__global__ void toUintArray_kernel(
        Dynamic_Bitset_Array_GPU bitsArray,
        unsigned int *uintArray
        )
{
    uint idx = blockIdx.x*blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;
    while (idx < bitsArray.numElem)
    {
        uintArray[idx] = bitsArray.to_uint(idx);
        idx += stride;
    }
}
__global__ void toNormalizedUintArray(
        Dynamic_Bitset_Array_GPU bitsArray,
        unsigned int capValue,
        unsigned int maxValue,
        unsigned int *uintArray
        )
{
    uint idx = blockIdx.x*blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;
    while (idx < bitsArray.numElem)
    {
        uintArray[idx] = (uint)((long unsigned int)bitsArray.to_uint(idx) * (long unsigned int)capValue / maxValue);
        idx += stride;
    }
}

} // namespace Kernel
} // namespace SLS

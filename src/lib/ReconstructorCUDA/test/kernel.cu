#include "../Dynamic_bits.cuh"

__global__ void setBits(SLS::Dynamic_Bitset_Array_GPU bits)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > bits.getNumElem())
        return;
    bits.setBit(0, idx);
}
__global__ void clearBits(SLS::Dynamic_Bitset_Array_GPU bits)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > bits.getNumElem())
        return;
    bits.clearBit(0, idx);
}

__global__ void toUintArray(SLS::Dynamic_Bitset_Array_GPU bits, unsigned int *output)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > bits.getNumElem())
        return;
    output[idx] = bits.to_uint(idx);
}


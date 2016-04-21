#include "Dynamic_bits.cuh"
#include <core/Dynamic_Bitset.h>
namespace SLS
{
bool Dynamic_Bitset_Array::writeToPGM( std::string fileName, size_t elemIdx, const size_t &w, const size_t &h, bool transpose)
{
    const size_t numBytes = (bitsPerElem)/BITS_PER_BYTE;
    unsigned char *startBit = &bits[numBytes*elemIdx];
    unsigned char bits_h[numBytes];
    gpuErrchk(cudaMemcpy(bits_h, startBit, numBytes, cudaMemcpyDeviceToHost));
    Dynamic_Bitset bitset_h(numBytes, bits_h);
    bitset_h.writeToPGM(fileName, w, h, transpose);
    return true;
}
} // namespace SLS

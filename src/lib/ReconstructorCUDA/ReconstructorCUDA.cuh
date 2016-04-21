#pragma once
#include <core/Reconstructor.h>
#include <core/log.hpp>
#include "Dynamic_bits.cuh"

namespace SLS
{
class ReconstructorCUDA: public Reconstructor
{
private:
public:

     ReconstructorCUDA(const size_t projX, const size_t projY);
    ~ReconstructorCUDA() override;
    void addCamera(Camera *cam) override;
    void renconstruct() override;
};


__global__ void buildBucket_kernel(
        const uchar * imgs,
        size_t numimgs,
        size_t XtimesY,
        uchar whiteThreshold,
        Dynamic_Bitset_Array_GPU mask,
        Dynamic_Bitset_Array_GPU patterns
        );
__global__ void bucket2uint_kernel(
        Dynamic_Bitset_Array_GPU patterns,
        size_t XtimesY,
        uint * output
        );


// A helper function to output array to gray scale pgm
inline bool uint2PGM(const std::string &fileName, size_t w, size_t h, uint* array, uint max, bool transposed=false)
{
    FILE *pFp = std::fopen(fileName.c_str(), "wb");
    if (pFp)
    {
        std::fprintf(pFp, "P5\n%zu\n%zu\n%d\n", w, h, 255);
        for (size_t i = 0; i < h*w; i++)
        {
            uchar val = (transposed?array[(i%w)*h+i/w]:array[i])/max*255;
            putc(val,pFp);
        }
        return true;
    }
    else return false;
}

} // namespace SLS

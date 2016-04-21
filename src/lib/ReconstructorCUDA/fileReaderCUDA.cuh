#pragma once
#include <core/fileReader.h>
#include "Dynamic_bits.cuh"
#include <string>
namespace SLS
{
class FileReaderCUDA: public FileReader
{
protected:
    Dynamic_Bitset_Array *maskGPU_; // Sorry can't init before reading the image

public:
    FileReaderCUDA()=delete; 
    FileReaderCUDA(const std::string& cName): 
        FileReader(cName), maskGPU_(nullptr){};
    void computeShadowsAndThreasholds() override;
    ~FileReaderCUDA() override
    {
        if (maskGPU_ != nullptr)
            delete maskGPU_;
    }
    const Dynamic_Bitset_Array* getMask() const
    { return maskGPU_;}
};
__global__ void computeMask_kernel(
        unsigned char *brightImg,
        unsigned char *darkImg,
        uchar blackThreashold,
        size_t resX,
        size_t resY,
        Dynamic_Bitset_Array_GPU mask
        );
} // namespace SLS

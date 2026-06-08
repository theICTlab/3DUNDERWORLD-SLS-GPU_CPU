#pragma once
// Upstream renamed the camera base Camera -> ImageProcessor and FileReader ->
// ImageFileProcessor, but left this GPU camera pointing at the deleted
// core/FileReader.h / FileReader base, so the CUDA path no longer compiled on
// any backend. Repoint it at the current concrete base ImageFileProcessor,
// which supplies every accessor this class and the kernels use.
#include <core/ImageFileProcessor.h>
#include "DynamicBits.cuh"
#include <string>
namespace SLS
{
class FileReaderCUDA: public ImageFileProcessor
{
protected:
    Dynamic_Bitset_Array *maskGPU_; // Sorry can't init before reading the images

    // Init configurations in device
    std::array<float*, PARAM_COUNT> params_d_;
    float* camTransMat_d_;
    

public:
    FileReaderCUDA()=delete; 
    FileReaderCUDA(const std::string& cName):
        ImageFileProcessor(cName), maskGPU_(nullptr){
            gpuErrchk( cudaMalloc((void**)&params_d_[CAMERA_MAT], sizeof(float)*9));
            gpuErrchk( cudaMalloc((void**)&params_d_[DISTOR_MAT], sizeof(float)*5));
            gpuErrchk( cudaMalloc((void**)&params_d_[ROT_MAT], sizeof(float)*9));
            gpuErrchk( cudaMalloc((void**)&params_d_[TRANS_MAT], sizeof(float)*3));
            gpuErrchk( cudaMalloc((void**)&camTransMat_d_, sizeof(float)*16));
        };
    void computeShadowsAndThresholds() override;

    void loadConfig(const std::string& configFile) override;
    ~FileReaderCUDA() override {
        if (maskGPU_ != nullptr)
            delete maskGPU_;
        for (auto &p: params_d_)
            gpuErrchk( cudaFree(p));
        gpuErrchk( cudaFree(camTransMat_d_));
    }

    float* getDeviceCamMat() const { return params_d_[CAMERA_MAT];}
    float* getDeviceDistMat() const { return params_d_[DISTOR_MAT];}
    float* getDeviceCamTransMat() const { return camTransMat_d_;}

    const Dynamic_Bitset_Array* getMask() const { return maskGPU_;}
};
__global__ void computeMask_kernel(
        unsigned char *brightImg,
        unsigned char *darkImg,
        uchar blackThreshold,
        size_t resX,
        size_t resY,
        Dynamic_Bitset_Array_GPU mask
        );
} // namespace SLS

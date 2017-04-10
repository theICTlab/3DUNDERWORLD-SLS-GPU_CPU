#pragma once
#include <core/ImageFileProcessor.h>
#include "DynamicBits.cuh"
#include <string>

namespace SLS {
class ImageFileProcessorCUDA : public ImageFileProcessor {
    protected:
    Dynamic_Bitset_Array *maskGPU_; // Sorry can't init before reading the images

    // Init configurations in device
    std::array<float*, PARAM_COUNT> params_d_;
    float* camTransMat_d_;
public:
    ImageFileProcessorCUDA()=delete; 
    ImageFileProcessorCUDA(const std::string& cName): 
        ImageFileProcessor(cName), maskGPU_(nullptr){
            gpuErrchk( cudaMalloc((void**)&params_d_[CAMERA_MAT], sizeof(float)*9));
            gpuErrchk( cudaMalloc((void**)&params_d_[DISTOR_MAT], sizeof(float)*5));
            gpuErrchk( cudaMalloc((void**)&params_d_[ROT_MAT], sizeof(float)*9));
            gpuErrchk( cudaMalloc((void**)&params_d_[TRANS_MAT], sizeof(float)*3));
            gpuErrchk( cudaMalloc((void**)&camTransMat_d_, sizeof(float)*16));
        };
    void computeShadowsAndThresholds() override;

};
}

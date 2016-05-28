#pragma once
#include <opencv2/opencv.hpp>
#include <core/Projector.h>
#include <cmath>

namespace SLS{
class GrayCode: public Projector
{
public:
    explicit GrayCode(size_t projW, size_t projH);
    ~GrayCode() override
    {
        for (auto &img: grayCodes_)
            img.release();
    }
    void generateGrayCode();
private:
    std::vector<cv::Mat> grayCodes_;
    size_t currImg;
    size_t numColBits, numRowBits;
    void setColRowBitsNum()
    {
        numColBits = (size_t)std::ceil(std::log2((float) width_));
        numRowBits = (size_t)std::ceil(std::log2((float) height_));
    }
};
} // SLS


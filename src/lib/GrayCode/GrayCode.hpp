#pragma once
#include <opencv2/opencv.hpp>
#include <core/Projector.h>
#include <cmath>

namespace SLS{
class GrayCode
{
public:
    //! Construct gray code with size of projector
    explicit GrayCode(size_t projW, size_t projH);
    ~GrayCode() 
    {
        for (auto &img: grayCodes_)
            img.release();
    }
    void generateGrayCode();
private:
    //! Projector width and height
    size_t width_, height_;
    std::vector<cv::Mat> grayCodes_;
    size_t currImg;
    size_t numColBits, numRowBits;
};
} // SLS


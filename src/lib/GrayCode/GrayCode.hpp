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

    /*! Generate and display gray code
     *
     * The patterns are encoded by [gray code](https://en.wikipedia.org/wiki/Gray_code) to 
     * avoid error.
     * The binary are encoded separately for columns and rows. i.e. for a projector pixel (x, y)
     * we have (BitSeqX, BitSeqY).
     */
    const std::vector<cv::Mat>& generateGrayCode();
private:
    //! Projector width and height
    size_t width_, height_;
    std::vector<cv::Mat> grayCodes_;
    size_t currImg;
    size_t numColBits, numRowBits;
};
} // SLS


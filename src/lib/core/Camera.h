/* Copyright (C) 
 * 2016 - Tsing Gu
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 */

#pragma once
#include <string>
#include <array>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Dynamic_Bitset.h"
#include <glm/glm.hpp>
#include "Ray.h"

namespace SLS
{
/**
 * @ Base class of camera
 * 1. This class manages acquiring images form file/camera. 
 * 2. Loading configurations
 * 3. Undistort images
 * 4. Calculate mask
 * 5. Compute colors on point cloud
 */

class Camera
{
protected:

    // Name of camera
    std::string name_;

    // Camera Resolution
    size_t resX_, resY_;

    // Mask of valid pixels
    Dynamic_Bitset shadowMask_; //Color mask

    // An all lit image contains color information of reconstruction object
    cv::Mat color_;

    // Thresholds are used to filter out invalid pixel.
    uchar whiteThreshold_;

    // Contrast threshold
    // If the contrast of a pixel between lit and unlit is smaller than the 
    // dark threshold, the pixel is invalid.
    uchar blackThreshold_;

    // adaptive Thresholds of each pixel. 
    // The threshold is used filter out invalid pixels
    std::vector<unsigned char> thresholds_; //Threshold for each pixel with in [0,255], 
public:
    Camera() = delete;
    /*
     * Construction of a camera, it takes the name and the resolution
     * of the camera.
     */
    explicit Camera(const std::string &cName):name_(cName),resX_(0),resY_(0)
    {whiteThreshold_=5; blackThreshold_=40;}   //Hacking, need to read from file
    const std::string& getName() const {return name_;}
    void setName(const std::string &cName) {name_ = cName;}
    void getResolution(size_t &x, size_t &y) const{x=resX_; y=resY_;}
    const unsigned char& getThreashold(const size_t &idx){return thresholds_[idx];}
    // Get a mask by index
    bool queryMask(const size_t &idx){return shadowMask_.getBit(idx);}
    // Set the value of mask to one at given index
    void setMask(const size_t &idx){ shadowMask_.setBit(idx);}
    // Clear the value of mask to one at given index
    void clearMask(const size_t &idx){ shadowMask_.clearBit(idx);}
    uchar getWhiteThreshold() const {return whiteThreshold_;}
    uchar getblackThreshold() const {return blackThreshold_;}
    virtual const cv::Mat& getColorFrame() const
    {
        return color_;
    }
    void getColor(size_t x, size_t y, unsigned char &r, unsigned char &g, unsigned char &b) const
    {
        auto color = color_.at<cv::Vec3b>(y, x);
        b = color.val[0];
        g = color.val[1];
        r = color.val[2];
    }
    void getColor(size_t idx, unsigned char &r, unsigned char &g, unsigned char &b) const
    {
        getColor(idx/resY_, idx%resY_, r,g,b);
    }

    // Interfaces
    /**
     * @brief Get a ray in world space by given pixel
     *
     * @param x x coordinate of pixel
     * @param y y coordinate of pixel
     *
     * @return Ray shot from camera to this pixel
     */

    // Reconstruction relies on find intersection of two rays at the same point
    // from two cameras. The rays can be get from the following functions. 
    virtual Ray getRay(const size_t &x, const size_t &y)=0;
    virtual Ray getRay(const size_t &idx)=0;
    virtual void setResolution(const size_t &x, const size_t &y) {resX_ = x; resY_ = y;}
    virtual ~Camera(){}

    // Load camera intrinsic/extrinsic parameters and distortion from file
    virtual void loadConfig(const std::string &configFile) = 0;

    // Get next frame from camera
    virtual const cv::Mat& getNextFrame() = 0;

    // Undistort image using the distortion parameter.
    virtual void undistort()=0;

    // Compute masks and thresholds ( if dynamic threshold enabled 
    virtual void computeShadowsAndThresholds()=0;
};
}  // namespace SLS

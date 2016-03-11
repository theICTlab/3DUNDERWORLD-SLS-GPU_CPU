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
 * 1. this class manages aquiring images form file/camera. 
 * 2. loading configurations
 * 3. Undistort images
 * 4. Calculate shadows to discard
 * 5. save colors
 */

class Camera
{
 protected:
    std::string name_;


    size_t resX_, resY_;        //Camera Resolution
    
    std::vector<unsigned char> threasholds_; //Threashold for each pixel [0,255]
    Dynamic_Bitset shadowMask_; //Color mask
    cv::Mat color_;     //Color image
    uchar whiteThreshold_;
    uchar blackThreshold_;
 public:
    Camera() = delete;
    explicit Camera(const std::string &cName):name_(cName),resX_(0),resY_(0)
    {whiteThreshold_=250; blackThreshold_=5;}   //Hacking, need to read from file
    const std::string& getName() const {return name_;}
    void setName(const std::string &cName) {name_ = cName;}
    void getResolution(size_t &x, size_t &y) const{x=resX_; y=resY_;}
    const unsigned char& getThreashold(const size_t &idx){return threasholds_[idx];}
    bool queryMask(const size_t &idx){return shadowMask_.getBit(idx);}

    // Interfaces
    
    /**
     * @brief Get a ray in world space by given pixel
     *
     * @param x x coordinate of pixel
     * @param y y coordinate of pixel
     *
     * @return Ray shot from camera to this pixel
     */
    virtual Ray getRay(const size_t &x, const size_t &y)=0;
    virtual Ray getRay(const size_t &idx)=0;
    virtual void setResolution(const size_t &x, const size_t &y) {resX_ = x; resY_ = y;}
    virtual ~Camera(){}
    virtual void loadConfig(const std::string &configFile) = 0;
    virtual const cv::Mat& getNextFrame() = 0;
    virtual void undistort()=0;
    virtual void computeShadowsAndThreasholds()=0;
    virtual void nextFrame()=0;
};
}  // namespace SLS

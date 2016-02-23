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
namespace SLS
{
enum CAMERA_MAT{
    CAMERA_MAT=0,
    DISTOR_MAT,
    ROT_MAT,
    TRANS_MAT,
    PARAM_COUNT
};

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
    std::array<cv::Mat, PARAM_COUNT> params_;
    size_t resX_, resY_;        //Camera Resolution
    std::vector<unsigned char> threasholds_; //Threashold for each pixel [0,255]
    std::vector<unsigned char> shadowMask_; //Color mask
    cv::Mat color_;     //Color image
 public:
    Camera() = delete;
    explicit Camera(const std::string &cName):name_(cName) {}
    const std::string& getName() const {return name_;}
    void setName(const std::string &cName) {name_ = cName;}
    const std::array<cv::Mat, PARAM_COUNT>& getParams()const{return params_;}

    // Interfaces
    virtual ~Camera(){};
    virtual void loadConfig(const std::string &configFile) = 0;
    virtual const cv::Mat& getNextFrame() = 0;
    virtual void undistort()=0;
    virtual void computeShadowsAndThreasholds()=0;
};
}  // namespace SLS

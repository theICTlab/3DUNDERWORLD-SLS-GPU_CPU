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
class Camera
{
 protected:
    std::string name_;
    std::array<cv::Mat, PARAM_COUNT> params_;
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
};
}  // namespace SLS

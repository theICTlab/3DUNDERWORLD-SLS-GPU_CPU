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
#include "Camera.h"
namespace SLS
{
enum CAMERA_MAT{
    CAMERA_MAT=0,
    DISTOR_MAT,
    ROT_MAT,
    TRANS_MAT,
    PARAM_COUNT
};
class FileReader: public Camera
{
 private:
    std::vector<cv::Mat> images_;
    std::vector<cv::Mat> undistortedImages_;
    size_t frameIdx_;
    std::array<cv::Mat, PARAM_COUNT> params_;
    glm::mat4 camTransMat_; //Transformation matrix for camera
 public:
    //Constructors
    FileReader() = delete;
    explicit FileReader(const std::string& cName):Camera(cName),frameIdx_(0),camTransMat_(glm::mat4(1.0)){}

    //Extra functions
    void loadImages(const std::string& folder, bool isGL=false);
    void previousFrame() {frameIdx_=frameIdx_==0?frameIdx_:frameIdx_-1;}
    void nextFrame() override{frameIdx_=frameIdx_>=images_.size()?frameIdx_:frameIdx_+1;}
    const cv::Mat& getCurrentFrame() const {return images_[frameIdx_];}
    size_t getNumFrames() const { return images_.size(); }
    size_t getCurrentIdx() const {return frameIdx_;}
    const std::array<cv::Mat, PARAM_COUNT>& getParams()const{return params_;}
    
    //Implementing interfaces
    ~FileReader() override{}
    Ray getRay(const size_t &x, const size_t &y) override;
    void loadConfig(const std::string& configFile) override;
    const cv::Mat& getNextFrame() override;
    void undistort() override;
    void computeShadowsAndThreasholds() override;
};
}

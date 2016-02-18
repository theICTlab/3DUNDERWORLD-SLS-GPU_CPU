#include "fileReader.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include "log.hpp"
namespace SLS
{
void FileReader::loadImages(const std::string& folder, bool isGL)
{
    std::stringstream ss;
    if (folder.back() != '/')
        ss<<folder<<'/';
    else
        ss<<folder;
    while(true)
    {
        std::stringstream jpgss;
        jpgss<<std::setfill('0')<<std::setw(4)<<images_.size()<<".jpg";
        cv::Mat img=cv::imread(ss.str()+jpgss.str(), CV_LOAD_IMAGE_COLOR);
        if (!img.data)
            break;
        else
            images_.push_back(img);
    }
}

void FileReader::loadConfig(const std::string& configFile)
{
    LOG::writeLog("Loading config from: %s\n", configFile.c_str());
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    fs.root()["Camera"]["Matrix"]>>params_[CAMERA_MAT];
    fs.root()["Camera"]["Distortion"]>>params_[DISTOR_MAT];
    fs.root()["Camera"]["Translation"]>>params_[TRANS_MAT];
    fs.root()["Camera"]["Rotation"]>>params_[ROT_MAT];
    //Validation
    for (size_t i=0; i<PARAM_COUNT; i++)
        if (params_[i].empty())
            LOG::writeLogErr("Failed to load config %s\n", configFile.c_str());

}
const cv::Mat& FileReader::getNextFrame() 
{
    frameIdx_ = (frameIdx_+1)%(images_.size());
    return images_[frameIdx_];
}

void FileReader::undistortAll()
{
    //Validate matrices
    for (size_t i=0; i<PARAM_COUNT; i++)
        if (params_[i].empty())
        {
            LOG::writeLogErr("No parameters set for undistortion\n");
            return;
        }
    for (auto &img : images_)
    {
        cv::Mat temp;
        cv::undistort(img, temp, params_[CAMERA_MAT], params_[DISTOR_MAT]);
        temp.copyTo(img);
    }
}

}

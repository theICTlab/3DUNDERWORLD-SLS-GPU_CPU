#include "fileReader.h"
#include <sstream>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <iomanip>
#include "log.hpp"

//using uchar=unsigned char;
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
        std::string fName = ss.str()+jpgss.str();
        cv::Mat img=cv::imread(fName, CV_LOAD_IMAGE_COLOR);
        LOG::writeLog( "Reading image %s", fName.c_str());
        if (!img.data)
            break;
        else 
        {
            
            if( images_.size() == 0)    
                //Copy the first image to color
                img.copyTo(color_);

            cv::Mat gray;
            cv::cvtColor(img, gray, CV_BGR2GRAY);
            images_.push_back(gray);
        }
    }
    if (images_.empty())
        LOG::writeLogErr(" No image readed from %s", ss.str().c_str());
    else
    {
        resX_ = images_[0].cols;
        resY_ = images_[0].rows;
    }
}

void FileReader::loadConfig(const std::string& configFile)
{
    // Please refer to this link for paramters.
    // http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
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
    LOG::writeLog("Loading Camera %s\n", name_.c_str());
    //Write to camera transformation matrix
    //Mat = R^T*(p-T) => R^T * T * P;
    // Translation is performed before rotation
    //-T
    glm::mat4 translationMat(1.0);
    translationMat[3][0]=-params_[TRANS_MAT].at<double>(0,0);
    translationMat[3][1]=-params_[TRANS_MAT].at<double>(1,0);
    translationMat[3][2]=-params_[TRANS_MAT].at<double>(2,0);
    //R^T
    glm::mat4 rotationMat(1.0);
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            rotationMat[i][j]=params_[ROT_MAT].at<double>(i,j);
    camTransMat_ = rotationMat*translationMat;
    //-T
    //camTransMat_[3][0]=-params_[TRANS_MAT].at<double>(0,0);
    //camTransMat_[3][1]=-params_[TRANS_MAT].at<double>(1,0);
    //camTransMat_[3][2]=-params_[TRANS_MAT].at<double>(2,0);
    //std::cout<<params_[ROT_MAT].at<double>(0,0)<<std::endl;
    //std::cout<<"Rot\n"<<params_[ROT_MAT]<<std::endl;
    //std::cout<<"Trans\n"<<params_[TRANS_MAT]<<std::endl;

    std::cout<<glm::to_string(camTransMat_)<<std::endl;


    
}
const cv::Mat& FileReader::getNextFrame() 
{
    frameIdx_ = (frameIdx_+1)%(images_.size());
    return images_[frameIdx_];
}

void FileReader::undistort()
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


void FileReader::computeShadowsAndThreasholds()
{
    /*
     * Black threashold = 5;
     */
    cv::Mat& brightImg=images_[0];
    cv::Mat& darkImg=images_[1];
    shadowMask_.resize(resX_*resY_); 
    threasholds_.resize(resX_*resY_);
    //Column based
    for (size_t i=0; i< resX_; i++)
        for (size_t j=0; j<resY_; j++)
        {
            threasholds_[j+i*resY_] = brightImg.at<uchar>(j,i)-darkImg.at<uchar>(j,i);
            if (threasholds_[j+i*resY_] > blackThreshold_)
                shadowMask_.setBit(j+i*resY_);
            else
                shadowMask_.clearBit(j+i*resY_);
        }
}
Ray FileReader::getRay(const size_t &x, const size_t &y)
{
    Ray ray;
    ray.origin = camTransMat_*glm::vec4(0.0,0.0,0.0,1.0);

    //TODO: finish the ray function here
    ray.dir.x = 1.0;
    return ray;
}
}

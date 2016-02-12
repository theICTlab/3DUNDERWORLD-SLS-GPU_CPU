#include "fileReader.h"
#include <sstream>
#include <iostream>
#include <iomanip>
namespace SLS
{

void FileReader::loadImages(const std::string& folder)
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
}
const cv::Mat& FileReader::getNextFrame() 
{
    return images_[(frameIdx_++)%images_.size()];
}

}

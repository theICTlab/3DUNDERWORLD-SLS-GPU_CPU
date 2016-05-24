#include "fileReader.h"
#include <glm/gtx/string_cast.hpp>
#include <iomanip>
#include <fstream>

//using uchar=unsigned char;
namespace SLS
{
void FileReader::loadImages(const std::string& folder, bool isGL)
{
    LOG::startTimer("Loading image from %s ... ", folder.c_str());
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
        if (!img.data)
            break;
        else 
        {
            if( images_.size() == 0)
                //Copy the first image to color
                img.copyTo(color_);

            cv::Mat gray;
            cv::cvtColor(img, gray, CV_BGR2GRAY);
            images_.push_back(gray.clone());
        }
        img.release();
    }
    if (images_.empty())
        LOG::writeLogErr(" No image read from %s ... ", ss.str().c_str());
    else
    {
        resX_ = images_[0].cols;
        resY_ = images_[0].rows;
        //rayTable.resize(resX_*resY_);
        thresholds_.resize(resY_*resX_, whiteThreshold_);
        LOG::writeLog("%d images loaded ...", images_.size());
    }
    LOG::endTimer('s');
}

void FileReader::loadConfig(
        const std::string& distMat,
        const std::string& camMat,
        const std::string& transMat,
        const std::string& rotMat
        )
{
    // Not implemented yet
    cv::FileStorage fs(distMat, cv::FileStorage::READ);
    
}
        

void FileReader::loadConfig(const std::string& configFile)
{
    // Please refer to this link for paramters.
    // http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    LOG::writeLog("Loading config for \"%s\" from: %s\n", name_.c_str(), configFile.c_str());
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    fs.root()["Camera"]["Matrix"]>>params_[CAMERA_MAT];
    fs.root()["Camera"]["Distortion"]>>params_[DISTOR_MAT];
    fs.root()["Camera"]["Translation"]>>params_[TRANS_MAT];
    fs.root()["Camera"]["Rotation"]>>params_[ROT_MAT];
    fs.release();

    //Debug
    //Validation
    for (size_t i=0; i<PARAM_COUNT; i++)
        if (params_[i].empty())
            LOG::writeLogErr("Failed to load config %s\n", configFile.c_str());
    //Write to camera transformation matrix
    //Mat = R^T*(p-T) => R^T * -T * P;
    // Translation is performed before rotation
    //-T
    glm::mat4 translationMat(1.0);
    translationMat[3][0]=-params_[TRANS_MAT].at<double>(0);
    translationMat[3][1]=-params_[TRANS_MAT].at<double>(1);
    translationMat[3][2]=-params_[TRANS_MAT].at<double>(2);
    //R^T, row base to column base, translated
    glm::mat4 rotationMat(1.0);
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            rotationMat[i][j]=params_[ROT_MAT].at<double>(j,i);
    rotationMat = glm::inverse(rotationMat);
    camTransMat_ = rotationMat*translationMat;

    //No need to do this
    //Generating raytable
    //LOG::startTimer("Generating ray lookup table with size of %d*%d...",
    //        resX_, resY_);
    //for (size_t i=0; i<resX_; i++)
    //    for (size_t j=0; j<resY_; j++){
    //        Ray ray;
    //        ray.origin = camTransMat_*glm::vec4(0.0,0.0,0.0,1.0);

    //        ray.dir.x = ((float)i-params_[CAMERA_MAT].at<double>(0,2))/params_[CAMERA_MAT].at<double>(0,0);
    //        ray.dir.y = ((float)j-params_[CAMERA_MAT].at<double>(1,2))/params_[CAMERA_MAT].at<double>(1,1);
    //        ray.dir.z=1.0;
    //        ray.dir.w=0.0;
    //        ray.dir = camTransMat_*ray.dir;
    //        ray.dir=glm::normalize(ray.dir);
    //        rayTable[j+i*resY_] = ray;
    //    }
    //LOG::endTimer('s');
}
const cv::Mat& FileReader::getNextFrame() 
{
    //Return the current frame and move on
    frameIdx_ = frameIdx_ % images_.size();
    return images_[frameIdx_++];
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
    LOG::startTimer();
    for (auto &img : images_)
    {
        cv::Mat temp;
        cv::undistort(img, temp, params_[CAMERA_MAT], params_[DISTOR_MAT]);
        temp.copyTo(img);
    }
    LOG::endTimer("Undistorted %d images in ", images_.size());
}


void FileReader::computeShadowsAndThresholds()
{
    /*
     * Black threashold = 5;
     */
    cv::Mat& brightImg=images_[0];
    cv::Mat& darkImg=images_[1];
    shadowMask_.resize(resX_*resY_); 
    cv::waitKey(0);
    //Column based
    for (size_t i=0; i< resX_; i++)
        for (size_t j=0; j<resY_; j++)
        {
            auto diff = brightImg.at<uchar>(j,i) - darkImg.at<uchar>(j,i);
            if (diff > blackThreshold_)
                shadowMask_.setBit(j+i*resY_);
            else
                shadowMask_.clearBit(j+i*resY_);
#ifdef AUTO_CONTRAST
            thresholds_[j+i*resY_]=diff/2;
#endif
        }
    std::string pgmFName=name_+".pgm";
    LOG::writeLog("Writing mask to %s\n", pgmFName.c_str());
    shadowMask_.writeToPGM(pgmFName.c_str(), resX_, resY_, true);
}
Ray FileReader::getRay(const size_t &x, const size_t &y)
{
        glm::vec2 undistorted = undistortPixel(glm::vec2(x, y));
        Ray ray;
        if (undistorted.x > resX_ || undistorted.y > resY_)
        {
            ray.dir = vec4(0.0);
            LOG::writeLogErr("Invalid ray");
            return ray;
        }
        ray.origin = camTransMat_*glm::vec4(0.0,0.0,0.0,1.0);
        ray.dir.x = (undistorted.x-params_[CAMERA_MAT].at<double>(0,2))/params_[CAMERA_MAT].at<double>(0,0);
        ray.dir.y = (undistorted.y-params_[CAMERA_MAT].at<double>(1,2))/params_[CAMERA_MAT].at<double>(1,1);
        ray.dir.z=1.0;
        ray.dir.w=0.0;
        ray.dir = camTransMat_*ray.dir;
        ray.dir=glm::normalize(ray.dir);
        return ray;
}
Ray FileReader::getRay(const size_t &pixelIdx) 
{
        glm::vec2 undistorted = undistortPixel(pixelIdx);
        Ray ray;

        if (undistorted.x > resX_ || undistorted.y > resY_)
        {
            ray.dir = vec4(0.0);
            LOG::writeLogErr("Invalid ray");
            return ray;
        }

        ray.origin = camTransMat_*glm::vec4(0.0,0.0,0.0,1.0);
        ray.dir.x = (undistorted.x-params_[CAMERA_MAT].at<double>(0,2))/params_[CAMERA_MAT].at<double>(0,0);
        ray.dir.y = (undistorted.y-params_[CAMERA_MAT].at<double>(1,2))/params_[CAMERA_MAT].at<double>(1,1);
        ray.dir.z=1.0;
        ray.dir.w=0.0;
        ray.dir = camTransMat_*ray.dir;
        ray.dir=glm::normalize(ray.dir);

        return ray;
}



void FileReader::rayTableToPointCloud(std::string fileName) const
{

    std::ofstream of(fileName);
    for (size_t i=0; i<rayTable.size(); i++)
    {
        const auto& ray = rayTable[i];
        if (shadowMask_.getBit(i))
        {
            auto endP = ray.origin+ray.dir*5000.0f;
            of<<"v "<<endP.x<<" "<<endP.y<<" "<<endP.z<<std::endl;
        }
    }
    of.close();
}

glm::vec2 FileReader::undistortPixel(const glm::vec2 &distortedPixel) const
{
    double k[5] = {0.0};
    double fx, fy, ifx, ify, cx, cy;
    int iters = 1;

    k[0] = params_[DISTOR_MAT].at<double> (0);
    k[1] = params_[DISTOR_MAT].at<double> (1);
    k[2] = params_[DISTOR_MAT].at<double> (2);
    k[3] = params_[DISTOR_MAT].at<double> (3);
    k[4] = 0;

    iters = 5;

    fx = params_[CAMERA_MAT].at<double>(0,0);
    fy = params_[CAMERA_MAT].at<double>(1,1);
    ifx = 1.0/fx;
    ify = 1.0/fy;
    cx = params_[CAMERA_MAT].at<double>(0,2);
    cy = params_[CAMERA_MAT].at<double>(1,2);

    double x,y,x0,y0;

    x = distortedPixel.x;
    y = distortedPixel.y;

	x0 = x = (x - cx)*ifx;
	y0 = y = (y - cy)*ify;

    for(int jj = 0; jj < iters; jj++ )
	{
		double r2 = x*x + y*y;
		double icdist = 1./(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
		double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x);
		double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y;
		x = (x0 - deltaX)*icdist;
		y = (y0 - deltaY)*icdist;
	}
    return glm::vec2((float)(x*fx)+cx,(float)(y*fy)+cy);
}

}

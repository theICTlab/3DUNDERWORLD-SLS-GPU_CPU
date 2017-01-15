#pragma once
#include <core/FileReader.h>
#include <string>
#include <condition_variable>

namespace SLS
{

const int WINDOW_WIDTH=1024;
const int WINDOW_HEIGHT=768;

class Calibrator
{

    static bool closeAsynImg;

    static void showImgAsync(const cv::Mat &img, const std::string &windowName)
    {
        while (!closeAsynImg)
        {
            cv::imshow(windowName, img);
            cv::waitKey(30);
        }
    }
    
    static void showImgWithText_Block( const cv::Mat &img, const std::string &text, const std::string &windowName)
    {
        cv::Mat textImg;
        cv::cvtColor(img, textImg, CV_GRAY2RGB);
        cv::putText(textImg, text, cvPoint(20,70), 
                cv::FONT_HERSHEY_SIMPLEX, 3.0, cvScalar(0,0,255), 2, CV_AA);
        //cv::putText(image, cv::Point(10,10), str.str(), CV_FONT_HERSHEY_PLAIN, CV_RGB(0,0,250));
        cv::imshow(windowName, textImg);
        textImg.release();
    }
    /**
     * @brief   Manually pick for extrenal corners of a image of checkerboard
     *
     * @param img   Input image
     *
     * @return 4 external points
     */
    static cv::vector<cv::Point2f>  manualMarkCheckBoard(cv::Mat img);

    /**
     * @brief Extract corners in an image
     *
     * @param img           Image to find corners in gray scale
     * @param camCorners    Output of image corners
     * @param objCorners    Output of object corners
     * @param squareSize    size of square
     *
     * @return 
     */
    static bool findCornersInCamImg(const cv::Mat &img,cv::vector<cv::Point2f> &camCorners,cv::vector<cv::Point3f> &objCorners, cv::Size squareSize);
    static float markWhite(const cv::Mat &img);
public:
    static void Calibrate(FileReader *cam, const std::string& calibImgsDir, const std::string &calibFile);
};
} // namespace SLS

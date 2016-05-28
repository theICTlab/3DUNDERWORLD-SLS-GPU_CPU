#pragma once
#include <core/fileReader.h>
#include <string>

namespace SLS
{
class Calibrator
{


    static void showImgAsync(const cv::Mat &img, const std::string &windowName)
    {
        cv::imshow(windowName, img);
        cv::waitKey(30);
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
    static bool findCornersInCamImg(cv::Mat img,cv::vector<cv::Point2f> *camCorners,cv::vector<cv::Point3f> *objCorners, cv::Size squareSize);
public:
    static void Calibrate(FileReader *cam, const std::string& calibImgsDir, const std::string &calibFile);
};
} // namespace SLS

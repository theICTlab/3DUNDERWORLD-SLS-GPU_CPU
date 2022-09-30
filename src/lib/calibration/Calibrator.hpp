#pragma once
#include <core/ImageFileProcessor.h>
#include <string>
#include <condition_variable>

namespace SLS {

const int WINDOW_WIDTH = 1024;
const int WINDOW_HEIGHT = 768;
/*! Calibrator is a manual calibration tool to generate calibration
 * configuration from checkerboard images. The calibration images must be in the
 * form of **x.JPG** where x is a one digit number.
 */

class Calibrator {

    /*! Show \p img with \p text on it. The window name is \p windowName
     *
     * \param img Image to overlay
     * \param text to overlay
     * \param window name
     */
    static void showImgWithText_Block(const cv::Mat &img,
                                      const std::string &text,
                                      const std::string &windowName)
    {
        cv::Mat textImg;
        cv::cvtColor(img, textImg, cv::COLOR_BGR2GRAY);
        cv::putText(textImg, text, cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX,
                    3.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::imshow(windowName, textImg);
        textImg.release();
    }

    /**
     *! Manually pick for extrenal corners of a image of checkerboard
     *
     * /param img   Input image
     * /return 4 external points
     */
    static std::vector<cv::Point2f> manualMarkCheckBoard(cv::Mat img);

    /*! Extract corners in an image
     *
     * \param img           Image to find corners in gray scale
     * \param camCorners    Output of image corners
     * \param objCorners    Output of object corners
     * \param squareSize    size of square
     *
     * \return
     */
    static bool findCornersInCamImg(const cv::Mat &img,
                                    std::vector<cv::Point2f> &camCorners,
                                    std::vector<cv::Point3f> &objCorners,
                                    cv::Size squareSize);
    /*! Mark a white pixel on \p img image. 
     */
    static float markWhite(const cv::Mat &img);

public:
    /*! Calibrate a camera
     * \param cam A fileReader to load calibration checkerboards
     * \param calibImgsDir Directory contains checkerboard
     * \param calibFile Output calibration result
     */
    static void Calibrate(ImageFileProcessor *cam, const std::string &calibImgsDir,
                          const std::string &calibFile);
};
}  // namespace SLS

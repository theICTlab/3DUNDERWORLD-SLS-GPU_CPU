#include "fileReader.h"
#include "log.hpp"

int main()
{
    LOG::restartLog();
    //SLS::FileReader reader("rightReader");
    //std::cout<<"Loading"<<std::endl;

    ////reader.loadImages("../../../data/alexander/rightCam/dataset1/");

    //reader.loadConfig("../../../data/alexander/rightCam/calib/output/calib.xml");
    //reader.loadConfig("../../../data/alexander/leftCam/calib/output/calib.xml");

    //std::cout<<"Undistorting"<<std::endl;
    //reader.undistort();
    //for(;;)
    //{
    //    cv::namedWindow("right", cv::WINDOW_NORMAL);
    //    cv::imshow("right", reader.getNextFrame());
    //    cv::resizeWindow("right", 800, 600);
    //    if(cv::waitKey(100)==27) 
    //        break;
    //}
    

    SLS::FileReader rightCam("rightCamera");
    SLS::FileReader leftCam("leftCamera");
    rightCam.loadConfig("../../../data/alexander/rightCam/calib/output/calib.xml");
    leftCam.loadConfig("../../../data/alexander/leftCam/calib/output/calib.xml");
    //rightCam.loadImages("../../../data/alexander/rightCam/dataset1/");
    //leftCam.loadImages("../../../data/alexander/leftCam/dataset1/");
    //for (;;)
    //{
    //    cv::namedWindow("right", cv::WINDOW_NORMAL);
    //    cv::imshow("right", rightCam.getNextFrame());
    //    cv::namedWindow("left", cv::WINDOW_NORMAL);
    //    cv::imshow("left", leftCam.getNextFrame());
    //    if (cv::waitKey(200)==27)
    //        break;
    //}
    
    return 0;
}

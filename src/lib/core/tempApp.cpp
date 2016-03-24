#include "fileReader.h"
#include "log.hpp"
#include <glm/gtx/string_cast.hpp>
#include "ReconstructorCPU.h"

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
    

    SLS::FileReader *rightCam=new SLS::FileReader("rightCamera");
    SLS::FileReader *leftCam= new SLS::FileReader("leftCamera");
    rightCam->loadImages("../../../data/alexander/rightCam/dataset1/");
    leftCam->loadImages("../../../data/alexander/leftCam/dataset1/");
    rightCam->loadConfig("../../../data/alexander/rightCam/calib/output/calib.xml");
    leftCam->loadConfig("../../../data/alexander/leftCam/calib/output/calib.xml");

    //leftCam->computeShadowsAndThreasholds();
    //leftCam->rayTableToPointCloud("testLeft.obj");
    //rightCam->computeShadowsAndThreasholds();
    //rightCam->rayTableToPointCloud("testRight.obj");

    
    SLS::ReconstructorCPU renconstruct(1024,768);
    renconstruct.addCamera(rightCam);
    renconstruct.addCamera(leftCam);
    //renconstruct.renconstruct();




    //for (;;)
    //{
    //    cv::namedWindow("right", cv::WINDOW_NORMAL);
    //    cv::imshow("right", rightCam->getNextFrame());
    //    cv::namedWindow("left", cv::WINDOW_NORMAL);
    //    cv::imshow("left", leftCam->getNextFrame());
    //    if (cv::waitKey(0)==27)
    //        break;
    //}
    LOG::writeLog("DONE!\n");
    
    return 0;
}

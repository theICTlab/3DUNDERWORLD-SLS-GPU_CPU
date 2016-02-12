#include "fileReader.h"

int main()
{
    SLS::FileReader reader("myReader");
    reader.loadImages("../../../data/alexander/rightCam/dataset1/");
    cv::namedWindow("test", cv::WINDOW_NORMAL);
    //for(;;)
    {
        cv::imshow("test", reader.getCurrentFrame());
        cv::resizeWindow("test", 800, 600);
        //if(cv::waitKey(0)=='q') ;
        cv::waitKey(0);
    }
    return 0;
}

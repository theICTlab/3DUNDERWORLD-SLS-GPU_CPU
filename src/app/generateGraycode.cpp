#include <iostream>
#include <GrayCode/GrayCode.hpp>
int main()
{
    // Generate gray code based on the resolution of the projector.
    SLS::GrayCode gc(1024, 768);
    std::vector<cv::Mat> grayCodeImages = gc.generateGrayCode();

    // Display graycode
    std::cout<<"Press 'q' to exit\n";
    std::cout<<"Press any key to show next image\n";
    for (const auto & image: grayCodeImages)
    {
        cv::imshow("GrayCode", image);
        if (cv::waitKey(0) == 'q')
            break;
    }
    return 0;
}

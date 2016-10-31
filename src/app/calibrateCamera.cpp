#include <calibration/Calibrator.hpp>
using namespace SLS;
int main()
{
    FileReader rightCam("CalibCameraRight");
    FileReader leftCam("CalibCameraLeft");

    // Calibrate right camera image, save the result to right.xml
    Calibrator::Calibrate(&rightCam,"../../data/alexander/rightCam/calib", "right.xml");
    // Calibrate left camera image, save the result to left.xml
    Calibrator::Calibrate(&leftCam,"../../data/alexander/leftCam/calib", "left.xml");
    return 0;
}

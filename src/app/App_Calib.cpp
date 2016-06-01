#include <calibration/Calibrator.hpp>
using namespace SLS;
int main()
{
    FileReader rightCam("CalibCameraRight");
    FileReader leftCam("CalibCameraLeft");
    Calibrator::Calibrate(&rightCam,"../../data/alexander/rightCam/calib", "right.xml");
    Calibrator::Calibrate(&leftCam,"../../data/alexander/leftCam/calib", "left.xml");
    return 0;
}

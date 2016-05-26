#include <calibration/Calibrator.hpp>
using namespace SLS;
int main()
{
    FileReader fr("CalibCamera");
    Calibrator::Calibrate(&fr,"/home/tsing/project/SLS/data/alexander/rightCam/calib", "test.xml");
    return 0;
}

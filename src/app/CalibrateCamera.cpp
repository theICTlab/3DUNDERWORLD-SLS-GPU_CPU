#include <calibration/Calibrator.hpp>
#include "cmdline.h"
using namespace SLS;
int main(int argc, char** argv)
{
    cmdline::parser p;
    p.add<std::string>("images", 'i', "Folder contains images for calibration", true);
    p.add<std::string>("output", 'o', "Output of calibration file", true);
    p.parse_check(argc, argv);

    FileReader cam("calibCam");

    // Calibrator takes a camera as input and output camera configuration.
    std::cout<<"Input param: "<<p.get<std::string>("images")<<std::endl;
    Calibrator::Calibrate(&cam, "../../data/alexander/rightCam/calib", p.get<std::string>("output"));
    return 0;
}

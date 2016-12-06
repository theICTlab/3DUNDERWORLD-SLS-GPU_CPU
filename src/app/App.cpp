#include <core/FileReader.h>
#include <core/Reconstructor.h>
#include <core/Log.hpp>
#include <glm/gtx/string_cast.hpp>
#include <core/ReconstructorCPU.h>
#include <memory>
#include "cmdline.h"

int main(int argc, char** argv)
{
    
    cmdline::parser p;
    p.add<std::string>("leftcam", 'l',"Left camera image folder", false, "../../data/alexander/leftCam/dataset1/");
    p.add<std::string>("rightcam", 'r',"Right camera image folder", false, "../../data/alexander/rightCam/dataset1/");
    p.add<std::string>("leftconfig", 'L',"Left camera configuration file", false, "../../data/alexander/leftCam/calib/output/calib.xml");
    p.add<std::string>("rightconfig", 'R',"Right camera configuration file", false, "../../data/alexander/rightCam/calib/output/calib.xml");
    p.add<std::string>("output", 'o',"Right camera configuration file", false, "output.ply");
    p.add<std::string>("format", 'f',"suffix of image files, e.g. jpg", false, "jpg");
    p.add<size_t>("width", 'w',"Projector width", false, 1024);
    p.add<size_t>("height", 'h',"Projector height", false, 768);
    p.parse_check(argc, argv);

    // TODO: Add parameters
    // 1. file suffix 
    // 2. projector resolution

    LOG::restartLog();
    std::string leftCameraFolder = p.get<std::string>("leftcam");
    std::string rightCameraFolder = p.get<std::string>("rightcam");
    std::string leftConfigFile = p.get<std::string>("leftconfig");
    std::string rightConfigFile = p.get<std::string>("rightconfig");
    std::string output = p.get<std::string>("output");
    std::string suffix = p.get<std::string>("format");

    auto rightCam = std::unique_ptr<SLS::FileReader>(new SLS::FileReader("rightCamera"));
    auto leftCam = std::unique_ptr<SLS::FileReader>(new SLS::FileReader("leftCamera"));

    rightCam->loadImages(rightCameraFolder, suffix);
    leftCam->loadImages(leftCameraFolder, suffix);

    rightCam->loadConfig(rightConfigFile);
    leftCam->loadConfig(leftConfigFile);

    SLS::ReconstructorCPU reconstruct(p.get<size_t>("width"), p.get<size_t>("height"));
    reconstruct.addCamera(rightCam.get());
    reconstruct.addCamera(leftCam.get());

    reconstruct.reconstruct();

    auto extension = output.substr(output.find_last_of(".")+1);
    SLS::exportPointCloud(output, extension, reconstruct);

    LOG::writeLog("DONE!\n");
    
    return 0;
}

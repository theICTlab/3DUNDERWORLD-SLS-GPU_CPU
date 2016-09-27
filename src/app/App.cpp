#include <core/fileReader.h>
#include <core/Reconstructor.h>
#include <core/log.hpp>
#include <glm/gtx/string_cast.hpp>
#include <core/ReconstructorCPU.h>
#include "cxxopts.hpp"

int main(int argc, char** argv)
{
    
    cxxopts::Options options("SLS_CPU", "CPU implementation of SLS");
    options.add_options()
        ("l,leftcam", "Left camera image folder", cxxopts::value<std::string>()->default_value("../../data/alexander/leftCam/dataset1/"))
        ("lc", "Left camera configuration file", cxxopts::value<std::string>()->default_value("../../data/alexander/leftCam/calib/output/calib.xml"))
        ("r,rightcam", "Right camera image folder", cxxopts::value<std::string>()->default_value("../../data/alexander/rightCam/dataset1/"))
        ("rc", "Right camera configuration file", cxxopts::value<std::string>()->default_value("../../data/alexander/rightCam/calib/output/calib.xml"))
        ("o,output", "Output file", cxxopts::value<std::string>()->default_value("test.ply"))
        ("h,help","Print this help")
        ;
    options.parse(argc, argv);

    if (options.count("help"))
    {
        std::cout<<options.help()<<std::endl;
        return 0;
    }

    LOG::restartLog();
    std::string leftCameraFolder = options["l"].as<std::string>();
    std::string rightCameraFolder = options["r"].as<std::string>();
    std::string leftConfigFile = options["lc"].as<std::string>();
    std::string rightConfigFile = options["rc"].as<std::string>();
    std::string output = options["output"].as<std::string>();

    SLS::FileReader *rightCam=new SLS::FileReader("rightCamera");
    SLS::FileReader *leftCam= new SLS::FileReader("leftCamera");

    rightCam->loadImages(rightCameraFolder);
    leftCam->loadImages(leftCameraFolder);

    rightCam->loadConfig(rightConfigFile);
    leftCam->loadConfig(leftConfigFile);

    
    SLS::ReconstructorCPU reconstruct(1024,768);
    reconstruct.addCamera(rightCam);
    reconstruct.addCamera(leftCam);
    reconstruct.reconstruct();

    SLS::exportPLYGrid(output,  reconstruct);

    LOG::writeLog("DONE!\n");
    
    return 0;
}

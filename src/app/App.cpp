#include <core/FileReader.h>
#include <core/Reconstructor.h>
#include <core/Log.hpp>
#include <glm/gtx/string_cast.hpp>
#include <core/ReconstructorCPU.h>
#include "cmdline.h"

int main(int argc, char** argv)
{
    
    cmdline::parser p;
    p.add<std::string>("leftcam", 'l',"Left camera image folder", false, "../../data/alexander/leftCam/dataset1/");
    p.add<std::string>("rightcam", 'r',"Right camera image folder", false, "../../data/alexander/rightCam/dataset1/");
    p.add<std::string>("leftconfig", 'L',"Left camera configuration file", false, "../../data/alexander/leftCam/calib/output/calib.xml");
    p.add<std::string>("rightconfig", 'R',"Right camera configuration file", false, "../../data/alexander/rightCam/calib/output/calib.xml");
    p.add<std::string>("output", 'o',"Right camera configuration file", false, "output.ply");
    p.parse_check(argc, argv);

    LOG::restartLog();
    std::string leftCameraFolder = p.get<std::string>("leftcam");
    std::string rightCameraFolder = p.get<std::string>("rightcam");
    std::string leftConfigFile = p.get<std::string>("leftconfig");
    std::string rightConfigFile = p.get<std::string>("rightconfig");
    std::string output = p.get<std::string>("output");

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

    SLS::exportPointCloud(output, "PLY", reconstruct);

    LOG::writeLog("DONE!\n");
    
    return 0;
}

#include <ReconstructorCUDA/FileReaderCUDA.cuh>
#include <ReconstructorCUDA/ReconstructorCUDA.cuh>
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

    LOG::restartLog();
    SLS::FileReaderCUDA *rightCam=new SLS::FileReaderCUDA("rightCamera");
    SLS::FileReaderCUDA *leftCam= new SLS::FileReaderCUDA("leftCamera");
    rightCam->loadImages(rightCameraFolder);
    leftCam->loadImages(leftCameraFolder);
    rightCam->loadConfig(rightConfigFile);
    leftCam->loadConfig(leftConfigFile);
    SLS::ReconstructorCUDA reconstruct(1024,768);
    reconstruct.addCamera(rightCam);
    reconstruct.addCamera(leftCam);
    reconstruct.reconstruct();
    //SLS::exportOBJVec4(output,  reconstruct);
    SLS::exportPointCloud(output, "PLY", reconstruct);
    LOG::writeLog("DONE\n");
    return 0;
}

#include <ReconstructorCUDA/FileReaderCUDA.cuh>
#include <ReconstructorCUDA/ReconstructorCUDA.cuh>
#include "cmdline.h"

int main(int argc, char** argv)
{
    cmdline::parser p;
    p.add<std::string>("leftcam", 'l',"Left camera image folder", true);
    p.add<std::string>("rightcam", 'r',"Right camera image folder", true);
    p.add<std::string>("leftconfig", 'L',"Left camera configuration file", true);
    p.add<std::string>("rightconfig", 'R',"Right camera configuration file", true);
    p.add<std::string>("output", 'o',"Right camera configuration file", true);
    p.add<std::string>("format", 'f',"Suffix of image files, e.g. jpg", true);
    p.add<size_t>("width", 'w',"Projector width", true);
    p.add<size_t>("height", 'h',"Projector height", true);
    p.parse_check(argc, argv);

    LOG::restartLog();
    std::string leftCameraFolder = p.get<std::string>("leftcam");
    std::string rightCameraFolder = p.get<std::string>("rightcam");
    std::string leftConfigFile = p.get<std::string>("leftconfig");
    std::string rightConfigFile = p.get<std::string>("rightconfig");
    std::string output = p.get<std::string>("output");
    std::string suffix = p.get<std::string>("format");

    LOG::restartLog();
    SLS::FileReaderCUDA rightCam("rightCamera");
    SLS::FileReaderCUDA leftCam("leftCamera");

    rightCam.loadImages(rightCameraFolder, suffix);
    leftCam.loadImages(leftCameraFolder, suffix);

    rightCam.loadConfig(rightConfigFile);
    leftCam.loadConfig(leftConfigFile);

    SLS::ReconstructorCUDA reconstruct(p.get<size_t>("width"), p.get<size_t>("height"));
    reconstruct.addCamera(&rightCam);
    reconstruct.addCamera(&leftCam);
    reconstruct.reconstruct();

    auto extension = output.substr(output.find_last_of(".")+1);
    SLS::exportPointCloud(output, extension, reconstruct);
    LOG::writeLog("DONE\n");
    return 0;
}

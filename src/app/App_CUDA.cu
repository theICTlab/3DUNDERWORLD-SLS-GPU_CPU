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

    // Start logging, this function will clear log file and start a new logging.
    LOG::restartLog();

    // Setup left camera image folder
    std::string leftCameraFolder = p.get<std::string>("leftcam");
    // Setup left camera calibration parameters
    std::string leftConfigFile = p.get<std::string>("leftconfig");

    // Setup right camera
    std::string rightCameraFolder = p.get<std::string>("rightcam");
    std::string rightConfigFile = p.get<std::string>("rightconfig");

    std::string output = p.get<std::string>("output");
    std::string suffix = p.get<std::string>("format");

    // Initialize two file readers to load images from file
    SLS::FileReaderCUDA rightCam("rightCamera");
    SLS::FileReaderCUDA leftCam("leftCamera");

    // Load images
    // void loadImages( const std::string &folder, std::string prefix, size_t numDigits, size_t startIdx, std::string suffix )
    rightCam.loadImages(rightCameraFolder, "", 4, 0,suffix);
    leftCam.loadImages(leftCameraFolder, "", 4, 0, suffix);

    // Load configurations, mainly calibration parameters
    rightCam.loadConfig(rightConfigFile);
    leftCam.loadConfig(leftConfigFile);

    // Initialize a reconstructor with the resolution of the projection to project patterns
    SLS::ReconstructorCUDA reconstruct(p.get<size_t>("width"), p.get<size_t>("height"));

    // Add cameras to reconstructor
    reconstruct.addCamera(&rightCam);
    reconstruct.addCamera(&leftCam);

    // Run reconstructio and get the point cloud
    auto pointCloud = reconstruct.reconstruct();
    // Get extension of output file
    auto extension = output.substr(output.find_last_of(".")+1);
    // Export point cloud to file
    pointCloud.exportPointCloud(output, extension);
    LOG::writeLog("DONE\n");
    return 0;
}

#include <core/ImageFileProcessor.h>
#include <core/Reconstructor.h>
#include <core/Log.hpp>
#include <glm/gtx/string_cast.hpp>
#include <core/ReconstructorCPU.h>
#include <memory>
#include "cmdline.h"

int main(int argc, char** argv)
{
    
    // Parsing parameters
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
    SLS::ImageFileProcessor rightCamProcessor("rightCamProcessor");
    SLS::ImageFileProcessor leftCamProcessor("rightCamProcessor");

    // Load images
    // void loadImages( const std::string &folder, std::string prefix, size_t numDigits, size_t startIdx, std::string suffix )
    rightCamProcessor.loadImages(rightCameraFolder, "", 4, 0,suffix);
    leftCamProcessor.loadImages(leftCameraFolder, "", 4, 0, suffix);

    // Load configurations, mainly calibration parameters
    rightCamProcessor.loadConfig(rightConfigFile);
    leftCamProcessor.loadConfig(leftConfigFile);

    // Construct a projector parameter provider
    SLS::Projector proj(p.get<size_t>("width"),p.get<size_t>("height"));

    // Generate reconstruction buckets from image processors.
    // generateBuckets function requires parameter extracted from the `Projector` object,
    // namely projector width, height and number of frames required to reconstruct with the 
    // projected patterns.
    std::vector<SLS::Buckets> bucketsVec
    {
        rightCamProcessor.generateBuckets(proj.getWidth(), proj.getHeight(), proj.getRequiredNumFrames()),
        leftCamProcessor.generateBuckets(proj.getWidth(), proj.getHeight(), proj.getRequiredNumFrames())
    };

    // Initialize a reconstruct
    SLS::ReconstructorCPU reconstruct;

    // Run reconstructio and get the point cloud
    auto pointCloud = reconstruct.reconstruct(bucketsVec);

    // Get extension of output file
    auto extension = output.substr(output.find_last_of(".")+1);

    // Export point cloud to file
    pointCloud.exportPointCloud( p.get<std::string>("output"), extension);

    LOG::writeLog("DONE!\n");
    
    return 0;
}

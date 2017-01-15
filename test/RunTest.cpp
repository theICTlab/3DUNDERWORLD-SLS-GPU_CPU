#include <fstream>
#include <core/FileReader.h>
#include <core/ReconstructorCPU.h>
#include <sstream>
#include <gtest/gtest.h>

const float MAX_DIFF = 0.5;

inline bool comparePlyLine(std::string line1, std::string line2)
{
    if (!std::isdigit(line1[0]) && !std::isdigit(line1[1]))    // if is header
        return line1 == line2;

    std::istringstream ssline1(line1);
    std::istringstream ssline2(line2);
    std::string num1, num2;

    while (std::getline(ssline1, num1, ' ') && std::getline(ssline2, num2, ' '))
        if ( std::fabs( std::stof(num1) - std::stof(num2) ) > MAX_DIFF)
            return false;

    return true;
}

inline bool compareObjLine(std::string line1, std::string line2)
{
    std::istringstream ssline1(line1);
    std::istringstream ssline2(line2);
    std::string num1, num2;

    if (std::getline(ssline1, num1, ' ') && std::getline(ssline2, num2, ' ')) // Skip first split
    {
        while (std::getline(ssline1, num1, ' ') && std::getline(ssline2, num2, ' '))
            if ( std::fabs( std::stof(num1) - std::stof(num2) ) > MAX_DIFF)
                return false;
    }

    return true;
}

bool compareObjFiles(std::string file1, std::string file2)
{
    std::fstream is1, is2;
    std::string buf1, buf2;
    is1.open(file1); is2.open(file2);
    while (std::getline( is1, buf1))
    {
        if (! std::getline(is2, buf2)) return false; // shorter
        if (! compareObjLine(buf1, buf2)) return false;
    }
    if (std::getline(is2, buf2)) return false;  // longer
    return true;
}

bool comparePlyFiles(std::string file1, std::string file2)
{
    std::fstream is1, is2;
    std::string buf1, buf2;
    is1.open(file1); is2.open(file2);
    while (std::getline( is1, buf1))
    {
        if (! std::getline(is2, buf2)) return false; // shorter
        if (! comparePlyLine(buf1, buf2)) return false;
    }
    if (std::getline(is2, buf2)) return false;  // longer
    return true;
}

TEST( RunCPUTest, Arch)
{
    const std::string L_IMGS = std::string(TEST_DATA_PATH) + "/arch/leftCam/dataset1/";
    const std::string L_CFG = std::string(TEST_DATA_PATH) + "/arch/leftCam/calib/output/calib.xml";
    const std::string R_IMGS = std::string(TEST_DATA_PATH) + "/arch/rightCam/dataset1/";
    const std::string R_CFG = std::string(TEST_DATA_PATH) + "/arch/rightCam/calib/output/calib.xml";
    const std::string SUFFIX = "tif";
    const size_t W=1280, H=800;
    const std::string TEST_PLY = std::string(TEST_REF_PATH) + "/arch.ply";
    const std::string TEST_OBJ = std::string(TEST_REF_PATH) + "/arch.obj";
    const std::string O_PLY="arch.ply", O_OBJ="arch.obj";

    auto RC = SLS::FileReader("RightCamera");
    RC.loadImages(R_IMGS, "", 4, 0, SUFFIX);
    RC.loadConfig(R_CFG);

    auto LC = SLS::FileReader("LeftCamera");
    LC.loadImages(L_IMGS, "", 4, 0, SUFFIX);
    LC.loadConfig(L_CFG);

    SLS::ReconstructorCPU rec(W, H);
    rec.addCamera(&LC);
    rec.addCamera(&RC);
    auto pc = rec.reconstruct();

    pc.exportPointCloud( O_PLY, "ply");
    pc.exportPointCloud( O_OBJ, "obj");
    EXPECT_TRUE(compareObjFiles(TEST_OBJ, O_OBJ));
    EXPECT_TRUE(comparePlyFiles(TEST_PLY, TEST_PLY));
}

TEST( RunCPUTest, Alexander)
{
    const std::string L_IMGS = std::string(TEST_DATA_PATH) + "/alexander/leftCam/dataset1/";
    const std::string L_CFG = std::string(TEST_DATA_PATH) + "/alexander/leftCam/calib/output/calib.xml";
    const std::string R_IMGS = std::string(TEST_DATA_PATH) + "/alexander/rightCam/dataset1/";
    const std::string R_CFG = std::string(TEST_DATA_PATH) + "/alexander/rightCam/calib/output/calib.xml";
    const std::string SUFFIX = "jpg";
    const size_t W=1024, H=768;
    const std::string TEST_PLY = std::string(TEST_REF_PATH) + "/alexander.ply";
    const std::string TEST_OBJ = std::string(TEST_REF_PATH) + "/alexander.obj";
    const std::string O_PLY="alexander.ply", O_OBJ="alexander.obj";

    auto RC = SLS::FileReader("RightCamera");
    RC.loadImages(R_IMGS, "", 4, 0, SUFFIX);
    RC.loadConfig(R_CFG);

    auto LC = SLS::FileReader("LeftCamera");
    LC.loadImages(L_IMGS, "", 4, 0, SUFFIX);
    LC.loadConfig(L_CFG);

    SLS::ReconstructorCPU rec(W, H);
    rec.addCamera(&LC);
    rec.addCamera(&RC);
    auto pc = rec.reconstruct();

    pc.exportPointCloud( O_PLY, "ply");
    pc.exportPointCloud( O_OBJ, "obj");
    EXPECT_TRUE(compareObjFiles(TEST_OBJ, O_OBJ));
    EXPECT_TRUE(comparePlyFiles(TEST_PLY, TEST_PLY));
}



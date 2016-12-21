#include <fstream>
#include <core/FileReader.h>
#include <core/ReconstructorCPU.h>
#include "splitstring.h"
#include <gtest/gtest.h>

const float MAX_DIFF = 0.5;

inline bool comparePlyLine(std::string line1, std::string line2)
{
    if (!std::isdigit(line1[0]) && !std::isdigit(line1[1]))    // if is header
        return line1 == line2;

    splitstring split1(line1);
    splitstring split2(line2);
    vector<std::string> splited1 = split1.split(' ');
    vector<std::string> splited2 = split2.split(' ');
    if (splited1.size() > 0 && splited2.size() > 0)
    {
        for (size_t i=0 ; i<splited1.size(); i++)
            if ( std::fabs( std::stof(splited1[i]) - std::stof(splited2[i]) ) > MAX_DIFF)
                return false;
    }
    return true;
}

inline bool compareObjLine(std::string line1, std::string line2)
{
    splitstring split1(line1);
    splitstring split2(line2);
    vector<std::string> splited1 = split1.split(' ');
    vector<std::string> splited2 = split2.split(' ');
    if (splited1.size() > 0 && splited2.size() > 0)
    {
        for (size_t i=1 ; i<splited1.size(); i++)
            if ( std::fabs( std::stof(splited1[i]) - std::stof(splited2[i]) ) > MAX_DIFF)
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
    const std::string L_IMGS = "../../data/arch/leftCam/dataset1/";
    const std::string L_CFG = "../../data/arch/leftCam/calib/output/calib.xml";
    const std::string R_IMGS = "../../data/arch/rightCam/dataset1/";
    const std::string R_CFG = "../../data/arch/rightCam/calib/output/calib.xml";
    const std::string SUFFIX = "tif";
    const size_t W=1280, H=800;
    const std::string TEST_PLY = "../../test/data/arch.ply";
    const std::string TEST_OBJ = "../../test/data/arch.obj";
    const std::string O_PLY="arch.ply", O_OBJ="arch.obj";

    auto RC = SLS::FileReader("RightCamera");
    RC.loadImages(R_IMGS, SUFFIX);
    RC.loadConfig(R_CFG);

    auto LC = SLS::FileReader("LeftCamera");
    LC.loadImages(L_IMGS, SUFFIX);
    LC.loadConfig(L_CFG);

    SLS::ReconstructorCPU rec(W, H);
    rec.addCamera(&LC);
    rec.addCamera(&RC);
    rec.reconstruct();

    SLS::exportPointCloud( O_PLY, "ply", rec);
    SLS::exportPointCloud( O_OBJ, "obj", rec);
    EXPECT_TRUE(compareObjFiles(TEST_OBJ, O_OBJ));
    EXPECT_TRUE(comparePlyFiles(TEST_PLY, TEST_PLY));
    EXPECT_TRUE(true);
}

TEST( RunCPUTest, Alexander)
{
    const std::string L_IMGS = "../../data/alexander/leftCam/dataset1/";
    const std::string L_CFG = "../../data/alexander/leftCam/calib/output/calib.xml";
    const std::string R_IMGS = "../../data/alexander/rightCam/dataset1/";
    const std::string R_CFG = "../../data/alexander/rightCam/calib/output/calib.xml";
    const std::string SUFFIX = "jpg";
    const size_t W=1024, H=768;
    const std::string TEST_PLY = "../../test/data/alexander.ply";
    const std::string TEST_OBJ = "../../test/data/alexander.obj";
    const std::string O_PLY="alexander.ply", O_OBJ="alexander.obj";

    auto RC = SLS::FileReader("RightCamera");
    RC.loadImages(R_IMGS, SUFFIX);
    RC.loadConfig(R_CFG);

    auto LC = SLS::FileReader("LeftCamera");
    LC.loadImages(L_IMGS, SUFFIX);
    LC.loadConfig(L_CFG);

    SLS::ReconstructorCPU rec(W, H);
    rec.addCamera(&LC);
    rec.addCamera(&RC);
    rec.reconstruct();

    SLS::exportPointCloud( O_PLY, "ply", rec);
    SLS::exportPointCloud( O_OBJ, "obj", rec);
    EXPECT_TRUE(compareObjFiles(TEST_OBJ, O_OBJ));
    EXPECT_TRUE(comparePlyFiles(TEST_PLY, TEST_PLY));
    EXPECT_TRUE(true);
}



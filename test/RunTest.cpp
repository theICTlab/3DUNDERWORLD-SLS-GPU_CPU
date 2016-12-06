#include <gtest/gtest.h>
#include <fstream>
#include <core/FileReader.h>
#include <core/ReconstructorCPU.h>
#include "splitstring.h"

inline bool comparePlyLine(std::string line1, std::string line2)
{
    return false;
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
            if ( std::fabs( std::stof(splited1[i]) - std::stof(splited2[i]) ) > 0.001)
                return false;
    }
    return true;
}
bool compareFiles(std::string file1, std::string file2)
{
    std::fstream is1, is2;
    std::string buf1, buf2;
    is1.open(file1); is2.open(file2);
    while (std::getline( is1, buf1))
    {
        if (! std::getline(is2, buf2)) return false; // sorter
        if (! compareObjLine(buf1, buf2)) return false;
    }
    if (std::getline(is2, buf2)) return false;  // longer
    return true;
}

bool compareOBJs(std::string file1, std::string file2)
{
}
TEST( SLS_CPU, Arc)
{
    const std::string L_IMGS = "../../data/arc/leftCam/dataset1/";
    const std::string L_CFG = "../../data/arc/leftCam/calib/output/calib.xml";
    const std::string R_IMGS = "../../data/arc/rightCam/dataset1/";
    const std::string R_CFG = "../../data/arc/rightCam/calib/output/calib.xml";
    const std::string SUFFIX = "tif";
    const size_t W=1280, H=800;
    const std::string TEST_PLY = "../../test/data/arc.ply";
    const std::string TEST_OBJ = "../../test/data/arc.obj";
    const std::string O_PLY="arc.ply", O_OBJ="arc.obj";

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
    //EXPECT_TRUE(compareFiles(TEST_PLY, O_PLY));
    EXPECT_TRUE(compareFiles(TEST_OBJ, O_OBJ));

}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

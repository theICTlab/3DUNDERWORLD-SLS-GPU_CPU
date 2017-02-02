#include <core/PointCloud.hpp>
#include <cstring> // memcpy
#include <cmath>
#include <array>

namespace SLS
{
    void exportPointCloud2OBJ(std::string fileName, const std::vector<float> &pointCloud)
    {
        LOG::startTimer("Exporting OBJ to %s ... ", fileName.c_str());
        std::ofstream of(fileName, std::ofstream::out);
        for (size_t i=0; i<pointCloud.size(); i+=6)
            if ( pointCloud[i] != 0)
                of<<"v "<<pointCloud[i]<<" "<<pointCloud[i+1]<<" "<<pointCloud[i+2]<<std::endl;
        of.close();
        LOG::endTimer();
    }

    void exportPointCloud2PLY(std::string fileName, const std::vector<float> &pointCloud)
    {
        LOG::startTimer("Exporting PLY to %s ... ", fileName.c_str());
        std::ofstream of(fileName, std::ofstream::out);
        std::stringstream headerSS;
        std::stringstream dataSS;

        size_t pointCount = 0;
        // Writing data
        for (size_t i=0; i<pointCloud.size(); i+=6)
        {
            if ( pointCloud[i] != 0)
            {
                std::array<float, 6> p;
                memcpy (p.data(), &pointCloud[i], sizeof(float) * 6);
                float sum = 0.0;
                for (const auto &elem : p)
                    sum += std::abs(elem);
                if (std::abs(sum - 0.0) < 0.0000001)
                    continue;
                dataSS<<pointCloud[i+0]<<" "<<pointCloud[i+1]<<" "<<pointCloud[i+2]<<" "   
                    <<(int)pointCloud[i+3]<<" "<<(int)pointCloud[i+4]<<" "<<(int)pointCloud[i+5]<<std::endl;
                pointCount++;
            }
        }
        // Writing Headers
        headerSS <<"ply\n"<<
            "format ascii 1.0\n"<<
            "element vertex "<<pointCount<<std::endl<<
            "property float x\n"<<
            "property float y\n"<<
            "property float z\n"<<
            "property uchar red\n"<<
            "property uchar green\n"<<
            "property uchar blue\n"<<
            //"property list uchar int vertex_indices\n"<<
            "end_header\n";

        of<<headerSS.rdbuf();
        of<<dataSS.rdbuf();
        of.close();

        LOG::endTimer();   LOG::startTimer("Exporting PLY to %s ... ", fileName.c_str());
    }


} // namespace SLS

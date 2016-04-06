#include "Reconstructor.h"
#include <fstream>

namespace SLS
{
    class Reconstructor;
    void exportPLY( std::string fileName ,const Reconstructor& reconstructor)
    {
        LOG::startTimer("Exporting PLY to %s ... ", fileName.c_str());

        std::ofstream of(fileName, std::ofstream::out);
        const auto &pointCloud = reconstructor.pointCloud_;
        // Writing Headers
        of <<"ply\n"<<
            "format ascii 1.0\n"<<
            "element vertex "<<pointCloud.size()/6<<std::endl<<
            "property float x\n"<<
            "property float y\n"<<
            "property float z\n"<<
            "property float red\n"<<
            "property float geen\n"<<
            "property float blue\n"<<
            //"property list uchar int vertex_indices\n"<<
            "end_header\n";
        // Writing vertex list
        // Writing face list
        for (size_t i=0; i<pointCloud.size(); i+=6)
            of<<pointCloud[i+0]<<" "<<pointCloud[i+1]<<" "<<pointCloud[i+2]<<" "
            <<pointCloud[i+3]<<" "<<pointCloud[i+4]<<" "<<pointCloud[i+5]<<std::endl;
        of.close();

        LOG::endTimer();
    }
    void exportOBJ( std::string fileName ,const Reconstructor& reconstructor)
    {
        LOG::startTimer("Exporting PLY to %s ... ", fileName.c_str());

        std::ofstream of(fileName, std::ofstream::out);
        const auto &pointCloud = reconstructor.pointCloud_;
        for (size_t i=0; i<pointCloud.size(); i+=6)
            of<<"v "<<pointCloud[i]<<" "<<pointCloud[i+1]<<" "<<pointCloud[i+2]<<std::endl;
        of.close();

        LOG::endTimer();
    }
}

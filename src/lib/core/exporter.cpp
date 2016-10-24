#include "exporter.hpp"
#include <fstream>

namespace SLS
{
    void exportPLYGrid(std::string fileName, const Reconstructor& reconstructor)
    {
        LOG::startTimer("Exporting PLY Grid to %s ... ", fileName.c_str());
        std::ofstream of(fileName, std::ofstream::out);
        const auto &pointCloud = reconstructor.pointCloud_;
        //Writing headers
        of<<"ply\n";
        of<<"format ascii 1.0\n";
		of<<"obj_info is_interlaced 0\n";
		of<<"obj_info num_cols " << reconstructor.projector_->getHeight() << "\n";
		of<<"obj_info num_rows " << reconstructor.projector_->getWidth() << "\n";
        of<<"element vertex " << pointCloud.size()/4 << "\n";
        of<<"property float x\n";
        of<<"property float y\n";
        of<<"property float z\n";
		of<<"element range_grid " << reconstructor.projector_->getHeight() *
            reconstructor.projector_->getWidth()<< "\n";
        LOG::endTimer();
        of<<"property list uchar int vertex_indices\n";
        of<<"end_header\n";
        for (size_t i=0; i<pointCloud.size(); i+=4)
        {
            of<<pointCloud[i+0]<<" "<<pointCloud[i+1]<<" "<<pointCloud[i+2]<<std::endl;
        }
        std::cout<<"PointCloud size: "<<pointCloud.size()<<std::endl;
        std::vector<bool> ranged_grid(reconstructor.projector_->getHeight()*reconstructor.projector_->getWidth(), false);
        for (size_t i=0; i<pointCloud.size(); i+=4)
        {
            unsigned clmBasedIdx = i/4;
            unsigned x = clmBasedIdx / reconstructor.projector_->getHeight();
            unsigned y = clmBasedIdx % reconstructor.projector_->getHeight();
            unsigned rowBasedIdx = x+y*reconstructor.projector_->getWidth();
            if (pointCloud[i+3] != 0)
                ranged_grid[rowBasedIdx] = true;
            else
                ranged_grid[rowBasedIdx] = false;
        }
        for (size_t i=0; i<ranged_grid.size(); i++)
            if (ranged_grid[i])
                of << "1 "<<i<<std::endl;
            else
                of<<"0"<<std::endl;
    }

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
        LOG::startTimer("Exporting OBJ to %s ... ", fileName.c_str());

        std::ofstream of(fileName, std::ofstream::out);
        const auto &pointCloud = reconstructor.pointCloud_;
        for (size_t i=0; i<pointCloud.size(); i+=6)
            of<<"v "<<pointCloud[i]<<" "<<pointCloud[i+1]<<" "<<pointCloud[i+2]<<std::endl;
        of.close();

        LOG::endTimer();
    }

    void exportOBJVec4( std::string fileName ,const Reconstructor& reconstructor)
    {
        LOG::startTimer("Exporting OBJ vec4 to %s ... ", fileName.c_str());

        std::ofstream of(fileName, std::ofstream::out);
        const auto &pointCloud = reconstructor.pointCloud_;
        for (size_t i=0; i<pointCloud.size(); i+=4)
        {
            if (pointCloud[i+3] < 0.5 ) continue;
            of<<"v "<<pointCloud[i]<<" "<<pointCloud[i+1]<<" "<<pointCloud[i+2]<<std::endl;
        }
        of.close();

        LOG::endTimer();
    }


    // Refactoring export functions
    

void exportPointCloud(std::string fileName, std::string type, const Reconstructor& reconstructor)
{
    std::hash<std::string> hashFunc;
    switch (hashFunc(type)){
        case (const size_t)15537605204898678351U: // "OBJ"
        case (const size_t)7208462448059259249U:  // "obj"
            exportPointCloud2OBJ(fileName, reconstructor.pointCloud_);
            break;
        default:
            std::cerr<<type<<" is not supported\n";
            break;

    }
}
    void exportPointCloud2OBJ(std::string fileName, const std::vector<float> &pointCloud)
    {
        LOG::startTimer("Exporting OBJ to %s ... ", fileName.c_str());
        std::ofstream of(fileName, std::ofstream::out);
        for (size_t i=0; i<pointCloud.size(); i+=6)
            of<<"v "<<pointCloud[i]<<" "<<pointCloud[i+1]<<" "<<pointCloud[i+2]<<std::endl;
        of.close();
        LOG::endTimer();
    }
}

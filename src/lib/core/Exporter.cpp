#include "Exporter.hpp"
#include <fstream>
#include <sstream>

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

// Refactoring export functions

void exportPointCloud(std::string fileName, std::string type, const Reconstructor& reconstructor)
{
    std::hash<std::string> hashFunc;
    switch (hashFunc(type)){
        case (const size_t)15537605204898678351U: // "OBJ"
        case (const size_t)7208462448059259249U:  // "obj"
            exportPointCloud2OBJ(fileName, reconstructor.pointCloud_);
            break;
        case (const size_t)9263517021921177413U:    //"PLY"
        case (const size_t)15116060116567580642U:   //"ply"
            exportPointCloud2PLY(fileName, reconstructor.pointCloud_);
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

void exportPointCloud2PLY(std::string fileName, const std::vector<float> &pointCloud)
{
    LOG::startTimer("Exporting PLY to %s ... ", fileName.c_str());
    std::ofstream of(fileName, std::ofstream::out);
    std::stringstream headerSS;
    std::stringstream dataSS;
  
    size_t pointCount = 0;
    for (size_t i=0; i<pointCloud.size(); i+=6)
    {
        std::array<float, 6> p;
        memcpy ( p.data(), &pointCloud[i], sizeof(float) * 6);
        float sum = 0.0;
        for (const auto &elem : p)
            sum += abs(elem);
        if (abs(sum - 0.0) < 0.0000001)
            continue;
        dataSS<<pointCloud[i+0]<<" "<<pointCloud[i+1]<<" "<<pointCloud[i+2]<<" "   
            <<(int)pointCloud[i+3]<<" "<<(int)pointCloud[i+4]<<" "<<(int)pointCloud[i+5]<<std::endl;
        pointCount++;
    }
  // Writing Headers
    headerSS <<"ply\n"<<
        "format ascii 1.0\n"<<
        "element vertex "<<pointCount<<std::endl<<
        "property float x\n"<<
        "property float y\n"<<
        "property float z\n"<<
        "property uchar red\n"<<
        "property uchar geen\n"<<
        "property uchar blue\n"<<
        //"property list uchar int vertex_indices\n"<<
        "end_header\n";

    of<<headerSS.rdbuf();
    of<<dataSS.rdbuf();
    of.close();

    LOG::endTimer();   LOG::startTimer("Exporting PLY to %s ... ", fileName.c_str());
}
    
}

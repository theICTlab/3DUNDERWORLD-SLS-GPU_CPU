#pragma once
#include <string>
#include <vector>
#include <core/Reconstructor.h>
namespace SLS
{
class Reconstructor;
void exportPLYGrid(std::string fileName, const Reconstructor& reconstructor);
void exportPLY( std::string fileName ,const Reconstructor& reconstructor);
void exportOBJ( std::string fileName ,const Reconstructor& reconstructor);
void exportOBJVec4( std::string fileName ,const Reconstructor& reconstructor);

void exportPointCloud(std::string fileName, std::string type, const Reconstructor& reconstructor);
void exportPointCloud2OBJ(std::string fileName, const std::vector<float> &pointCloud);
} // namespace SLS

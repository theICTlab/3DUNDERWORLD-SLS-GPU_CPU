#pragma once
#include <core/fileReader.h>
#include <string>

namespace SLS
{
class Calibrator
{
public:
    static void Calibrate(FileReader *cam, const std::string& calibImgsDir, const std::string &calibFile);
};
} // namespace SLS

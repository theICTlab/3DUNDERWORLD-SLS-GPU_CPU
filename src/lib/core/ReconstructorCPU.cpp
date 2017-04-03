#include "ReconstructorCPU.h"
#include <iomanip>
#include <limits>
#include "ImageFileProcessor.h"

namespace SLS {

std::array<glm::vec3, 2> ReconstructorCPU::intersectionOfBucket_(
    const Bucket& firstBucket, const Bucket& secondBucket)
{
    // for each camera pixels in two buckets
    size_t pointCount = 0;
    glm::vec3 averagePosition(0.0);
    glm::vec3 averageColor(0.0);
    for (const auto &ray0 : firstBucket)
        for (const auto &ray1 : secondBucket) {
            float dist = -1.0f;
            glm::vec4 midP = midPoint(ray0, ray1, dist);
            if (dist > 0.0) {
                pointCount++;
                averagePosition += glm::vec3(midP);
                averageColor += (ray0.color + ray1.color) / 2.0f;
            }
        }
    if (pointCount != 0)
        return std::array<glm::vec3, 2>{averagePosition / (float)pointCount,
                                        averageColor / (float)pointCount};
    else
        return std::array<glm::vec3, 2>{glm::vec3(0.0), glm::vec3(0.0)};
}

std::array<glm::vec3, 2> ReconstructorCPU::intersectionOfBucketMinDist_(
    const Bucket &firstBucket, const Bucket &secondBucket)
{
    // for each camera pixels in two buckets
    glm::vec3 minPosition(0.0);
    glm::vec3 minColor(0.0);
    float minDist = std::numeric_limits<float>::max();
    for (const auto &ray0 : firstBucket)
        for (const auto &ray1 : secondBucket) {
            float dist = -1.0f;
            glm::vec4 midP = midPoint(ray0, ray1, dist);
            if (dist > 0.0 && dist < minDist) {
                minDist = dist;
                minPosition = glm::vec3(midP);
                minColor = (ray0.color + ray1.color / 2.0f);
            }
        }
    return std::array<glm::vec3, 2>{minPosition, minColor};
}

PointCloud ReconstructorCPU::reconstruct(
    const std::vector<Buckets>& bucketsArray)
{
    PointCloud res;
    LOG::startTimer();
    for (size_t i = 0; i < bucketsArray[0].size(); i++) {
        std::array<glm::vec3, 2> point =
            intersectionOfBucket_(bucketsArray[0][i], bucketsArray[1][i]);
        if (glm::all(glm::equal(point[0], glm::vec3(0.0))) &&
            glm::all(glm::equal(point[1], glm::vec3(0.0))))
            continue;
        else
            res.pushPoint(point[0], point[1]);
    }
    LOG::endTimer("Finished reconstruction in ");
    return res;
}

}  // namespace SLS

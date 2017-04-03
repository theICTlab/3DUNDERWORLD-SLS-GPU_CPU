#pragma once
#include <core/PointCloud.hpp>
#include <memory>
#include "Log.hpp"
#include "Reconstructor.h"
namespace SLS {

// CPU reconstructor
/*! Reconstruct point cloud from buckets provided by image processor.
 */
class ReconstructorCPU : public Reconstructor {
private:

    //! Find intersection of two projector pixel
    /*! Each projector pixel contains multiple camera pixels.
     * This function takes two buckets from two cameras. Shoot a ray from each
     * pixel to find a pair of rays with minimum distance. In this function, the
     * position of the pixel is calculated by the average position and color of
     * all pairs of rays in a projector pixel.
     *
     * The retrun value is put in a std::array<glm::vec3, 2>, in which the first
     * vec3 is the position and the second is the color
     * \param bucket0 First bucket
     * \param bucket1 Another bucket
     * \return midpoint and color
     */
    std::array<glm::vec3, 2> intersectionOfBucket_(const Bucket& firstBucket,
                                                   const Bucket& secondBucket);

    /*! Similar to intersectionOfBucket_(), this function finds pair of rays
     * with minimu distance to reconstruct the depth.
     *
     * This function is not used for now since the average method yield more
     * structural result.
     */
    std::array<glm::vec3, 2> intersectionOfBucketMinDist_(
        const Bucket& firstBucket, const Bucket& secondBucket);

public:

    /**
     * !Synopsis  
     *
     * \param bucketsArray buckets for reconstruction
     *
     * \returns Point cloud
     */
    PointCloud reconstruct(const std::vector<Buckets>& bucketsArray) override;
};
}  // namespace SLS

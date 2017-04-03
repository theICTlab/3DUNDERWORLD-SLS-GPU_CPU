/* Copyright (C)
 * 2016 - Tsing Gu
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 */

#pragma once
#include <array>
#include <glm/glm.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "DynamicBitset.h"
#include "Ray.h"

namespace SLS {

using Bucket = std::vector<Ray>;
using Buckets = std::vector<Bucket>;

/*! Base class of image processor
 *
 * The image processor includes the full acquisition pipeline.
 * ```
 * +-------+  +-----------+ +--------+   +--------+   +--------+
 * | Get   |  |Undistort  | |Compute |   |Cast    |   |Genearte|
 * | Image +-->Image      +->Mask    +--->Ray     +--->Buckets |
 * +-------+  +-----------+ +--------+   +--------+   +--------+
 * ```
 *
 * * The image acquisition is implemented as a forward iterator (getNextFrame)
 * to
 * be compatible with image cameras, which can only take a new image.
 *
 * * Image undistortion happens soon after a image is imported.
 *
 * * The mask is used to discard invalid pixels and computed by a contrast
 * threshold. When the contrast of a pixel in lit and dark images is smaller
 * than a threshold, the pixel is considered invalid and would be `0` in the
 * mask.
 *
 * * The camera casts a ray into the 3D space based on the intrinsic
 * parameters.
 *
 * * A buckets of rays is generated to pass to reconstructor.
 *
 * The Buckets is the data structure to hash the ray into projector pixel indices.
 *
 * ```
 * ProjPixels(Buckets)   Rays
 * +------------------+  +---+---+---+
 * |       0          +->+R0 |R1 |R2 |
 * +------------------+  +-------+---+
 * |       1          +->+R3 |
 * +------------------+  +----
 *
 * +------------------+  +---+
 * |       n          +->+Rm |
 * +------------------+  +---+
 *
 * ```
 */
class ImageProcessor {
protected:
    // Name of camera
    std::string name_;  //!< name of the camera

    // Camera Resolution
    size_t resX_, resY_;

    // Mask of valid pixels
    Dynamic_Bitset shadowMask_;  //!< Color mask

    cv::Mat litImage_;  //!< An all lit image contains color information of
                        //!reconstruction object

    uchar whiteThreshold_;  //!< Thresholds are used to filter out invalid pixel.

    // If the contrast of a pixel between lit and unlit is smaller than the
    // dark threshold, the pixel is invalid.
    uchar blackThreshold_;  //!< Contrast threshold

    // adaptive Thresholds of each pixel.
    // The threshold is used filter out invalid pixels
    std::vector<unsigned char>
        thresholds_;  //!< Threshold for each pixel with in [0,255],
public:
    ImageProcessor() = delete;

    /*
     * Construction of a camera, it takes the name and the resolution
     * of the camera.
     */
    explicit ImageProcessor(const std::string &cName)
        : name_(cName), resX_(0), resY_(0)
    {
        whiteThreshold_ = 5;
        blackThreshold_ = 40;
    }  // Hack, need to read from file

    const std::string &getName() const { return name_; }
    void setName(const std::string &cName) { name_ = cName; }
    void getResolution(size_t &x, size_t &y) const
    {
        x = resX_;
        y = resY_;
    }

    const unsigned char &getThreshold(const size_t &idx)
    {
        return thresholds_[idx];
    }

    // Get a mask by index
    bool queryMask(const size_t &idx) { return shadowMask_.getBit(idx); }
    // Set the value of mask to one at given index
    void setMask(const size_t &idx) { shadowMask_.setBit(idx); }
    // Clear the value of mask to one at given index
    void clearMask(const size_t &idx) { shadowMask_.clearBit(idx); }
    uchar getWhiteThreshold() const { return whiteThreshold_; }
    uchar getblackThreshold() const { return blackThreshold_; }
    virtual const cv::Mat &getColorFrame() const { return litImage_; }
    glm::vec3 getColor(size_t x, size_t y) const
    {
        auto color = litImage_.at<cv::Vec3b>(y, x);
        // OpenCV BGR
        float b = (float)color.val[0];
        float g = (float)color.val[1];
        float r = (float)color.val[2];
        return vec3(r, g, b);
    }

    glm::vec3 getColor(size_t idx) const
    {
        return getColor(idx / resY_, idx % resY_);
    }

    // Interfaces
    /**
    *!  Get a ray in world space by given pixel
    * \param x x coordinate of pixel
    * \param y y coordinate of pixel
    *
    * \return Ray shot from camera to this pixel
    */
    virtual Ray getRay(const size_t &x, const size_t &y) = 0;

    /**
     * ! Get a ray by pixel index.
     *
     * \param idx pixel index
     *
     * \returns Ray shot from the pixel.
     */
    virtual Ray getRay(const size_t &idx) = 0;

    virtual void setResolution(const size_t &x, const size_t &y)
    {
        resX_ = x;
        resY_ = y;
    }

    virtual ~ImageProcessor();

    // Load camera intrinsic/extrinsic parameters and distortion from file
    virtual void loadConfig(const std::string &configFile) = 0;

    // Get next frame from camera
    virtual const cv::Mat &getNextFrame() = 0;

    // Undistort image using the distortion parameter.
    virtual void undistort() = 0;

    // Compute masks and thresholds ( if dynamic threshold enabled
    virtual void computeShadowsAndThresholds() = 0;

    /**
     * ! Generate bucket
     *
     * \param bucketSize number of buckets
     *
     * \returns A buckets filled with pixel indices
     */
    virtual Buckets generateBuckets(size_t projWidth, size_t projHeight,
                                    size_t requiredNumFrames) = 0;
};
}  // namespace SLS

#include <iomanip>
#include "ReconstructorCPU.h"
#include "FileReader.h"
#include <limits>

namespace SLS{

ReconstructorCPU::~ReconstructorCPU()
{
    delete projector_;
}
void ReconstructorCPU::initBuckets()
{
    size_t x,y;
    projector_->getSize(x,y);
    buckets_.resize(cameras_.size());
    for (auto& b: buckets_)
        b.resize(x*y);
    generateBuckets();
}

void ReconstructorCPU::addCamera(Camera *cam)
{
    cameras_.push_back(cam);
}

void ReconstructorCPU::generateBuckets()
{
    //Generating reconstruction bucket for each camera
    //for each camera
    for (size_t camIdx = 0; camIdx < cameras_.size(); camIdx++) {
        const auto &cam = cameras_[camIdx];
        LOG::writeLog("Generating reconstruction bucket for \"%s\" ... \n", cam->getName().c_str());

        cam->computeShadowsAndThresholds();

        size_t x = 0, y = 0, xTimesY = 0;
        cam->getResolution(x, y);
        xTimesY = x * y;

        // For each camera pixel
        for (size_t i = 0; i < xTimesY; i++) {
            if (!cam->queryMask(i)) // No need to process if in shadow
                continue;

            cam->getNextFrame();
            cam->getNextFrame(); //skip first two frames

            Dynamic_Bitset bits(projector_->getRequiredNumFrames());

            bool discard = false;

            // for each frame
            for (int bitIdx = projector_->getRequiredNumFrames() - 1; bitIdx >= 0; bitIdx--) {

                auto frame = cam->getNextFrame();
                auto invFrame = cam->getNextFrame();
                unsigned char pixel = frame.at<uchar>(i % y, i / y);
                unsigned char invPixel = invFrame.at<uchar>(i % y, i / y);

                // Not considering shadow mask. But the following test should be
                // more strict than shadow mask.
                if (invPixel > pixel && invPixel - pixel >= ((FileReader*)cam)->getWhiteThreshold(i)) {
                    // No need to do anything since the array is initialized as all zeros
                    bits.clearBit((size_t) bitIdx);
                    continue;
                }
                else if (pixel > invPixel && pixel - invPixel > 
                        ((FileReader*)cam)->getWhiteThreshold(i)) {
                    bits.setBit((size_t )bitIdx);

                }
                else {
                    cam->clearMask(i);
                    discard = true;
                    continue;
                }
            } // end for each frame

            // if the pixel is valid, add to bucket.
            if (!discard) 
            {
                auto vec2Idx = bits.to_uint_gray();
                if ( projector_->getWidth() > vec2Idx.x &&
                        vec2Idx.y < projector_->getHeight()) {
                    buckets_[camIdx][vec2Idx.x * projector_->getHeight() + vec2Idx.y].push_back(i);
                }
            }
        }
        // Keep track of maximum number of pixels in a grid
        //unsigned  maxCount = 0;
        //for (const auto &bucket: buckets_[camIdx]){
        //    if (bucket.size() > maxCount)
        //        maxCount = bucket.size();
        //}
    }

}

PointCloud ReconstructorCPU::reconstruct()
{

    PointCloud res;
    size_t x,y;
    cameras_[0]->getResolution(x,y);
    initBuckets();
    LOG::startTimer();
    for ( size_t i=0; i<buckets_[0].size(); i++)
    {
        const auto &cam0bucket = buckets_[0][i];
        const auto &cam1bucket = buckets_[1][i];
        size_t minCam0Idx=0;
        size_t minCam1Idx=0;
        if ((!cam0bucket.empty()) && (!cam1bucket.empty()))
        {
            float minDist=std::numeric_limits<float>::max();
            glm::vec4 minMidP(0.0f);

            float ptCount = 0.0;
            glm::vec4 midPointAvg(0.0f);

            for (const auto& cam0P: cam0bucket)
                for (const auto& cam1P: cam1bucket)
                {
                    float dist=-1.0f;
                    
                    auto midP=midPoint(cameras_[0]->getRay(cam0P), cameras_[1]->getRay(cam1P), dist);
                    if (dist > 0.0) // if dist is valid
                    {
                        ptCount += 1.0;
                        midPointAvg += midP;
                        if (dist < minDist)
                        {
                            minDist = dist;
                            minMidP = midP;
                            minCam0Idx = cam0P;
                            minCam1Idx = cam1P;
                        }
                    }
                }
            midPointAvg = midPointAvg/ptCount;
            {
                auto color0 = cameras_[0]->getColor(minCam0Idx);
                auto color1 = cameras_[1]->getColor(minCam1Idx);
                res.pushPoint( glm::vec3(midPointAvg), (color0 + color1) / 2.0f);
            }
        }
    }
    LOG::endTimer("Finished reconstruction in ");
    return res;
}


} // namespace SLS

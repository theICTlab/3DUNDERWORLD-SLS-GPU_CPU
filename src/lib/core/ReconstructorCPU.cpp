#include <iomanip>
#include "ReconstructorCPU.h"
#include "fileReader.h"
#include <limits>

namespace SLS{

ReconstructorCPU::~ReconstructorCPU()
{
    for (auto &cam: cameras_)
        delete cam;
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
        //assert(dynamic_cast<FileReader*>(cam)->getNumFrames() == projector_->getRequiredNumFrames()*2+2);

        for (size_t i = 0; i < xTimesY; i++) {
            if (!cam->queryMask(i)) // No need to process if in shadow
                continue;

            cam->getNextFrame();
            cam->getNextFrame();//skip first two frames

            Dynamic_Bitset bits(projector_->getRequiredNumFrames());

            bool discard = false;
            // for each frame
            for (int bitIdx = projector_->getRequiredNumFrames() - 1; bitIdx >= 0; bitIdx--) {
            //for (int bitIdx = 0; bitIdx <projector_->getRequiredNumFrames(); bitIdx++) {

                //std::cout<<((FileReader*)cam)->getCurrentIdx()<<std::endl;
                auto frame = cam->getNextFrame();
                //std::cout<<((FileReader*)cam)->getCurrentIdx()<<std::endl;
                auto invFrame = cam->getNextFrame();
                unsigned char pixel = frame.at<uchar>(i % y, i / y);
                unsigned char invPixel = invFrame.at<uchar>(i % y, i / y);

                // Not considering shadow mask. But the following test should be
                // more strict than shadow mask.
                if (invPixel > pixel && invPixel - pixel >= ((FileReader*)cam)->getWhiteThreshold(i)) {
                    // No need to do anything since the array is initialized as all zeros
                    bits.clearBit((size_t) bitIdx);
                    continue;
                    //std::cout<<"-----\n"<<bits<<std::endl;
                    //bits.clearBit(bitIdx);
                    //std::cout<<std::setw(bits.size()-bitIdx)<<"c"<<std::endl;
                    //std::cout<<bits<<std::endl;
                   
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
            if (!discard) {

                auto vec2Idx = bits.to_uint_gray();
                if ( projector_->getWidth() > vec2Idx.x &&
                        vec2Idx.y < projector_->getHeight()) {
                    buckets_[camIdx][vec2Idx.x * projector_->getHeight() + vec2Idx.y].push_back(i);
                }
            }
        }
        unsigned  maxCount = 0;
        for (const auto &bucket: buckets_[camIdx]){
            if (bucket.size() > maxCount)
                maxCount = bucket.size();
        }
        std::cout<<"Max count = "<<maxCount<<std::endl;
    }

}

void ReconstructorCPU::reconstruct()
{
    size_t x,y;
    cameras_[0]->getResolution(x,y);
    initBuckets();

    LOG::startTimer();

    for ( size_t i=0; i<buckets_[0].size(); i++)     // 1024*768
    {
        //Progress
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
                    
                    auto midP=midPointBkp(cameras_[0]->getRay(cam0P), cameras_[1]->getRay(cam1P), dist);
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
                pointCloud_.push_back(midPointAvg.x);
                pointCloud_.push_back(midPointAvg.y);
                pointCloud_.push_back(midPointAvg.z);
                //pointCloud_.push_back(1);
                unsigned char r0, g0, b0;
                unsigned char r1, g1, b1;
                cameras_[0]->getColor(minCam0Idx, r0, g0, b0);
                cameras_[1]->getColor(minCam1Idx, r1, g1, b1);
                pointCloud_.push_back((r0+r1)/2);
                pointCloud_.push_back((g0+g1)/2);
                pointCloud_.push_back((b0+b1)/2);
            }
        }
    }
    LOG::endTimer("Finished reconstruction in ");
}


} // namespace SLS

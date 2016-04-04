#include "ReconstructorCPU.h"
#include <cassert>
#include "fileReader.h"
#include <fstream>
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
        b.resize(1 << projector_->getRequiredNumFrames());
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
    //for (auto &cam : cameras_)
    for (size_t camIdx = 0; camIdx < cameras_.size(); camIdx++)
    {
        auto &cam = cameras_[camIdx];
        LOG::writeLog("Generating reconstruction bucket for \"%s\" ... \n", cam->getName().c_str());

        //cam->undistort();
        cam->computeShadowsAndThreasholds();

        size_t x=0,y=0,xTimesY=0;
        cam->getResolution(x,y);
        xTimesY=x*y;

        // For each camera pixle
        //assert(dynamic_cast<FileReader*>(cam)->getNumFrames() == projector_->getRequiredNumFrames()*2+2);

        for ( size_t i=0; i<xTimesY; i++)
        {
            if (! cam->queryMask(i) )
                continue;
            cam->nextFrame();cam->nextFrame();//skip first two frames
            size_t bitIdx = 0;

            Dynamic_Bitset bits(projector_->getRequiredNumFrames());

            bool discard=false;

            // for each image
            while(bitIdx < projector_->getRequiredNumFrames()/2)
            {

                auto frame = cam->getNextFrame();
                auto invFrame = cam->getNextFrame();
                auto pixel = frame.at<uchar>(i%y,i/y);
                auto invPixel = invFrame.at<uchar>(i%y,i/y);



                // Not considering shadow mask. But the following test should be
                // more strict than shadow mask.
                if (invPixel > pixel && invPixel-pixel >= cam->getThreashold(i))
                    bits.clearBit(bitIdx);
                else if (pixel > invPixel && pixel-invPixel > cam->getThreashold(i))
                    bits.setBit(bitIdx);
                else
                {
                    //discard this pixel
                    discard=true;
                    break;
                }
                bitIdx++;
            }
            if (!discard)
                buckets_[camIdx][bits.to_uint()].push_back(i);
        }
    }

    //Sanity check
    //for (size_t camIdx = 0; camIdx < cameras_.size(); camIdx++)
    //{
    //    for (size_t i=0; i< buckets_[camIdx].size(); i++)
    //    {
    //        for (size_t j=0; j < buckets_[camIdx][i].size(); j++)
    //        {
    //            glm::vec2 p0(buckets_[camIdx][i][j]/4896, buckets_[camIdx][i][j]%4896);
    //            for (size_t k=0; k < buckets_[camIdx][i].size(); k++)
    //            {
    //                glm::vec2 p1(buckets_[camIdx][i][k]/4896, buckets_[camIdx][i][j]%4896);
    //                float dist = glm::distance(p0, p1);
    //                if (dist > 100)
    //                    LOG::writeLog("Huge distance %f\n", dist);
    //            }
    //        }
    //    }
    //}
}

void ReconstructorCPU::renconstruct()
{
    /* Assuming there are two cameras 
     * For each projector pixel,
     * find the min dis
     * */
    size_t x,y;
    cameras_[0]->getResolution(x,y);
    initBuckets();

    LOG::startTimer();
    std::ofstream of("test.obj");

    for ( size_t i=0; i<buckets_[0].size(); i++)
    {
        if (!buckets_[1][i].empty() && !buckets_[0][i].empty())
        {
            const auto &cam0bucket = buckets_[0][i];
            const auto &cam1bucket = buckets_[1][i];
            //size_t cam0=buckets_[0][i][0];
            //size_t cam1=buckets_[1][i][0];
            float minDist=9999999999.0;
            glm::vec4 minMidP(0.0f);
            //Refinement

            for (const auto& cam0P: cam0bucket)
                for (const auto& cam1P: cam1bucket)
                {
                    float dist;
                    Ray ray0 = cameras_[0]->getRay(cam0P);
                    Ray ray1 = cameras_[1]->getRay(cam1P);

                    if (glm::length(ray0.dir) < 0.5 || glm::length(ray1.dir) < 0.5) continue;
                    
                    auto midP=midPointBkp(cameras_[0]->getRay(cam0P), cameras_[1]->getRay(cam1P), dist);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        minMidP = midP;
                    }
                }
            of<<"v "<<minMidP.x<<" "<<minMidP.y<<" "<<minMidP.z<<std::endl;
        }
    }
    of.close();
    LOG::endTimer("Finished reconstruction in ");
}


} // namespace SLS

#include "ReconstructorCPU.h"
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
    /*
     * For each camera
     *  Undistort
     *  Calculate mask
     *  Put pixel into projector bucket
     * For each projector pixel
     *  calculate the minimum distance of pixel within it
     *  output result
     */
    //Generating reconstruction bucket for each camera
    //for each camera
    //for (auto &cam : cameras_)
    for (size_t camIdx = 0; camIdx < cameras_.size(); camIdx++)
    {
        auto &cam = cameras_[camIdx];
        LOG::writeLog("Generating reconstruction for %s\n", cam->getName().c_str());
        cam->undistort();
        cam->computeShadowsAndThreasholds();
        size_t x=0,y=0,xTimesY=0;
        cam->getResolution(x,y);
        xTimesY=x*y;
        //For each pixle
        for ( size_t i=0; i<xTimesY; i++)
        {
            if (! cam->queryMask(i) )
                continue;
            cam->nextFrame();cam->nextFrame();//skip first two frames
            size_t bitIdx = 0;
            Dynamic_Bitset bits(projector_->getRequiredNumFrames());
            bool discard=false;
            // for each image
            while(bitIdx < projector_->getRequiredNumFrames())
            {
                auto pixel = cam->getNextFrame().at<uchar>(i/y,i%y);
                auto invPixel = cam->getNextFrame().at<uchar>(i/y,i%y);

                // Not considering shadow mask. But the following test should be
                // more strict than shadow mask.
                if (invPixel > pixel && invPixel-pixel > cam->getThreashold(i))
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
}

void ReconstructorCPU::renconstruct()
{
    /* Assuming there are two cameras 
     * For each projector pixel,
     * find the min dis
     * */
    for ( size_t i=0; i<buckets_[0].size(); i++)
    {
        float minDist = 9999.0;
        size_t minIdx0 = 0;
        size_t minIdx1 = 0;
        for ( size_t j=0; j<buckets_[0][i].size(); j++) //left camera
        {
            for (size_t k=0; k<buckets_[1][i].size();k++) //right camera
            {
                /* float dist = rayMinDist(cameras_[0].getRay(buckets_[0][i][j]), 
                 *  cameras_[1].getRay(buckets_[0][i][k]))
                 * if (dist < minDist)
                 *      blah blah
                 */
                ;
            }
        }
    }
    
}

} // namespace SLS

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
        b.resize(1<<projector_->getRequiredNumFrames());
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
        LOG::writeLog("Generating reconstruction bucket for \"%s\" ... \n", cam->getName().c_str());

        cam->undistort();
        cam->computeShadowsAndThreasholds();

        size_t x=0,y=0,xTimesY=0;
        cam->getResolution(x,y);
        xTimesY=x*y;
        //For each camera pixle
        assert(dynamic_cast<FileReader*>(cam)->getNumFrames() == projector_->getRequiredNumFrames()*2+2);


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
                //std::cout<<dynamic_cast<FileReader*>(cam)->getCurrentIdx()<<std::endl;
                //auto img = cam->getNextFrame();
                //auto invImg = cam->getNextFrame();

                //auto pixel = img.at<uchar>(i%y, i/y);
                //auto invPixel = invImg.at<uchar>(i%y, i/y);
                //
                //cv::namedWindow("right", cv::WINDOW_NORMAL);
                //cv::imshow("right", img);
                //cv::namedWindow("left", cv::WINDOW_NORMAL);
                //cv::imshow("left", invImg);
                //cv::waitKey(0);
                //std::cout<<dynamic_cast<FileReader*>(cam)->getCurrentIdx()<<std::endl;
                //std::cout<<dynamic_cast<FileReader*>(cam)->getNumFrames()<<std::endl;

                auto pixel = cam->getNextFrame().at<uchar>(i%y,i/y);
                auto invPixel = cam->getNextFrame().at<uchar>(i%y,i/y);


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
    size_t x,y;
    cameras_[0]->getResolution(x,y);
    LOG::startTimer("Reconstructing with CPU ... ");
    initBuckets();

    std::ofstream of("test.obj");

    for ( size_t i=0; i<buckets_[0].size(); i++)
    {
        if (!buckets_[1][i].empty() && !buckets_[0][i].empty())
        {
            float dist=0.0;
            size_t cam0=buckets_[0][i][0];
            size_t cam1=buckets_[1][i][0];

            //LOG::writeLog("Cam0 %d,%d Cam1 %d,%d\n",cam0/y, cam0%y, cam1/y, cam1%y);
            //LOG::writeLog("Cam0 Ray: origin: %s dir %s, Cam1 Ray: origin %s dir %s\n", 
            //        glm::to_string(cameras_[0]->getRay(cam0).origin).c_str(),
            //        glm::to_string(cameras_[0]->getRay(cam0).dir).c_str()
            //        , glm::to_string(cameras_[1]->getRay(cam1).origin).c_str()
            //        , glm::to_string(cameras_[1]->getRay(cam1).dir).c_str());

            auto midP=midPoint(cameras_[0]->getRay(cam0), cameras_[1]->getRay(cam1), dist);
            of<<"v "<<midP.x<<" "<<midP.y<<" "<<midP.z<<std::endl;
        }
    }
    of.close();
    LOG::endTimer('s');
    
}

} // namespace SLS

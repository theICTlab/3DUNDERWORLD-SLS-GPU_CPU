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
    }

    // Intensive debugging area===================
    int count=0;
    for (int i = 0; i < 1024*768; i++) {
        if ((!buckets_[0][i].empty()) && (!buckets_[1][i].empty()))
            count++;
    }
    std::cout<<"Matched pixels: "<<count<<std::endl;

    // Write to binary ppm
    {
        FILE *pFp = std::fopen("testBucket0.ppm", "wb");
        size_t w = 1024;
        size_t h = 768;
        if (pFp) {
            std::fprintf(pFp, "P5\n%zu\n%zu\n%d\n", (size_t )w, (size_t )h, 255);
            for (unsigned i = 0; i < w * h; i++) {   // i row based
                auto bucketIdx = (i % w) * h + i / w;   // bucketIdx is col based
                if (!buckets_[0][bucketIdx].empty())
                    putc((unsigned char) buckets_[0][bucketIdx].size()*255/60, pFp);
                else
                    putc((unsigned char) 0, pFp);
            }
            std::fclose(pFp);
        }
    }
    // Write to binary ppm
    {
        FILE *pFp = std::fopen("testBucket1.ppm", "wb");
        size_t w = 1024;
        size_t h = 768;
        if (pFp) {
            std::fprintf(pFp, "P5\n%zu\n%zu\n%d\n", (size_t) w, (size_t) h, 255);
            for (unsigned i = 0; i < w * h; i++) {
                auto bucketIdx = (i % w) * h + i / w;
                if (!buckets_[1][bucketIdx].empty())
                    putc((unsigned char) 255, pFp);
                else
                    putc((unsigned char) 0, pFp);
            }
            std::fclose(pFp);

        }
    }
    //// Write to binary
    //{
    //    for (size_t i=0; i< buckets_[0].size(); i++)
    //    {
    //        if (!buckets_[0][i].empty() && !(buckets_[1][i].empty())){
    //            cv::Mat projImg(cv::Mat(768, 1024, CV_8UC1, cv::Scalar(0)));
    //            cv::Mat camImg(3264, 4896, CV_8UC1, cv::Scalar(0));
    //            projImg.at<uchar>(i%768, i/768) = 255;
    //            //for (unsigned i=0; i< 768; i++)
    //            //    for (unsigned j=0; j< 768; j++)
    //            //        projImg.at<uchar>(i,j) = 255;
    //            const auto &patch = buckets_[0][i];
    //            const auto &patch1 = buckets_[1][i];
    //            std::cout<<i%768<<"\t"<<i/768<<std::endl;
    //            std::cout<<patch.size()<<std::endl;
    //            const auto smallerPatch=patch.size()<patch1.size()?patch:patch1;
    //            for (unsigned j=0; j<smallerPatch.size(); j++){
    //                std::cout<<"============\n";
    //                std::cout<<patch[i]%3264<<"\t"<<patch[i]/3264<<std::endl;
    //                std::cout<<patch1[i]%3264<<"\t"<<patch1[i]/3264<<std::endl;
    //            }
    //            //for (const auto &element: patch){
    //            //    camImg.at<uchar>(element%3264, element/3264) = 255;
    //            //    std::cout<<element%3264<<"\t"<<element/3264<<std::endl;
    //            //}
    //            cv::namedWindow("CamImage", CV_WINDOW_NORMAL);
    //            cv::namedWindow("projImage", CV_WINDOW_NORMAL);
    //            cv::imshow("projImage", projImg);
    //            cv::imshow("CamImage", camImg);
    //            cv::waitKey(0);
    //            projImg.release();
    //            camImg.release();
    //            break;
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

            //if (minDist < 0.3) //Setting threshold
            {
                pointCloud_.push_back(midPointAvg.x);
                pointCloud_.push_back(midPointAvg.y);
                pointCloud_.push_back(midPointAvg.z);
                pointCloud_.push_back(1);
                unsigned char r0, g0, b0;
                unsigned char r1, g1, b1;
                cameras_[0]->getColor(minCam0Idx, r0, g0, b0);
                cameras_[1]->getColor(minCam1Idx, r1, g1, b1);
                //   pointCloud_.push_back((float)(r0+r1)/255.0f);
                //   pointCloud_.push_back((float)(g0+g1)/255.0f);
                //   pointCloud_.push_back((float)(b0+b1)/255.0f);
            }
        }
    }
    LOG::endTimer("Finished reconstruction in ");
}


} // namespace SLS

#include "ReconstructorCUDA.cuh"
#include "fileReaderCUDA.cuh"
namespace SLS
{

ReconstructorCUDA :: ReconstructorCUDA(const size_t projX, const size_t projY): 
    Reconstructor()
{
    projector_ = new Projector(projX, projY);
}
ReconstructorCUDA::~ReconstructorCUDA(){
    for (auto &cam: cameras_)
        delete cam;
    delete projector_;
}
void ReconstructorCUDA::addCamera(Camera *cam)
{
    cameras_.push_back(cam);
}
void ReconstructorCUDA::renconstruct()
{
    // For each camera
    for(size_t camIdx = 0; camIdx < cameras_.size(); camIdx++)
    {
        auto &cam = cameras_[camIdx];
        LOG::writeLog("Generating reconstruction bucket for \"%s\" ... \n", cam->getName().c_str());
        cam->computeShadowsAndThreasholds();    // can it be done in GPU?
        size_t x=0,y=0,xTimesY=0;
        cam->getResolution(x,y);
        xTimesY=x*y;
        cam->nextFrame();cam->nextFrame();//skip first two frames
        // Load all images into GPU memory
        uchar *images_d=nullptr;
        gpuErrchk(cudaMalloc((void**)&images_d, sizeof(uchar)*xTimesY*projector_->getRequiredNumFrames()*2));
        Dynamic_Bitset_Array bitsetArray(xTimesY, projector_->getRequiredNumFrames());

        //Preparing data
        for (size_t i=0; i<projector_->getRequiredNumFrames(); i++)
        {
            auto frm = cam->getNextFrame();
            auto invFrm = cam->getNextFrame();
            assert(frm.isContinuous() && invFrm.isContinuous());
            gpuErrchk( cudaMemcpy( &images_d[xTimesY*2*i], frm.data, 
                    sizeof(uchar)*xTimesY, cudaMemcpyHostToDevice));
            gpuErrchk( cudaMemcpy( &images_d[xTimesY*(2*i+1)], invFrm.data, 
                    sizeof(uchar)*xTimesY, cudaMemcpyHostToDevice));
        }
        FileReaderCUDA *cudaCam = dynamic_cast<FileReaderCUDA*> (cam);
        assert(cam != nullptr);
        buildBucket_kernel<<<200, 200>>> (
                images_d, 
                projector_->getRequiredNumFrames(),
                xTimesY,
                cam->getWhiteThreshold(),
                cudaCam->getMask()->getGPUOBJ(),
                bitsetArray.getGPUOBJ()
                );
        //Check for errors
        gpuErrchk(cudaPeekAtLastError());

        uint *patternDec_d = nullptr;
        gpuErrchk( cudaMalloc((void**)&patternDec_d, sizeof(uint)*xTimesY));

        bucket2uint_kernel<<<200,200>>> (
                bitsetArray.getGPUOBJ(),
                xTimesY,
                patternDec_d);

        gpuErrchk(cudaPeekAtLastError());

        // debugging the pattern by write the to file


        uint *patternDec_h = new uint[xTimesY];
        printf("Device: %p, Host: %p and sizeof uint is %d\n", patternDec_d, patternDec_h, sizeof(uint));
        gpuErrchk( cudaMemcpy(patternDec_h, patternDec_d, sizeof(uint)*xTimesY, cudaMemcpyDeviceToHost));
        assert( uint2PGM( "test"+cam->getName()+".pgm", x, y, patternDec_h,(uint)1048576 ));

        delete[] patternDec_h;
        gpuErrchk(cudaFree(patternDec_d));
        gpuErrchk(cudaFree(images_d));
    }
}

// Kernels 

__global__ void buildBucket_kernel(
        const uchar * imgs,
        size_t numimgs,
        size_t XtimesY,
        uchar whiteThreshold,
        Dynamic_Bitset_Array_GPU mask,
        Dynamic_Bitset_Array_GPU patterns
        )
{
    uint idx = blockIdx.x*blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;
    const size_t BITS_PER_BYTE = mask.BITS_PER_BYTE;
    while (idx < XtimesY)
    {
        // each thread take care of 8 bits (a byte) to avoid race condition
        for (size_t i=0; i< BITS_PER_BYTE; i++)
        {
            size_t pixelIdx = idx*BITS_PER_BYTE+i;

            if (!mask.getBit(pixelIdx, 0)) continue;

            for (size_t bitIdx = 0; bitIdx < numimgs; bitIdx++)
            {
                uchar pixel = imgs[pixelIdx + XtimesY*(2*bitIdx)];
                uchar invPixel = imgs[pixelIdx + XtimesY * (2*bitIdx+1)];
                // TODO: Set elemnt at bitIdx and modify mask
                if (invPixel > pixel && invPixel-pixel >= whiteThreshold)
                    patterns.clearBit(bitIdx, pixelIdx);
                else if (pixel > invPixel && pixel-invPixel > whiteThreshold)
                    patterns.setBit(bitIdx, pixelIdx);
                else
                    mask.clearBit(pixelIdx, 0);
            }
        }
        idx += stride;
    }
}

__global__ void bucket2uint_kernel(
        Dynamic_Bitset_Array_GPU patterns,
        size_t XtimesY,
        uint * output)
{
    uint idx = blockIdx.x*blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;
    while (idx < XtimesY)
    {
        output[idx] = patterns.to_uint(idx);
        idx += stride;
    }
}

} // namespace SLS

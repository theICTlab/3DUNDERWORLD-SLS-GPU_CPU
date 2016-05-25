#include "../Dynamic_bits.cuh"
#include "./kernel.cu"
#include "../fileReaderCUDA.cuh"
#include "../ReconstructorCUDA.cuh"

int main()
{
    /* Bitset test */
    /*
    SLS::Dynamic_Bitset_Array bits(200*200, 40);

    // Prepare output GPU memory
    unsigned int *res=nullptr;
    // Prepare output CPU memory
    unsigned int resH[200*200]={0};
    // Allocate output GPU memory
    gpuErrchk( cudaMalloc ((void**)&res, sizeof(unsigned int)*200*200));

    // Run kernels
    setBits<<<200, 200>>>(bits.getGPUOBJ());    //Get an GPU object and pass it to kernel
    clearBits<<<200, 200>>>(bits.getGPUOBJ());
    toUintArray<<<200,200>>>( bits.getGPUOBJ(), res);

    // Copy from GPU memory to CPU memory
    gpuErrchk (cudaMemcpy(resH, res, sizeof(unsigned int)*200*200, cudaMemcpyDeviceToHost));
    // Free allocated GPU memory
    gpuErrchk( cudaFree(res));
    // Print it out
    for (size_t i=0; i<200*200; i++)
        std::cout<<resH[i];
    */
    LOG::restartLog();
    SLS::FileReaderCUDA *rightCamera  = new SLS::FileReaderCUDA("rightCamera");
    SLS::FileReaderCUDA *leftCamera  = new SLS::FileReaderCUDA("leftCamera");
    rightCamera->loadImages("/home/tsing/project/SLS/data/alexander/rightCam/dataset1");
    leftCamera->loadImages("/home/tsing/project/SLS/data/alexander/leftCam/dataset1");

    rightCamera->loadConfig("/home/tsing/project/SLS/data/alexander/rightCam/calib/output/calib.xml");
    leftCamera->loadConfig("/home/tsing/project/SLS/data/alexander/leftCam/calib/output/calib.xml");

    SLS::ReconstructorCUDA reconstructor(1024, 768);
    reconstructor.addCamera(rightCamera);
    reconstructor.addCamera(leftCamera);
    reconstructor.reconstruct();
    exportOBJVec4("test.obj", reconstructor);

    return 0;
}

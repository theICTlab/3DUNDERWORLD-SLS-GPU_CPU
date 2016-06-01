#include <ReconstructorCUDA/Dynamic_bits.cuh>
#include <ReconstructorCUDA/fileReaderCUDA.cuh>
#include <ReconstructorCUDA/ReconstructorCUDA.cuh>

int main()
{
    LOG::restartLog();
    SLS::FileReaderCUDA *rightCam=new SLS::FileReaderCUDA("rightCamera");
    SLS::FileReaderCUDA *leftCam= new SLS::FileReaderCUDA("leftCamera");
    rightCam->loadImages("../../data/alexander/rightCam/dataset1/");
    leftCam->loadImages("../../data/alexander/leftCam/dataset1/");
    rightCam->loadConfig("../../data/alexander/rightCam/calib/output/calib.xml");
    leftCam->loadConfig("../../data/alexander/leftCam/calib/output/calib.xml");
    SLS::ReconstructorCUDA reconstruct(1024,768);
    reconstruct.addCamera(rightCam);
    reconstruct.addCamera(leftCam);
    reconstruct.reconstruct();
    SLS::exportOBJVec4("test_GPU.obj",  reconstruct);
    LOG::writeLog("DONE\n");
    return 0;
}

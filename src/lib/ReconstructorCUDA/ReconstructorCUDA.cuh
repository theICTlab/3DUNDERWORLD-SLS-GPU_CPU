#pragma once
#include <core/Reconstructor.h>
#include <core/log.hpp>
#include "Dynamic_bits.cuh"

namespace SLS
{
class ReconstructorCUDA: public Reconstructor
{
private:
public:

     ReconstructorCUDA(const size_t projX, const size_t projY);
    ~ReconstructorCUDA() override;
    void addCamera(Camera *cam) override;
    void renconstruct() override;
};

struct GPUBucketsObj
{
    uint *data_;
    //size_t *offsets_;
    uint *count_; // num of element in buckets
    uint MAX_CNT_PER_BKT_;
    uint NUM_BKTS_;
    __device__ void add2Bucket(uint val, uint bktIdx)
    {
        data_[atomicAdd(&count_[bktIdx], 1)+bktIdx*MAX_CNT_PER_BKT_] = val;
        //if (bktIdx * MAX_CNT_PER_BKT_ > (1<<20)-1)
        //    printf("Too big!\n");
    }
};
class GPUBuckets
{
private:
    uint *data_;
    //size_t *offsets_;
    uint *count_; // num of element in buckets
    const uint MAX_CNT_PER_BKT_;
    const uint NUM_BKTS_;
public:
    GPUBuckets(size_t numBkt, size_t numPerBkt):
        MAX_CNT_PER_BKT_(numPerBkt), NUM_BKTS_(numBkt)
    {
        gpuErrchk (cudaMalloc( (void**)&data_, sizeof(uint)*MAX_CNT_PER_BKT_*NUM_BKTS_));
        gpuErrchk (cudaMemset( data_, 0, sizeof(uint)*MAX_CNT_PER_BKT_*NUM_BKTS_));
        gpuErrchk (cudaMalloc( (void**)&count_, sizeof(uint)*NUM_BKTS_));
        gpuErrchk (cudaMemset( count_, 0, sizeof(uint)*NUM_BKTS_));

    }
    GPUBucketsObj getGPUOBJ() 
    {
        GPUBucketsObj obj{data_, count_, MAX_CNT_PER_BKT_, NUM_BKTS_};
        return obj;
    }
    ~GPUBuckets()
    {
        gpuErrchk (cudaFree(data_));
        gpuErrchk (cudaFree(count_));
    }
    uint getNumBKTs() const{ return NUM_BKTS_;}

};


namespace Kernel{

__global__ void getPointCloud2Cam(
        GPUBucketsObj buckets0,
        Dynamic_Bitset_Array_GPU mask0,
        float *camMat0,
        float *distMat0,
        float *camTransMat0,

        GPUBucketsObj buckets1,
        Dynamic_Bitset_Array_GPU mask1,
        float *camMat1,
        float *distMat1,
        float *camTransMat1,

        uint camResX,
        uint camResY,

        float* pointCloud
        );

        

__global__ void testBitset(
        const uchar * imgs,
        size_t numimgs,
        size_t XtimesY,
        uchar whiteThreshold,
        Dynamic_Bitset_Array_GPU mask,
        Dynamic_Bitset_Array_GPU patterns
        );
__global__ void genPatternArray(
        const uchar * imgs,
        size_t numimgs,
        size_t XtimesY,
        uchar whiteThreshold,
        Dynamic_Bitset_Array_GPU mask,
        Dynamic_Bitset_Array_GPU patterns
        );
__global__ void buildBuckets(
        Dynamic_Bitset_Array_GPU mask,
        Dynamic_Bitset_Array_GPU patterns,
        size_t XtimesY,

        GPUBucketsObj bkts
        );
} // namespace Kernel

// Device functions
/**
 * @brief Undistort a pixel
 *
 * @param idx 1D index of pixel
 * @param resX 
 * @param resY 
 * @param camMat Camera Matrix
 * @param distMat Distortion coeff
 * @param undistorted output 2d undistored ray
 *
 */
inline  __device__ void undistortPixel(
        uint idx,
        uint resX,
        uint resY,
        float *camMat,
        float *distMat,
        
        float *undistorted
        )
{
    // Undistort the pxiel 
    // Row based
    uint distortedX = idx%resX;
    uint distortedY = idx/resX;

    float k[5] = {0.0};
    float fx, fy, ifx, ify, cx, cy;
    int iters = 1;

    k[0] = distMat [0];
    k[1] = distMat [1];
    k[2] = distMat [2];
    k[3] = distMat [3];
    k[4] = 0;

    iters = 5;

    fx = camMat[0];
    fy = camMat[4];
    ifx = 1.0/fx;
    ify = 1.0/fy;
    cx =camMat[2];
    cy = camMat[5];

    float x,y,x0,y0;

    x = distortedX;
    y = distortedY;
    x0 = (x-cx)*ifx;
    x = x0;
    y0 = (y-cy)*ify;
    y = y0;

    for(int jj = 0; jj < iters; jj++ )
    {
        float r2 = x*x + y*y;
        float icdist = 1./(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
        float deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x);
        float deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y;
        x = (x0 - deltaX)*icdist;
        y = (y0 - deltaY)*icdist;
    }
    undistorted[0] = (float)(x*fx)+cx;
    undistorted[1] = (float)(y*fy)+cy;

}

inline __device__ void getRay(
        float* undistorted,
        float* camMat,
        float* camTransMat,

        float* origin,  // vec4
        float* dir      // vec4
        )
{
    glm::mat4 lCamTransMat(1.0);
    memcpy(&lCamTransMat[0][0], camTransMat, sizeof(float)*16);
    auto lOrigin = lCamTransMat*glm::vec4(0.0, 0.0, 0.0, 1.0);
    auto lDir = glm::vec4 ( 
            (undistorted[0]-camMat[2])/camMat[0],
            (undistorted[1]-camMat[5])/camMat[4],
            1.0, 0.0
            );
    lDir = lCamTransMat*lDir;
    lDir = glm::normalize(lDir);

    // cpy back
    memcpy( origin, &lOrigin[0], sizeof(float)*4);
    memcpy( dir, &lDir[0], sizeof(float)*4);
}


/**
 * @brief Get mid point of two rays on the min
 * distance
 *
 * @param origin0 Origin point of first ray
 * @param dir0 Direction of first ray
 * @param origin1 Origin point of second ray
 * @param dir1 Direction of second ray
 * @param midPoint Return value of mid point
 *
 * @return Distance between two rays
 */
inline __device__ float getMidPoint(
        float* origin0,
        float* dir0, 
        float* origin1,
        float* dir1,

        float* midPoint // vec4
        )
{
    glm::vec3 v1 (0.0);
    glm::vec3 v2 (0.0);
    glm::vec3 p1 (0.0);
    glm::vec3 p2 (0.0);

    memcpy ( &v1[0], dir0, sizeof(float)*3);
    memcpy ( &v2[0], dir1, sizeof(float)*3);
    memcpy ( &p1[0], origin0, sizeof(float)*3);
    memcpy ( &p2[0], origin1, sizeof(float)*3);


    glm::vec3 v12 = p1-p2;
    float v1_dot_v1 = dot(v1, v1);
    float v2_dot_v2 = dot(v2, v2);
    float v1_dot_v2 = dot(v1, v2); 
    float v12_dot_v1 = dot(v12, v1);
    float v12_dot_v2 = dot(v12, v2);

    float denom = v1_dot_v1 * v2_dot_v2 - v1_dot_v2 * v1_dot_v2;
    float dist = -1.0;
    if (glm::abs(denom) < 0.1)
    {
        dist = -1.0;
        return dist;
    }

    float s =  (v1_dot_v2/denom) * v12_dot_v2 - (v2_dot_v2/denom) * v12_dot_v1;
    float t = -(v1_dot_v2/denom) * v12_dot_v1 + (v1_dot_v1/denom) * v12_dot_v2;
    dist = glm::length(p1+s*v1-p2-t*v2);
    auto midP = vec4((p1+s*v1+p2+t*v2)/2.0f, 1.0);
    //cpy back
    memcpy( midPoint, &midP[0], sizeof(float)*4);
    return dist;
}

} // namespace SLS

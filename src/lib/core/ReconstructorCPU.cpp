#include "ReconstructorCPU.h"
namespace SLS{
void ReconstructorCPU::renconstruct()
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
}
} // namespace SLS

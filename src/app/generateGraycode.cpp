#include <iostream>
#include <GrayCode/GrayCode.hpp>
int main()
{
    // Generate gray code based on the resolution of the projector.
    SLS::GrayCode gc(1024, 768);
    // Show the image sequence.
    // This function can be called when projecting patterns to the 
    // reconstruction object.
    gc.generateGrayCode();
    return 0;
}

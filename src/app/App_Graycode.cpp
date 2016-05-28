#include <iostream>
#include <GrayCode/GrayCode.hpp>
int main()
{
    SLS::GrayCode gc(1024, 768);
    gc.generateGrayCode();
    return 0;
}

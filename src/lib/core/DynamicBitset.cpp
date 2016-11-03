#include "DynamicBitset.h"
namespace SLS{
bool Dynamic_Bitset::writeToPGM( std::string fileName, const size_t &w, const size_t &h, bool transpose)
{
    FILE *pFp = std::fopen(fileName.c_str(), "wb");
    if (pFp)
    {
        //header of the pgm file
        std::fprintf(pFp, "P5\n%zu\n%zu\n%d\n", w, h, 255);
        for (size_t i = 0; i < h*w; i++)
        {
            if (transpose?getBit((i%w)*h+i/w):getBit(i))
                putc((unsigned char)255,pFp);
            else
                putc((unsigned char)0,pFp);
        }
        std::fclose(pFp);
        return true;
    }
    else return false;

}
};



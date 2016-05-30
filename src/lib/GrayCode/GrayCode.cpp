#include "GrayCode.hpp"
namespace SLS{
GrayCode::GrayCode(size_t projW, size_t projH):
    Projector(projW, projH), currImg(0)
{
    setColRowBitsNum();
    grayCodes_.resize(numColBits*2+numRowBits*2+2);
}
void GrayCode::generateGrayCode()
{
    // Init rest of the mat
    for (auto &mat: grayCodes_)
        mat = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(0));
    // Set first frame to white
    grayCodes_[0] = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(255));


    uchar flag = 0;
	for (size_t j=0; j<width_; j++)	
	{
		int rem=0, num=j, prevRem=j%2;
		for (size_t k=0; k<numColBits; k++)	
		{
			num=num/2;
			rem=num%2;
			if ((rem==0 && prevRem==1) || (rem==1 && prevRem==0)) { 
				flag=1;
			}
			else {
				flag= 0;
			}
			for (size_t i=0;i<height_;i++)	
			{
                uchar pixel_color = flag*255;
                grayCodes_[2*numColBits-2*k].at<uchar>(i, j) = pixel_color;
                pixel_color = pixel_color==0?255:0;
                grayCodes_[2*numColBits-2*k+1].at<uchar>(i,j) = pixel_color;

			}
			prevRem=rem;
		} 
		
	}
    
	for (size_t i=0;i<height_;i++)	
	{
		int rem=0, num=i, prevRem=i%2;
		for (size_t k=0; k<numRowBits; k++)	
		{

			num=num/2;
			rem=num%2;

			if ((rem==0 && prevRem==1) || (rem==1 && prevRem==0)) { 
				flag=1;
			}
			else {
				flag= 0;
			}

			for (size_t j=0; j<width_; j++)	
			{
                uchar pixel_color = flag*255;
                grayCodes_[2*numRowBits-2*k+2*numColBits].at<uchar>(i,j) = pixel_color;
                pixel_color = pixel_color==0?255:0;
                grayCodes_[2*numRowBits-2*k+2*numColBits+1].at<uchar>(i,j) = pixel_color;
			}

			prevRem=rem;
		} 
		
	}
    for (size_t i=0; i<grayCodes_.size(); i++)
    {
        cv::imshow("test",grayCodes_[i]);
        cv::waitKey(0);
    }
}
} // namespace SLS

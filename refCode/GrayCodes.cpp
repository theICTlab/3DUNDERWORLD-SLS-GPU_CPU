//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "GrayCodes.h"


GrayCodes::GrayCodes(int projW, int projH)	
{
	for (int i=0; i<GRAY_MAX_NUM; i++)
	{
		grayCodes[i]=NULL;

	}

	height=projH;
	width=projW;

	imgsLoaded=false;

	calNumOfImgs();

	currentImgNum=0;

}

GrayCodes::~GrayCodes()	
{
	if(imgsLoaded)
		unload();
}

int GrayCodes::getNumOfColBits()
{
	return numOfColImgs;
}

int GrayCodes::getNumOfRowBits()
{
	return numOfRowImgs;
}

void GrayCodes::calNumOfImgs()
{
	numOfColImgs = (int)ceil(log(double (width))/log(2.0));
	numOfRowImgs = (int)ceil(log(double (height))/log(2.0));

	numOfImgs= 2 + 2*numOfColImgs + 2*numOfRowImgs;
}

void GrayCodes::unload()
{

	for(int i=0; i < numOfImgs;i++ )
	{
		if(grayCodes[i])
		{
			cvReleaseImage(&grayCodes[i]);
			grayCodes[i]=NULL;
		}
	}
	imgsLoaded=false;
}

IplImage* GrayCodes::getImg(int num)
{
	if(num<numOfImgs)
	{
		currentImgNum = num;
		return grayCodes[num];
	}
	else
		return NULL;
}

IplImage* GrayCodes::getNextImg()
{
	
	if(currentImgNum<numOfImgs)
	{
		currentImgNum++;
		return grayCodes[currentImgNum-1];
	}
	else
		return NULL;
}

int GrayCodes::getNumOfImgs()
{
	return numOfImgs;
}


void GrayCodes::allocMemForImgs()
{

	for(int i=0; i<numOfImgs; i++)
	{
		grayCodes[i]=cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
	}
	imgsLoaded=true;
}

void GrayCodes::generateGrays()
{
	if(!imgsLoaded)
	{
		allocMemForImgs();
	}

	//generate all white and all black images
	for (int j=0; j<width; j++)	
	{	
		for (int i=0;i<height;i++)	
		{
			CvScalar pixel_color;
			pixel_color.val[0] = 255;
			cvSet2D(grayCodes[0], i, j, pixel_color);

			pixel_color.val[0] = 0;
			cvSet2D(grayCodes[1], i, j, pixel_color);
		} 
	}


	int flag=0;

	for (int j=0; j<width; j++)	
	{
		int rem=0, num=j, prevRem=j%2;
		
		for (int k=0; k<numOfColImgs; k++)	
		{
			
			num=num/2;
			rem=num%2;

			if ((rem==0 && prevRem==1) || (rem==1 && prevRem==0))
			{ 
				flag=1;
			}
			else 
			{
				flag= 0;
			}

			for (int i=0;i<height;i++)	
			{
				CvScalar pixel_color;
				pixel_color.val[0] = float(flag*255);
				cvSet2D(grayCodes[2*numOfColImgs-2*k], i, j, pixel_color);

				if(pixel_color.val[0]>0)
					pixel_color.val[0]=0;
				else
					pixel_color.val[0]=255;

				cvSet2D(grayCodes[2*numOfColImgs-2*k+1], i, j, pixel_color);
			}

			prevRem=rem;
		} 
		
	}

	for (int i=0;i<height;i++)	
	{
		int rem=0, num=i, prevRem=i%2;
		
		for (int k=0; k<numOfRowImgs; k++)	
		{

			num=num/2;
			rem=num%2;

			if ((rem==0 && prevRem==1) || (rem==1 && prevRem==0))
			{ 
				flag=1;
			}
			else 
			{
				flag= 0;
			}

			for (int j=0; j<width; j++)	
			{
				CvScalar pixel_color;
				pixel_color.val[0] = float(flag*255);
				cvSet2D(grayCodes[2*numOfRowImgs-2*k+2*numOfColImgs], i, j, pixel_color);

				if(pixel_color.val[0]>0)
					pixel_color.val[0]=0;
				else
					pixel_color.val[0]=255;

				cvSet2D(grayCodes[2*numOfRowImgs-2*k+2*numOfColImgs+1], i, j, pixel_color);
			}

			prevRem=rem;
		} 
		
	}
}

void GrayCodes::save()
{
	for(int i = 0; i < numOfImgs; i++)
	{
		std::stringstream path;

		if(i+1<10)
			path<<"0";
		
		path<< i+1 <<".png";

		cvSaveImage(path.str().c_str(),grayCodes[i]);
	}
}

//convert a gray code sequence to a decimal number
int GrayCodes::grayToDec(cv::vector<bool> gray)
{
	int dec=0;
	bool tmp = gray[0];
	
	if(tmp)
		dec+=(int) pow((float)2,int (gray.size() -1));

	for(int i=1; i< gray.size(); i++)
	{
		tmp=Utilities::XOR(tmp,gray[i]);

		if(tmp)
			dec+= (int) pow((float)2,int (gray.size()-i-1) );
	}
	return dec;
}



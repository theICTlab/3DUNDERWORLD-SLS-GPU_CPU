//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#include "StdAfx.h"
#include "PointCloudImage.h"



PointCloudImage::PointCloudImage(int imageW,int imageH, bool colorFlag_)
{
	w=imageW;
	h=imageH;
	colorFlag = colorFlag_;
	
	points = cv::Mat(h,w,CV_32FC3);

	if(colorFlag==true)
	{
		
		color = cv::Mat(h,w,CV_32FC3,cv::Scalar(0));
	}
	else
		color = NULL;

	numOfPointsForPixel =  cv::Mat(h,w,CV_8U,cv::Scalar(0));
}

PointCloudImage::~PointCloudImage(void)
{
	
}

bool PointCloudImage::hasColor()
{
	return colorFlag;
}

bool PointCloudImage::setPoint(int i_w, int j_h, cv::Point3f point, cv::Vec3f colorBGR)
{
	if(i_w>w || j_h>h)
		return false;

	setPoint(i_w,j_h,point);

	Utilities::matSet3D(color,i_w,j_h,colorBGR);

	return true;
}

bool PointCloudImage::setPoint(int i_w, int j_h, cv::Point3f point)
{
	if(i_w>w || j_h>h)
		return false;

	Utilities::matSet3D(points,i_w,j_h,(cv::Vec3f)point);
	Utilities::matSet2D(numOfPointsForPixel,i_w,j_h,1);

	return true;
}

bool PointCloudImage::getPoint(int i_w, int j_h, cv::Point3f &pointOut, cv::Vec3f &colorOut)
{
	if(i_w>w || j_h>h)
		return false;

	uchar num = numOfPointsForPixel.at<uchar>(j_h,i_w);

	if(num > 0)
	{
		
		pointOut = (cv::Point3f) (Utilities::matGet3D(points,i_w,j_h) / (float) num);

		
		if(!color.empty())
		{
			colorOut = (cv::Point3f) (Utilities::matGet3D(color,i_w,j_h) / (float) num);
		}
		else
		{
			return false;
		}
		

		return true;
	}
	else
	{
		return false;
	}
	
}

bool PointCloudImage::getPoint(int i_w, int j_h, cv::Point3f &pointOut)
{
	if(i_w>w || j_h>h)
		return false;

	uchar num = numOfPointsForPixel.at<uchar>(j_h,i_w);

	if(num > 0)
	{
		pointOut = (cv::Point3f) (Utilities::matGet3D(points,i_w,j_h) / (float) num);
		return true;
	}
	else
	{
		return false;
	}
	
}

bool PointCloudImage::addPoint(int i_w, int j_h, cv::Point3f point, cv::Vec3f colorBGR)
{
	if(i_w>w || j_h>h)
		return false;

	uchar num = numOfPointsForPixel.at<uchar>(j_h,i_w);

	if(num == 0)
		return setPoint(i_w,j_h,point,colorBGR);

	addPoint(i_w,j_h,point);

	if(!color.empty())
	{
		cv::Vec3f c = Utilities::matGet3D(color,i_w,j_h);

		Utilities::matSet3D(color,i_w,j_h,colorBGR + c);
	}
	else
	{
		return false;
	}

	return true;
}

bool PointCloudImage::addPoint(int i_w, int j_h, cv::Point3f point)
{
	if(i_w>w || j_h>h)
		return false;

	uchar num = numOfPointsForPixel.at<uchar>(j_h,i_w);

	if(num == 0)
		return setPoint(i_w,j_h,point);

	cv::Point3f p = Utilities::matGet3D(points,i_w,j_h);
	Utilities::matSet3D(points,i_w,j_h,(cv::Vec3f)(point + p));

	numOfPointsForPixel.at<uchar>(j_h,i_w) = num + 1;

	return true;
}



void PointCloudImage::exportXYZ(char path[], bool exportOffPixels, bool colorFlag)
{

	std::ofstream out; 
	out.open(path);

	int load;

	cv::Point3f p;
	cv::Vec3f c;

	std::cout<<"Export "<< path << "...";

	for(int i = 0; i<w; i++)
	{
		for(int j = 0; j<h; j++)
		{
			uchar num = numOfPointsForPixel.at<uchar>(j,i);

			if(!exportOffPixels && num == 0)
				continue;			
			
			getPoint(i,j,p,c);

			if(exportOffPixels && num == 0)
			{
				p = cv::Point3f(0,0,0);
				c = cv::Point3f(0,0,0);
			}

			out<<p.x<<" "<<p.y<<" "<<p.z;

			if(colorFlag && !color.empty())
			{
				out<<" "<<c[2]<<" "<<c[1]<<" "<<c[0]<<"\n";
			}
			else
			{
				out<<"\n";
			}
		}
	}

	out.close();
	std::cout<<"done\n";
}

void PointCloudImage::exportNumOfPointsPerPixelImg(char path[])
{
	
	cv::Mat projToCamRays(cvSize(w, h), CV_8U);

	float max=0;

	int maxX,maxY;

	for(int i=0; i<w; i++)
	{
		for(int j=0; j<h; j++)
		{
			uchar num = numOfPointsForPixel.at<uchar>(j,i);

			if(num > max)
			{
				max = num;
				maxX=i;
				maxY=j;
			}
		}
	} 

	for(int i=0; i<w; i++)
	{
		for(int j=0; j<h; j++)
		{

			uchar num = numOfPointsForPixel.at<uchar>(j,i);
			Utilities::matSet2D(projToCamRays,i,j, num/(float)(max*255.0));

		}
	}

	cv::imwrite("reconstruction/projToCamRays.png",projToCamRays);

	std::ofstream out1;
	std::stringstream txt;
	txt<<path<<".txt";
	out1.open(txt.str().c_str() );

	out1<< "black color = 0\nwhite color = "<< max <<"\nmax Pixel: ("<<maxX<<","<<maxY<<")";

	out1.close();

}

int PointCloudImage::getWidth()
{
	return w;
}

int PointCloudImage::getHeight()
{
	return h;
}


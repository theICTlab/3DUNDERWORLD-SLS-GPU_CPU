//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#include "StdAfx.h"
#include "Utilities.h"


Utilities::Utilities(void)
{
}


Utilities::~Utilities(void)
{

}

bool Utilities::XOR(bool val1, bool val2)
{
	if(val1==val2)
		return 0;
	else
		return 1;
}

void Utilities::normalize(cv::Vec3f &vec)	
{
	double mag = sqrt( vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
	
	vec[0] /= (float) max(0.000001, mag);
	vec[1] /= (float) max(0.000001, mag);
	vec[2] /= (float) max(0.000001, mag);
	
	return;
}

void Utilities::normalize3dtable(double vec[3])
{
	double mag = sqrt( vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
	
	vec[0] /= max(0.000001, mag);
	vec[1] /= max(0.000001, mag);
	vec[2] /= max(0.000001, mag);
}

//convert image pixel to image 3d space point
void Utilities::pixelToImageSpace(double p[3], CvScalar fc, CvScalar cc)
{

	p[0]=(p[0]-cc.val[0])/fc.val[0];
	p[1]=(p[1]-cc.val[1])/fc.val[1];
	p[2]=1;

}

cv::Point3f Utilities::pixelToImageSpace(cv::Point2f p,  VirtualCamera cam)
{
	cv::Point3f point;

	point.x = (p.x-cam.cc.x) / cam.fc.x;
	point.y = (p.y-cam.cc.y) / cam.fc.y;
	point.z = 1;
	
	return point;
}


cv::Point2f Utilities::undistortPoints( cv::Point2f p,  VirtualCamera cam)
{

    double  k[5]={0,0,0,0,0}, fx, fy, ifx, ify, cx, cy;
  
    int iters = 1;
	
	k[0] = cam.distortion.at<float>(0);
	k[1] = cam.distortion.at<float>(1);
	k[2] = cam.distortion.at<float>(2);
	k[3] = cam.distortion.at<float>(3);
	k[4]=0;

    iters = 5;

	fx = cam.fc.x; 
    fy = cam.fc.y; 
	
    ifx = 1./fx;
    ify = 1./fy;
    cx = cam.cc.x; 
    cy = cam.cc.y; 


	double x, y, x0, y0;

	x=p.x;
	y=p.y;

	x0 = x = (x - cx)*ifx;
	y0 = y = (y - cy)*ify;
			
	for(int jj = 0; jj < iters; jj++ )
	{
		double r2 = x*x + y*y;
		double icdist = 1./(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
		double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x);
		double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y;
		x = (x0 - deltaX)*icdist;
		y = (y0 - deltaY)*icdist;
	}
	
	return cv::Point2f((float)(x*fx)+cx,(float)(y*fy)+cy);
}

//calculate the intersection point of a ray and a plane, given the normal and a point of the plane, and a point and the vector of the ray
CvScalar Utilities::planeRayInter(CvScalar planeNormal,CvScalar planePoint, CvScalar rayVector, CvScalar rayPoint )
{
	double l;
	CvScalar point;

	CvScalar pSub;

	pSub.val[0] = - rayPoint.val[0] + planePoint.val[0];
	pSub.val[1] = - rayPoint.val[1] + planePoint.val[1];
	pSub.val[2] = - rayPoint.val[2] + planePoint.val[2];

	double dotProd1 = pSub.val[0] * planeNormal.val[0] + pSub.val[1] * planeNormal.val[1] + pSub.val[2] * planeNormal.val[2];
	double dotProd2 = rayVector.val[0] * planeNormal.val[0] + rayVector.val[1] * planeNormal.val[1] + rayVector.val[2] * planeNormal.val[2];
	
	if(fabs(dotProd2)<0.00001)
	{
		std::cout<<"Error 10\n";
		//getch();
		point.val[0]=0;
		point.val[1]=0;
		point.val[2]=0;
		return point;
	}

	l = dotProd1 / dotProd2;

	point.val[0] = rayPoint.val[0] + l * rayVector.val[0]; 
	point.val[1] = rayPoint.val[1] + l * rayVector.val[1]; 
	point.val[2] = rayPoint.val[2] + l * rayVector.val[2]; 

	return point;
}

double Utilities::matGet2D(cv::Mat m, int x, int y)
{
	int type = m.type();

	switch(type)
	{
		case CV_8U:
			return m.at<uchar>(y,x);
			break;
		case CV_8S:
			return m.at<schar>(y,x);
			break;
		case CV_16U:
			return m.at<ushort>(y,x);
			break;
		case CV_16S:
			return m.at<short>(y,x);
			break;
		case CV_32S:
			return m.at<int>(y,x);
			break;
		case CV_32F:
			return m.at<float>(y,x);
			break;
		case CV_64F:
			return m.at<double>(y,x);
			break;
	}
	
}

double Utilities::matGet3D(cv::Mat m, int x, int y, int i)
{
	int type = m.type();

	switch(type)
	{
		case CV_8U:
		case CV_MAKETYPE(CV_8U,3):
			return m.at<uchar>(y,x,i);
			break;
		case CV_8S:
		case CV_MAKETYPE(CV_8S,3):
			return m.at<schar>(y,x,i);
			break;
		case CV_16U:
		case CV_MAKETYPE(CV_16U,3):
			return m.at<ushort>(y,x,i);
			break;
		case CV_16S:
		case CV_MAKETYPE(CV_16S,3):
			return m.at<short>(y,x,i);
			break;
		case CV_32S:
		case CV_MAKETYPE(CV_32S,3):
			return m.at<int>(y,x,i);
			break;
		case CV_32F:
		case CV_MAKETYPE(CV_32F,3):
			return m.at<float>(y,x,i);
			break;
		case CV_64F:
		case CV_MAKETYPE(CV_64F,3):
			return m.at<double>(y,x,i);
			break;
	}

}

cv::Vec3d Utilities::matGet3D(cv::Mat m, int x, int y)
{
	int type = m.type();

	switch(type)
	{
		case CV_8U:
		case CV_MAKETYPE(CV_8U,3):
			return m.at<cv::Vec3b>(y,x);
			break;
		case CV_8S:
		case CV_MAKETYPE(CV_8S,3):
			return m.at<cv::Vec3b>(y,x);
			break;
		case CV_16U:
		case CV_MAKETYPE(CV_16U,3):
			return m.at<cv::Vec3w>(y,x);
			break;
		case CV_16S:
		case CV_MAKETYPE(CV_16S,3):
			return m.at<cv::Vec3s>(y,x);
			break;
		case CV_32S:
		case CV_MAKETYPE(CV_32S,3):
			return m.at<cv::Vec3i>(y,x);
			break;
		case CV_32F:
		case CV_MAKETYPE(CV_32F,3):
			return m.at<cv::Vec3f>(y,x);
			break;
		case CV_64F:
		case CV_MAKETYPE(CV_64F,3):
			return m.at<cv::Vec3d>(y,x);
			break;
	}

}

void Utilities::matSet2D(cv::Mat m, int x, int y, double val)
{
	int type = m.type();

	switch(type)
	{
		case CV_8U:
			m.at<uchar>(y,x)  = (uchar) val;
			break;
		case CV_8S:
			m.at<schar>(y,x)  = (schar) val;
			break;
		case CV_16U:
			m.at<ushort>(y,x) = (ushort) val;
			break;
		case CV_16S:
			m.at<short>(y,x)  = (short) val;
			break;
		case CV_32S:
			m.at<int>(y,x)	  = (int) val;
			break;
		case CV_32F:
			m.at<float>(y,x)  = (float) val;
			break;
		case CV_64F:
			m.at<double>(y,x) = (double) val;
			break;
	}

}

void Utilities::matSet3D(cv::Mat m, int x, int y,int i, double val)
{
	int type = m.type();

	switch(type)
	{
		case CV_8U:
		case CV_MAKETYPE(CV_8U,3):
			m.at<uchar>(y,x,i) = (uchar) val;
			break;
		case CV_8S:
		case CV_MAKETYPE(CV_8S,3):
			m.at<schar>(y,x,i) = (schar) val;
			break;
		case CV_16U:
		case CV_MAKETYPE(CV_16U,3):
			m.at<ushort>(y,x,i) = (ushort) val;
			break;
		case CV_16S:
		case CV_MAKETYPE(CV_16S,3):
			m.at<short>(y,x,i) = (short) val;
			break;
		case CV_32S:
		case CV_MAKETYPE(CV_32S,3):
			m.at<int>(y,x,i) = (int) val;
			break;
		case CV_32F:
		case CV_MAKETYPE(CV_32F,3):
			m.at<float>(y,x,i) = (float) val;
			break;
		case CV_64F:
		case CV_MAKETYPE(CV_64F,3):
			m.at<double>(y,x) = (double) val;
			break;
	}

}

void Utilities::matSet3D(cv::Mat m, int x, int y, cv::Vec3d val)
{
	int type = m.type();

	switch(type)
	{
		case CV_8U:
		case CV_MAKETYPE(CV_8U,3):
			m.at<cv::Vec3b>(y,x) =  val;
			break;
		case CV_8S:
		case CV_MAKETYPE(CV_8S,3):
			m.at<cv::Vec3b>(y,x) =  val;
			break;
		case CV_16U:
		case CV_MAKETYPE(CV_16U,3):
			m.at<cv::Vec3w>(y,x) = val;
			break;
		case CV_16S:
		case CV_MAKETYPE(CV_16S,3):
			m.at<cv::Vec3s>(y,x) = val;
			break;
		case CV_32S:
		case CV_MAKETYPE(CV_32S,3):
			m.at<cv::Vec3i>(y,x) = val;
			break;
		case CV_32F:
		case CV_MAKETYPE(CV_32F,3):
			m.at<cv::Vec3f>(y,x) = val;
			break;
		case CV_64F:
		case CV_MAKETYPE(CV_64F,3):
			m.at<cv::Vec3d>(y,x) = val;
			break;
	}

}

bool direction(cv::Point p1, cv::Point p2,cv::Point p3)
{
	int p = -p2.x*p1.y + p3.x*p1.y + p1.x*p2.y - p3.x*p2.y - p1.x*p3.y + p2.x*p3.y; 

	if(p<0)
		return false;
	else
		return true;
}


void Utilities::autoContrast(cv::Mat img_in, cv::Mat &img_out)
{

	double min=0,max=0;

	std::vector<cv::Mat> bgr;
	cv::split(img_in,bgr);

	for(int i=0; i<3; i++)
	{
		cv::minMaxIdx(bgr[i],&min,&max);
		min += 255*0.05;
			
		double a = 255/(max-min);
		bgr[i]-=min;
		bgr[i]*=a;
	}
	
	cv::merge(bgr,img_out);
}

void Utilities::autoContrast(IplImage *img_in, IplImage *img_out)
{
	
	cv::Mat tmp_in = img_in;
	cv::Mat tmp_out = img_out;

	autoContrast(tmp_in,tmp_out);
	
}

void Utilities::exportMat(const char *path, cv::Mat m)
{

	std:: ofstream out; 
	out.open(path);
	
	for(int i =0; i < m.rows; i++)
	{

		for(int j = 0; j < m.cols; j++)
		{

			out<< Utilities::matGet2D(m,j,i)<<"\t";

		}
		out<<"\n";
	}
	out.close();
}

bool Utilities::line_lineIntersection(cv::Point3f p1, cv::Vec3f v1, cv::Point3f p2,cv::Vec3f v2,cv::Point3f &p)
{
	
	cv::Vec3f v12;
	v12 = p1 - p2;

	float v1_dot_v1 = v1.dot(v1);
	float v2_dot_v2 = v2.dot(v2);
	float v1_dot_v2 = v1.dot(v2); 
	float v12_dot_v1 = v12.dot(v1);
	float v12_dot_v2 = v12.dot(v2);


	float s, t, denom;

	
	denom = v1_dot_v1 * v2_dot_v2 - v1_dot_v2 * v1_dot_v2;

	if(abs(denom)<0.1)
		return false;

	s =  (v1_dot_v2/denom) * v12_dot_v2 - (v2_dot_v2/denom) * v12_dot_v1;
	t = -(v1_dot_v2/denom) * v12_dot_v1 + (v1_dot_v1/denom) * v12_dot_v2;

	p = (p1 + s*(cv::Point3f)v1 ) + (p2 + t*(cv::Point3f) v2);
	
	p = 0.5*p;

	return true;
}

int Utilities::accessMat(cv::Mat m, int x, int y, int i)
{
	
	return y*m.cols*m.channels() + x*m.channels() + i;

}

int Utilities::accessMat(cv::Mat m, int x, int y)
{
	
	return y*m.cols*m.channels() + x*m.channels();

}


void Utilities::folderScan(const char *path)
{	

	_chdir(path);

	WIN32_FIND_DATA data;
	HANDLE h;

	h = FindFirstFile( L"*.*", &data);

	if( h!=INVALID_HANDLE_VALUE ) 
	{
		int numOfFiles=0;

		do
		{
			char*  nPtr = new char [lstrlen( data.cFileName ) + 1];

			for( int i = 0; i < lstrlen( data.cFileName ); i++ )
				nPtr[i] = char( data.cFileName[i] );

			nPtr[lstrlen( data.cFileName )] = '\0';

			std::cout<<nPtr<<"\n";

		} 
		while(FindNextFile(h,&data));

	}


	for(int i=0; path[i]!=NULL; i++)
	{
		if(path[i] == '/')
			_chdir("../");
	}
}

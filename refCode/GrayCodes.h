//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#include "stdafx.h"

#ifndef __GRAY_CODES_H__
#define __GRAY_CODES_H__

#include <iostream>
#include <fstream>
using std::ofstream;
#include <opencv2/opencv.hpp>
#include <math.h>
#include "Utilities.h"

#define GRAY_MAX_NUM 100

class GrayCodes	{
	public: 
		///Constructor
		GrayCodes(int projW, int projH);

		///Destructor
		~GrayCodes();

		void unload();
		int getNumOfImgs();
		IplImage* getNextImg();
		
		IplImage* getImg(int num);

		void generateGrays();

		void save();
		static int grayToDec(cv::vector<bool> gray);
		int getNumOfRowBits();
		int getNumOfColBits();
		
	protected:
		IplImage* grayCodes[GRAY_MAX_NUM];
		IplImage* colorCodes[GRAY_MAX_NUM];

		void calNumOfImgs();

		void allocMemForImgs();

		bool imgsLoaded;

		int numOfImgs;
		int numOfRowImgs;
		int numOfColImgs;

		int currentImgNum;
		
		int height;
		int width;
};

#endif

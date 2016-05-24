//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#pragma once

#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
//#include <conio.h>
#include "GrayCodes.h"
#include "VirtualCamera.h"
#include "Utilities.h"
#include "PointCloudImage.h"
#include "SLS2012.h"

#define RECONSTRUCTION_SAMPLING_ON
#define SAMPLES_NUM 25
#define COL_GRAY_OFFSET 0
#define ROW_GRAY_OFFSET 0
#define EXPORT_MASKED_POINTS
#define SCALE 1

class Reconstructor
{
	public:
		Reconstructor(int numOfCams);
		~Reconstructor(void);

		void loadCameras();
		void runReconstruction();

		VirtualCamera	*cameras;
		PointCloudImage *points3DProjView;

		void setBlackThreshold(int val);
		void setWhiteThreshold(int val);
		void setImgPath(const char path1st[],const char path2nd[],const char extension[], int cam_no );
		void saveShadowImg(const char path[]);
		void saveDecodedRowImg(const char path[]);
		void saveDecodedColImg(const char path[]);

		void enableAutoContrast();
		void disableAutoContrast();
		void enableSavingAutoContrast();
		void disableSavingAutoContrast();
		void enableRaySampling();
		void disableRaySampling();

		void enableSavingShadowMask();
		void disableSavingShadowMask();
	private:
		
        
		int numOfCams;

		VirtualCamera *camera;//general functions use this instead of camera1 or camera2

		cv::vector<cv::Point> **camsPixels;
		cv::vector<cv::Point> *camPixels; //general functions use this instead of cam1Pixels or cam2Pixels

		void loadCamImgs(std::string folder,std::string prefix,std::string suffix);
		void unloadCamImgs();

		void computeShadows();

		void cam2WorldSpace(VirtualCamera cam, cv::Point3f &p);
		
		bool getProjPixel(int x, int y, cv::Point &p_out);

		void decodePaterns();

		void triangulation(cv::vector<cv::Point> *cam1Pixels, VirtualCamera cameras1, cv::vector<cv::Point> *cam2Pixels, VirtualCamera cameras2, int cam1index, int cam2index);
		
		std::string *camFolder;
		std::string *imgPrefix;
		std::string *imgSuffix;

		int numberOfImgs;
		int numOfColBits;
		int numOfRowBits;

		int blackThreshold;
		int whiteThreshold;

		cv::vector<cv::Mat> camImgs;
		cv::vector<cv::Mat> colorImgs;

		cv::Mat color;

		cv::Mat shadowMask;					//matrix with vals 0 and 1 , CV_8U , uchar

		cv::Mat decRows;
		cv::Mat decCols;

		bool *pathSet;

		bool autoContrast_;
		bool saveAutoContrast_;
		bool raySampling_;
		bool saveShadowMask_;

		//acress
		int ac(int x,int y)
		{
			return x*proj_h+y;
		}

};


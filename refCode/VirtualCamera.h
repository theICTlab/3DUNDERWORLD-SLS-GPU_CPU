//------------------------------------------------------------------------------------------------------------
//
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>

class VirtualCamera
{

public:

	VirtualCamera(void);
	~VirtualCamera(void);


	void loadDistortion(std::string path);
	void loadCameraMatrix(std::string path);
	void loadRotationMatrix(std::string path);
	void loadTranslationVector(std::string path);
	void computeProjectionMatrix();

	
	cv::Mat distortion;
	cv::Mat rotationMatrix;
	cv::Mat translationVector;
	cv::Mat projectionMatrix;
	cv::Mat cameraMatrix;

	cv::Point3f position;

	cv::Point2f fc; 
	cv::Point2f cc; 

	int width;
	int height;

private:

	int loadMatrix(cv::Mat &matrix,int s1,int s2 ,std::string file);

};


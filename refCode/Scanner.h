//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#pragma once


#include "stdafx.h"
#include "SLS2012.h"
#include <iostream>
#include <fstream>
using std::ofstream;
#include "cv.h"
#include "highgui.h"

#include "EDSDK.h"
#include "EDSDKErrors.h"
#include "EDSDKTypes.h"

#define STRICT
#include <windows.h>
#include <algorithm>
using std::min;
using std::max;
#include <gdiplus.h>
#include "GrayCodes.h"

#include <conio.h>

#include "CameraController.h"
#include "WebCam.h"
#include "CanonCamera.h"
#include <atlimage.h>
#include "Projector.h"

#define SCANNER_USE_WEBCAM true
#define SCANNER_USE_CANON false

#define SCAN_ONLY true
#define SCAN_N_CALIB false

class Scanner
{

public:
	Scanner(bool web);

	~Scanner(void);

	void scan(bool scanOnly);

	void capturePaterns(CameraController *cameras[],int camCount);
	
	bool capturePhotoAllCams(CameraController *cameras[],int camCount);

	bool capturePhotoSequence(CameraController *camera);
	//capture images and save them on path folder
	bool capturePhotoSequence(CameraController *camera, char* path);

	

private:

	bool web;


	cv::Mat whiteImg;

	GrayCodes *grayCodes;

	Projector *proj;
};


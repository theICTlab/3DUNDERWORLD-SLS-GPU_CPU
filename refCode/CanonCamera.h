//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#pragma once

#include "cv.h"
#include "highgui.h"
#include <conio.h>

#include "EDSDK.h"
#include "EDSDKErrors.h"
#include "EDSDKTypes.h"

#define STRICT
#include <windows.h>
#include <algorithm>
using std::min;
using std::max;
#include <gdiplus.h>

#include <iostream>
#include <fstream>
using std::ofstream;

#include <atlimage.h>




class CanonCamera
{
	public:
		CanonCamera(void);

		~CanonCamera(void);

		EdsError startLiveview();
		EdsError endLiveview();
		void UpdateView();

		int getNumOfCams();
		void captureImg();

	private:

		std::string windowName;
		IplImage* liveImage;
		EdsCameraRef camera;
		static int numOfCameras;
		int detectedCams;
		int camID;
		bool liveView;

};


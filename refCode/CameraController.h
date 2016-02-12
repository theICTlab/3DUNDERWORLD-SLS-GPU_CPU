//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#pragma once

#include "WebCam.h"
#include "CanonCamera.h"

class CameraController
{
	public:
		CameraController(bool webFlag);

		~CameraController(void);

		void startLiveview();
		void endLiveview();

		void UpdateView();

		void captureImg();

		//store pictures on path, curently available only for webCam
		void captureImg(char* path);

		int getNumOfCams();

		bool isWebCam();
		bool isCanonCam();

		void resetSaveCount();

	private:
		bool web;
		WebCam *webCam;
		CanonCamera *canonCam;
};


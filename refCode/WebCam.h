//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <fstream>
using std::ofstream;
#include "cv.h"
#include "highgui.h"
#include "GrayCodes.h"
#include "Projector.h"



class WebCam

{
	public:
		WebCam(int id=0,int camW=1600,int camH=1200);

		~WebCam(void);

		void startLiveview();
		void endLiveview();

		void UpdateView();

		void captureImg();
		void captureImg(char* dirPath);

		int getNumOfCams();
		void resetSaveCount();

	private:

		int saveCount;
		bool liveView;
		CvCapture *capture;
};


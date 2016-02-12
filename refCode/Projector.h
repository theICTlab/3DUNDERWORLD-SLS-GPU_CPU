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
#include "GrayCodes.h"

#include "SLS2012.h"

class Projector
{

	public:
	
		Projector(int projW,int projH);
	
		~Projector(void);

		int getHeight();
		int getWidth();

		void initProjectorWindow();

		void showImg(IplImage* img);
		void showImg(cv::Mat img);

	private:

		static int numOfProjectors;
		
		int projNum;
		int height;
		int width;

};


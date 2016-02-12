//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#include "StdAfx.h"
#include "WebCam.h"


WebCam::WebCam(int id, int camW, int camH)
{
	liveView=false;
	saveCount=1;

	capture = cvCaptureFromCAM(id);
	cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, camW ); 
	cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, camH);
}

void WebCam::resetSaveCount()
{
	saveCount = 1;
}

WebCam::~WebCam(void)
{
	cvReleaseCapture(&capture);
}

void WebCam::startLiveview()
{
	cvNamedWindow("Scanner Window",CV_WINDOW_AUTOSIZE);
	cvResizeWindow("Scanner Window",640,480);

	liveView=true;
}

void WebCam::endLiveview()
{
	cvDestroyWindow("Scanner Window");
	liveView=false;
}

void WebCam::UpdateView()
{

		if(!liveView)
			return;

		IplImage* retrieved_frame = cvRetrieveFrame(capture);

		IplImage* resampled_image = cvCreateImage(cvSize(640, 480), retrieved_frame->depth, retrieved_frame->nChannels);

		cvResize(retrieved_frame, resampled_image, CV_INTER_LINEAR);

		///Show the image in the window given
		cvShowImage("Scanner Window", resampled_image);

		cvReleaseImage(&resampled_image);
}

void WebCam::captureImg()
{
	std::stringstream path;

	path<<"scan/capture/"<< saveCount <<".png";

	cvRetrieveFrame(capture);

	cvSaveImage(path.str().c_str(), cvRetrieveFrame(capture));
	saveCount++;

}

void WebCam::captureImg(char* dirPath)
{
	std::stringstream path;

	path<<dirPath<< saveCount <<".png";
	
	IplImage* resampled_image=cvQueryFrame(capture);
	cvWaitKey(20);
	resampled_image=cvQueryFrame(capture);
	cvWaitKey(20);
	resampled_image=cvQueryFrame(capture);

	cvSaveImage(path.str().c_str(), resampled_image);
	saveCount++;

}

int WebCam::getNumOfCams()
{
	return 1;
}

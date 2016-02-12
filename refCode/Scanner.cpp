//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#include "StdAfx.h"
#include "Scanner.h"


Scanner::Scanner(bool webCam)
{
	web=webCam;
	proj=new Projector(proj_w,proj_h);
	whiteImg = cv::Mat::ones(proj_h,proj_w,CV_8U)*255;
	EdsInitializeSDK();
}


Scanner::~Scanner(void)
{
	EdsTerminateSDK();
		///Destroy the window
	delete proj;
}

bool Scanner::capturePhotoSequence(CameraController *camera)
{

	proj->showImg(whiteImg);

	//startCameraLiveView
	camera->startLiveview();

	
	int key;
	int count=0;

	while(true)
	{
		

		camera->UpdateView();

		key = cvWaitKey(10);

		///If enter is pressed then capture the image
		if (key == 13)
		{
			camera->captureImg();
			count++;
			std::cout<<count<<" image(s) have been captured.\n";
		}
		
		//32-> enter / 27-> esc 
		if(key == 32 || key == 27)
		{
			break;
		}

		
	}

	camera->endLiveview();

	cvWaitKey(100);
	if(key == 27)
		return 0;
	else 
		return 1;

}

//capture photos for calibration, photos will saved on path
//this is curently available only for 
bool Scanner::capturePhotoSequence(CameraController *camera, char* path)
{
	
	//if is a canon cam regular captureCalib is called
	if(camera->isCanonCam())
	{
		return capturePhotoSequence(camera);
	}

	proj->showImg(whiteImg);
	//startCameraLiveView
	camera->startLiveview();

	
	int key;

	int count=0;

	while(true)
	{
	
		camera->UpdateView();

		key = cvWaitKey(10);

		///If enter is pressed then capture the image
		if (key == 13)
		{
			camera->captureImg(path);
			count++;
			std::cout<<count<<" image(s) have been captured.\n";
		}
		
		//32-> enter / 27-> esc 
		if(key == 32 || key == 27)
		{
			break;
		}

	}

	camera->endLiveview();

	cvWaitKey(100);

	if(key == 27)
		return 0;
	else 
		return 1;

}

//project paterns and capture photos from all cameras
void Scanner::capturePaterns(CameraController *camera[],int camCount)
{

	std::cout << "\t-Generate Gray Codes..."  ;

	grayCodes= new GrayCodes(proj_w,proj_h);
	grayCodes->generateGrays();

	proj->showImg(grayCodes->getNextImg());

	cv::waitKey(100);

	std::cout << "done!\n" << std::endl;
	
	std::cout<<"System is ready to scan object. Press 'Enter' to start the Automatic Scanning\n";
		
	int key=0;
		
	while(key!=13)
		key = cvWaitKey(10);

	int grayCount=0;

	for(int i=0; i<camCount; i++)
	{
		camera[i]->resetSaveCount();
	}

	

	while(true)
	{
		int key;

		for(int i=0; i<camCount; i++)
		{
			if(camera[i]->isWebCam())
				camera[i]->captureImg("scan/dataSet/");
			else
				camera[i]->captureImg();
		}
			
		grayCount++;

		std::cout<<"Capture " << grayCount<<" of "<<grayCodes->getNumOfImgs()<< ".\n";

		if(grayCount==grayCodes->getNumOfImgs())
			break;

		proj->showImg(grayCodes->getNextImg());
		
		

		key=cvWaitKey(100);
	
		if(key == 27)
			break;
	}

	cvWaitKey(300);

}

bool Scanner::capturePhotoAllCams(CameraController *cameras[],int camCount)
{

	std::cout << "\nPress 'Enter' to capture 1 photo (both Cameras) for camera - camera calibration..\n" << std::endl;

	proj->showImg(whiteImg);

	for(int i =0; i<camCount; i++)
	{
		//startCameraLiveView
		cameras[i]->startLiveview();
	}

	int key;

	while(true)
	{
		
		for(int i =0; i<camCount; i++)
		{
			//startCameraLiveView
			cameras[i]->UpdateView();
		}
		
		key = cvWaitKey(10);

		///If enter is pressed then capture the image
		if (key == 13)
		{
			for(int i =0; i<camCount; i++)
			{
				cameras[i]->captureImg();
			}
			break;
		}
		
		//32-> enter / 27-> esc 
		if(key == 32 || key == 27)
		{
			break;
		}

		
	}

	for(int i =0; i<camCount; i++)
	{
		cameras[i]->endLiveview();
	}

	cvWaitKey(100);

	if(key == 27)
		return 0;
	else 
		return 1;
}


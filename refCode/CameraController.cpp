//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#include "StdAfx.h"
#include "CameraController.h"


CameraController::CameraController(bool webFlag)
{
	web=webFlag;

	if(webFlag == true)
		webCam = new WebCam(webCamID,cam_w,cam_h);
	else
		canonCam = new CanonCamera();
}

CameraController::~CameraController(void)
{
	if(web == true)
		delete webCam;
	else
		delete canonCam;
}

bool CameraController::isWebCam()
{
	return web;
}

bool CameraController::isCanonCam()
{
	return !web;
}


void CameraController::startLiveview()
{
	if(web == true)
		webCam->startLiveview();
	else
		canonCam->startLiveview();
}

void CameraController::endLiveview()
{
	if(web == true)
		webCam->endLiveview();
	else
		canonCam->endLiveview();
}

void CameraController::UpdateView()
{
	if(web==true)
		webCam->UpdateView();
	else
		canonCam->UpdateView();
}

void CameraController::captureImg()
{
	if(web==true)
		webCam->captureImg();
	else
		canonCam->captureImg();
}

void CameraController::captureImg(char* path)
{
	if(web)
		webCam->captureImg(path);
	else
		canonCam->captureImg();
}

int CameraController::getNumOfCams()
{
	if(web==true)
		return webCam->getNumOfCams();
	else
		return canonCam->getNumOfCams();
}

//only available for webCams
void CameraController::resetSaveCount()
{
	if(web==true)
		webCam->resetSaveCount();
}

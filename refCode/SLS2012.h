//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#include "stdafx.h"

#ifndef __SLS_2012_H__
#define __SLS_2012_H__

#include <opencv2/opencv.hpp>

extern int proj_h;
extern int proj_w;
extern int black_threshold;
extern int white_threshold;
extern int webCamID ;
extern int cam_w;
extern int cam_h;
extern cv::Point2i projectorWinPos;
extern bool autoContrast;
extern bool saveAutoContrast;
extern bool raySampling;

#endif

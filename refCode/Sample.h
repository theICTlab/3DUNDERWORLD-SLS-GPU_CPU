//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------


#pragma once

#include <math.h>
#include <stdlib.h>
#include "Sample.h"
#include "cv.h"
#include "RNG.h"

//random number generator
static RNG randomNumberGenerator;

class Sample
{

public:
	Sample(void);
	~Sample(void);
	
	//2D sampling
	static void Random(cv::Point2d *samples, int numberOfSamples);	
	//jitter assumes number of samples is perfect square
	static void Jitter(cv::Point2d *samples, int numberOfSamples);	
	static void NRooks(cv::Point2d *samples, int numberOfSamples);
	//multi-jitter assumes number of samples is perfect square
	static void MultiJitter(cv::Point2d *samples, int numberOfSamples);
	static void Shuffle(cv::Point2d *samples, int numberOfSamples);

	//Filtering methods
	static void BoxFilter(cv::Point2d *samples, int numberOfSamples);
	static void TentFilter(cv::Point2d *samples, int numberOfSamples);
	static void CubicSplineFilter(cv::Point2d *samples, int numberOfSamples);

	//1D sampling
	static void Random(double *samples, int numberOfSamples);
	static void Jitter(double *samples, int numberOfSamples);
	static void Shuffle(double *samples, int numberOfSamples);

	
	//helper function for cubicSplineFilter
	static inline double Solve(double r)	{
		double u = r;
		for (int i=0;i<5;i++)	{
			u = (double) (11.0*r+u*u*(6.0+u*(8.0-9.0*u)))/(4.0+12.0*u*(1.0+u*(1.0-u)));
		}
		return u;
	}

	//helper function to cubicSplineFilter
	static inline double CubicFilter(double x)	{
		if (x < ((double) 1.0/24.0))	{
			return pow((double) 24.0*x,(double) 0.25)-((double) 2.0);
		}
		else	{
			if (x <((double) 0.5))	{
				return Solve((double) 24.0*(x-1.0/24.0)/11.0)-((double) 1.0);
			}
			else	{
				if (x < ((double) 23.0/24.0))	{
					return ((double) 1.0) - Solve((double) 24.0*(23.0/24.0-x)/11.0);
				}
				else	{
					return ((double) 2.0) - pow((double) 24.0*(1.0-x),(double)0.25);
				}
			}
		}
	}
};


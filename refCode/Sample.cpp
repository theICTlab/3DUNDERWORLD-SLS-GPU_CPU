//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------


#include "StdAfx.h"
#include "Sample.h"


Sample::Sample(void)
{

}


Sample::~Sample(void)
{

}

void Sample::Random(cv::Point2d *samples, int numberOfSamples)	
{
	for (int i=0;i<numberOfSamples;i++)	{
		samples[i].x = randomNumberGenerator();
		samples[i].y = randomNumberGenerator();
	}
}

//assumes number of samples is a perfect square
void Sample::Jitter(cv::Point2d *samples, int numberOfSamples)	{
	int sqrtSamples = (int) sqrt((double) numberOfSamples);
	for (int i=0;i<sqrtSamples;i++)	{
		for (int j=0;j<sqrtSamples;j++)	
		{
			double r = randomNumberGenerator();
			double x = ((double) i + r)/(double) sqrtSamples;
			r = randomNumberGenerator();
			double y = ((double) j + r)/(double) sqrtSamples;
			samples[i*sqrtSamples+j].x = x;
			samples[i*sqrtSamples+j].y = y;
		}
	}
}

void Sample::NRooks(cv::Point2d *samples, int numberOfSamples)	{
	for (int i=0;i<numberOfSamples;i++)	{
		samples[i].x = ((double) i + randomNumberGenerator())/(double) numberOfSamples;
		samples[i].y = ((double) i + randomNumberGenerator())/(double) numberOfSamples;
	}
	//shuffle the x coords
	for (int i=numberOfSamples-2;i>=0;i--)	{
		int target = int(randomNumberGenerator()*(double) i);
		double temp = samples[i+1].x;
		samples[i+1].x = samples[target].x;
		samples[target].x = temp;
	}
}

//assumes number of samples is perfect square
void Sample::MultiJitter(cv::Point2d *samples, int numberOfSamples)	{
	int i,j;
	int sqrtSamples = (int) sqrt((double) numberOfSamples);
	double subcellWidth = (double) 1.0/((double) numberOfSamples);

	//Initialize the points to the "canonical" multi-jittered pattern
	for (i=0;i<sqrtSamples;i++)	{
		for (int j=0;j<sqrtSamples;j++)	{
			samples[i*sqrtSamples+j].x = i*sqrtSamples*subcellWidth + j*subcellWidth + randomNumberGenerator()*subcellWidth;
			samples[i*sqrtSamples+j].y = j*sqrtSamples*subcellWidth + i*subcellWidth + randomNumberGenerator()*subcellWidth;
		}
	}

	//shuffle x coords within each column and y coords within each row
	for (i=0;i<sqrtSamples;i++)	{
		for (j=0;j<sqrtSamples;j++)	{
			int k=j+int(randomNumberGenerator()*(sqrtSamples-j-1));
			double t = samples[i*sqrtSamples+j].x;
			samples[i*sqrtSamples+j].x = samples[i*sqrtSamples+k].x;
			samples[i*sqrtSamples+k].x = t;

			k = j+int(randomNumberGenerator()*(sqrtSamples-j-1));
			t = samples[j*sqrtSamples+i].y;
			samples[j*sqrtSamples+i].y = samples[k*sqrtSamples+i].y;
			samples[k*sqrtSamples+i].y = t;
		}
	}
}

void Sample::Shuffle(cv::Point2d *samples, int numberOfSamples)	{
	for (int i=numberOfSamples-2;i>=0;i--)	{
		int target = int(randomNumberGenerator()*(double)i);
		cv::Point2d temp = samples[i+1];
		samples[i+1] = samples[target];
		samples[target] = temp;
	}
}

void Sample::BoxFilter(cv::Point2d *samples, int numberOfSamples)	{
	for (int i=0;i<numberOfSamples;i++)	{
		samples[i].x = samples[i].x-0.5;
		samples[i].y = samples[i].y-0.5;
	}
}

void Sample::TentFilter(cv::Point2d *samples, int numberOfSamples)	{
	for (int i=0;i<numberOfSamples;i++)	{
		double x = samples[i].x;
		double y = samples[i].y;

		if (x < (double) 0.5)	{
			samples[i].x = sqrt(2.0*(double)x)-1.0;
		}
		else	{
			samples[i].x = 1.0-sqrt(2.0-2.0*(double)x);
		}

		if (y < (double) 0.5)	{
			samples[i].y = sqrt(2.0*(double)y)-1.0;
		}
		else	{
			samples[i].y = 1.0-sqrt(2.0-2.0*(double)y);
		}
	}
}

void Sample::CubicSplineFilter(cv::Point2d *samples, int numberOfSamples)	{
	for (int i=0;i<numberOfSamples;i++)	{
		double x = samples[i].x;
		double y = samples[i].y;

		samples[i].x = CubicFilter(x);
		samples[i].y = CubicFilter(y);
	}
}

void Sample::Random(double *samples, int numberOfSamples)	{
	for (int i=0;i<numberOfSamples;i++)	{
		samples[i] = randomNumberGenerator();
	}
}

void Sample::Jitter(double *samples, int numberOfSamples)	{
	for (int i=0;i<numberOfSamples;i++)	{
		samples[i] = ((double) i + randomNumberGenerator())/(double) numberOfSamples;
	}
}

void Sample::Shuffle(double *samples, int numberOfSamples)	{
	for (int i=numberOfSamples-2;i>=0;i--)	{
		int target = int(randomNumberGenerator()*(double)i);
		double temp = samples[i+1];
		samples[i+1] = samples[target];
		samples[target] = temp;
	}
}

//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#pragma once

#include "PointCloudImage.h"

class MeshCreator
{

public:

	MeshCreator(PointCloudImage *in);
	~MeshCreator(void);
	void exportObjMesh(std::string path);
	void exportPlyMesh(std::string path, bool gridFormat);

private:

	int *pixelNum;
	PointCloudImage *cloud;
	int MeshCreator::access(int i,int j);
	int MeshCreator::access(int i,int j, int z);
	int w;
	int h;

};


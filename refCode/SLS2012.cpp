//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "SLS2012.h"
#include "GrayCodes.h"
#include "Projector.h"
#include "Scanner.h"
#include "Reconstructor.h"
#include "PointCloudImage.h"
#include "MeshCreator.h"
#include "CameraCalibration.h"
#include <direct.h>
#include <windows.h>


int proj_h;
int proj_w;
int black_threshold;
bool autoContrast;
bool saveAutoContrast;
bool raySampling;
int white_threshold;
int webCamID;
int cam_w;
int cam_h;
cv::Point2i projectorWinPos;

bool exportPly;
bool exportPlyGrid;
bool exportObj;
bool exportShadowMask;


void projectGraysOnly()
{

	std::cout << "Generating Gray Codes..."  ;
	GrayCodes *grayCode= new GrayCodes(proj_w,proj_h);
	grayCode->generateGrays();
	std::cout << "done!\n"  ;
	std::cout << "Press 'Enter to change projected code"  ;

	Projector *proj=new Projector(proj_w,proj_h);

	int i=0;
	int key = cvWaitKey(10);
	while(true)
	{
		key = cvWaitKey(10);
		proj->showImg( grayCode->getImg(i));

		if(key == 13)
			i++;

		if(i == (grayCode->getNumOfImgs()))
			i=0;
		if(key == 27)
			break;
	}

}

int renameDataSet()
{	

	WIN32_FIND_DATA data;
	HANDLE h;

	char sel = 0;

	while( sel!='j' && sel != 't' && sel != 'p')
	{
		std::cout<<"Please specify image format, 'j' for jpg, 't' for tif or 'p' for png.\n"; 
		std::cin>>sel;
	}
	char *format; 

	if(sel=='j')
	{
		h = FindFirstFile(L"*.jpg",&data);
		format = ".jpg";
	}
	else if(sel=='t')
	{
		format = ".tif";
		h = FindFirstFile(L"*.tif",&data);
	}
	else if(sel=='p')
	{
		format = ".png";
		h = FindFirstFile(L"*.png",&data);
	}

	int count = 1;

	std::vector<std::string> list;

	if( h!=INVALID_HANDLE_VALUE ) 
	{
		int numOfFiles=0;

		do
		{
			char*  nPtr = new char [lstrlen( data.cFileName ) + 1];

			for( int i = 0; i < lstrlen( data.cFileName ); i++ )
				nPtr[i] = char( data.cFileName[i] );

			nPtr[lstrlen( data.cFileName )] = '\0';

			list.push_back(nPtr);

		} 
		while(FindNextFile(h,&data));

		for(int i = 0; i<list.size(); i++)
		{
			

			std::stringstream path1,path2;
			path1 << list[i];
			
			if(count < 10)
				path2<<'0';

			path2 << count << format;
		
			(char *)data.cFileName;

			bool a = rename(path1.str().c_str(), path2.str().c_str());

			std::cout<<path1.str().c_str()<<" to "<<path2.str().c_str()<<"\n";

			count++;

		} 
		while(FindNextFile(h,&data) && count <= numOfFiles);
	} 
	else 
		std::cout << "Error: No such folder." << std::endl;
	
	FindClose(h);
	
	return 0;
}

void createConfigurationFile(char* path)
{
	//set default settings
	projectorWinPos.x = proj_w + 300;
	projectorWinPos.y = -20;
	proj_h = 768;
	proj_w = 1024;
	black_threshold = 40;
	white_threshold = 5;
	webCamID = 0;
	cam_w=1600;
	cam_h=1200;
	autoContrast = true;
	saveAutoContrast = false;
	raySampling = true;
	exportObj = false;
	exportPly = false;
	exportPlyGrid = true;
	exportShadowMask = false;

	cv::FileStorage fs(path, cv::FileStorage::WRITE);

	fs << "Projector" << "{:";
		fs << "Width" << proj_w << "Height" << proj_h ;
	fs<<"}";

	fs << "Camera" << "{:";
		fs << "ID"<< webCamID << "Width" << cam_w << "Height" << cam_h ;
	fs<<"}";

	fs << "ProjectorWindow" << "{:";
		fs << "Position" << "{:";
			fs << "x" << projectorWinPos.x << "y" << projectorWinPos.y ;
		fs<<"}";
	fs<<"}";
	
	fs << "Reconstruction" << "{:";
		fs<<"AutoContrast"<<autoContrast;
		fs<<"SaveAutoContrastImages"<<saveAutoContrast;
		fs<<"RaySampling"<<raySampling;
		fs<<"blackThreshold"<<black_threshold;
		fs<<"whiteThreshold"<<white_threshold;
	fs<<"}";
	fs << "Export" << "{:";
		fs<<"Obj"<<exportObj;
		fs<<"Ply"<<exportPly;
		fs<<"GridPly"<<exportPlyGrid;
		fs<<"ShadowMask"<<exportShadowMask;
	fs<<"}";

	fs.release();
}


bool loadConfigurations()
{
	cv::FileStorage fs("slsConfig.xml", cv::FileStorage::READ);

	if(!fs.isOpened())
	{
		std::cout << "Failed to open Configuration File. " << std::endl;
		return false;
	}

	cv::FileNode node = fs["Projector"];
	
		node["Width"] >> proj_w;
		node["Height"]>> proj_h;
	
	node= fs["Camera"];
		node["ID"] >> webCamID;
		node["Width"] >> cam_w;
		node["Height"] >> cam_h;
	
	node= fs["ProjectorWindow"];
		node = node["Position"];
			node["x"] >> projectorWinPos.x;
			node["y"] >> projectorWinPos.y;
	
	node= fs["Reconstruction"];
		node["blackThreshold"] >> black_threshold;
		node["whiteThreshold"] >> white_threshold;
		node["AutoContrast"] >> autoContrast;
		node["SaveAutoContrastImages"] >> saveAutoContrast;
		node["RaySampling"] >> raySampling;
	
	node = fs["Export"];
		node["Obj"]>>exportObj;
		node["Ply"]>>exportPly;
		node["GridPly"]>>exportPlyGrid;
		node["ShadowMask"]>>exportShadowMask;

	fs.release();

	return true;
}

void printCopyRight()
{
	std::cout<<"\n";
	std::cout<<"---------------------------------------------------------------------\n";
	std::cout<<"* Copyright © 2010-2013 Immersive and Creative Technologies Lab,    *\n";
	std::cout<<"* Cyprus University of Technology                                   *\n";
	std::cout<<"* Link: http://ict.cut.ac.cy                                        *\n";
	std::cout<<"* Software developer(s): Kyriakos Herakleous                        *\n";
	std::cout<<"* Researcher(s): Kyriakos Herakleous, Charalambos Poullis           *\n";
	std::cout<<"*                                                                   *\n";
	std::cout<<"* This work is licensed under a Creative Commons                    *\n";
	std::cout<<"* Attribution-NonCommercial-ShareAlike 3.0 Unported License.        *\n";
	std::cout<<"* Link: http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US *\n";
	std::cout<<"---------------------------------------------------------------------\n\n";

}

void reconstruct()
{
	//change directory
	_chdir("reconstruction/");
	
	std::string path;
	std::string extentsion;

	int num = 0;
	while(num<2)
	{
		std::cout<<"Please specify the number of cameras.\n"; 
		std::cin>>num;
	}

	Reconstructor *reconstructor= new Reconstructor(num);
 
	char sel=0;

	//load dataset
	while( sel!='j' && sel != 't' && sel != 'p')
	{
		std::cout<<"Please specify image format, 'j' for jpg, 't' for tif or 'p' for png.\n"; 
		std::cin>>sel;
	}
	
	if(sel == 'j')
	{
		extentsion = ".jpg";
	}
	else if(sel == 't')
	{
		extentsion = ".tif";
	}
	else if(sel == 'p')
	{
		extentsion = ".png";
	}

	//set camera's paths
	for(int i=0; i<num; i++)
	{
		std::string p ="dataset/Cam";
		p += '0'+ (i+1);
		p += '/';
		reconstructor->setImgPath(p.c_str(),"",".jpg",i);
	}

	//load projector and camera paramiters

    /*
     * Load Camera paramters
     */
	reconstructor->loadCameras();

	//set reconstuction parameters
	reconstructor->setBlackThreshold(black_threshold);
	reconstructor->setWhiteThreshold(white_threshold);

	
	if(autoContrast)
		reconstructor->enableAutoContrast();
	else
		reconstructor->disableAutoContrast();

	if(saveAutoContrast)
		reconstructor->enableSavingAutoContrast();
	else
		reconstructor->disableSavingAutoContrast();

	if(raySampling)
		reconstructor->enableRaySampling();
	else
		reconstructor->disableRaySampling();

	//recontruct
	reconstructor->runReconstruction();
	
	//Export mesh
	MeshCreator *meshCreator=new MeshCreator(reconstructor->points3DProjView);

	if(exportObj)
		meshCreator->exportObjMesh("output/projector_view.obj");

	if(exportPly || !(exportObj || exportPly || exportPlyGrid))
		meshCreator->exportPlyMesh("output/projector_view.ply",false);

	if(exportPlyGrid)
		meshCreator->exportPlyMesh("output/projector_view.grid.ply",true);

	delete meshCreator;
	delete reconstructor;

}

void generateGrayCodes()
{
	//change directory
	_chdir("gray/");

	std::cout << "Generating Gray Codes..."  ;
	GrayCodes *gray=new GrayCodes(proj_w,proj_h);
	gray->generateGrays();
	std::cout << "done!\n"  ;

	std::cout << "Saving..."  ;
	gray->save();
	std::cout << "done!\n"  ;

	delete gray;
}

void calibration()
{
	//change directory
	_chdir("calibration/");
	
	int sel = 0;
	while( sel <= 0)
	{
		std::cout<<"Please specify how many cameras you want to calibrate "; 
		std::cin>>sel;
	}

	for(int i=1; i<=sel; i++)
	{
		CameraCalibration *calib = new CameraCalibration();

		std::string path = "camera";
		path += '0' + i;
		path += '/';

		//load images
		calib->loadCameraImgs(path.c_str());
	
		calib->extractImageCorners();
		calib->calibrateCamera();
	
		calib->findCameraExtrisics();

		//export txt files
		std::string file_name;
		
		path += "output/";

		file_name =  path.c_str();
		file_name += "cam_matrix.txt";

		calib->exportTxtFiles(file_name.c_str(),CAMCALIB_OUT_MATRIX);

		file_name =  path.c_str();
		file_name += "cam_distortion.txt";

		calib->exportTxtFiles(file_name.c_str(),CAMCALIB_OUT_DISTORTION);

		file_name =  path.c_str();
		file_name += "cam_rotation_matrix.txt";

		calib->exportTxtFiles(file_name.c_str(),CAMCALIB_OUT_ROTATION);

		file_name =  path.c_str();
		file_name += "cam_trans_vectror.txt";

		calib->exportTxtFiles(file_name.c_str(),CAMCALIB_OUT_TRANSLATION);

		file_name =  path.c_str();
		file_name += "calib.xml";
		calib->saveCalibData(file_name.c_str());

		// show data on consol
		calib->printData();
	}
	getchar();
}


void scan()
{
	Scanner *scanner;
	scanner = new Scanner(SCANNER_USE_CANON);

	int numOfCams;
	CameraController *tmp = new CameraController(false);
	numOfCams = tmp->getNumOfCams();

	CameraController **cameras = new CameraController*[numOfCams];

	cameras[0] = tmp;

	for(int i=1; i<numOfCams; i++)
	{
		cameras[i] = new CameraController(false);
	}

	scanner->capturePaterns(cameras,numOfCams);

	for(int i=0; i<numOfCams; i++)
	{
		delete cameras[i];
	}
}


void captureCalibrationImagesAndScan()
{
	Scanner *scanner;
	scanner = new Scanner(SCANNER_USE_CANON);

	int numOfCams;
	CameraController *tmp = new CameraController(false);
	numOfCams = tmp->getNumOfCams();

	CameraController **cameras = new CameraController*[numOfCams];

	cameras[0] = tmp;

	for(int i=1; i<numOfCams; i++)
	{
		cameras[i] = new CameraController(false);
	}

	bool continue_val = true;

	//take calibration pictures
	for(int i=0; i<numOfCams; i++)
	{
			
		cvWaitKey(1);
			
		std::cout << "\nPress 'Enter' to capture photos for camera calibration. When you are done press 'Space'.\n" << std::endl;

		//capture calibration images with camera [i]
		continue_val = scanner->capturePhotoSequence(cameras[i]);

		//if user dont want ot continue break
		if(!continue_val)
		{
			for(int i=0; i<numOfCams; i++)
			{
				delete cameras[i];
			}
			return;
		}
			
	}

	continue_val = scanner->capturePhotoAllCams(cameras,numOfCams);

	//if user dont want ot continue break
	if(!continue_val)
	{
		for(int i=0; i<numOfCams; i++)
		{
			delete cameras[i];
		}
		return;
	}

	scanner->capturePaterns(cameras,numOfCams);

	for(int i=0; i<numOfCams; i++)
	{
		delete cameras[i];
	}

}

void rename()
{

	int sel = 0;
	while( sel <= 0)
	{
		std::cout<<"Please specify how many cameras you are using "; 
		std::cin>>sel;
	}

	_chdir("reconstruction/dataSet/");

	for(int i=0; i<sel; i++)
	{
		std::string p = "cam";
		p += '0'+(i+1);
		p += '/';
		_chdir(p.c_str());
		renameDataSet();
		_chdir("../");

	}

}

int _tmain(int argc, _TCHAR* argv[])
{
	
	printCopyRight();

	//load configurations
	if(!loadConfigurations())
	{
		std::cout<<"A new one with default settings has been created.\n\n";
		createConfigurationFile("slsConfig.xml");
		loadConfigurations();
	}

	std::cout<<"Task List\n1. Rename the dataSet \n2. Reconstruct\n3. Scan With Canon Camera(with Calib)\n4. Scan With Canon Camera(Scan Only)\n5. GrayCodes Projection\n6. Generate and Save gray codes\n7. Calibration \n8. Create Confiquration xml file with default settings  \n\n Pleace select task! ";

	int select;
	std::cin>>select;

	//clear console
	system("cls");
	
	Scanner *scanner;

	switch(select)
	{
		case 1:
			rename();
			break;
		//Reconstruction
		case 2:
			reconstruct();
			break;
		//Scan With Canon Camera(with Calib)
		case 3:
			captureCalibrationImagesAndScan();
			break;
		//Scan With Canon Camera(Scan Only)
		case 4:
			scan();
			break;
		//GrayCodes Projection
		case 5:
			projectGraysOnly();
			break;
		//Generate gray codes
		case 6:
			generateGrayCodes();
			break;
		//Calibration
		case 7:
			calibration();
			break;
		//Create Default Configuration Settings XML file
		case 8:
			createConfigurationFile("slsConfigDefault.xml");
			break;
		
	}

	std::cout<<"\nPress any key to exit.";
	getch();

	return 1;
}




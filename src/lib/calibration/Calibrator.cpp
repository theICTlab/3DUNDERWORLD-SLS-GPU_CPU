#include <calibration/Calibrator.hpp>

namespace SLS
{
    // Callbacks !!
void calib_board_corners_mouse_callback( int event, int x, int y, int flags, void* param )
{
	
	cv::vector<cv::Point2f> *corners= (cv::vector<cv::Point2f>*) param;

	switch( event )
	{
		
		case CV_EVENT_LBUTTONDOWN:
			if(corners->size() ==4)
				break;
			corners->push_back(cv::Point(x,y));
			break;

	}
}

void image_point_return( int event, int x, int y, int flags, void* param )
{

	CvScalar *point= (CvScalar*) param;

	switch( event )
	{
		case CV_EVENT_LBUTTONDOWN:
			
			point->val[0]=x;
			point->val[1]=y;
			point->val[2]=1;
			break;
	}

}

cv::vector<cv::Point2f>  manualMarkCheckBoard(cv::Mat img)
{
	
	cv::vector<cv::Point2f> corners;
		
	cv::namedWindow("Mark Calibration Board",CV_WINDOW_NORMAL);
	cv::resizeWindow("Mark Calibration Board",800,600);

	//Set a mouse callback
	cv::setMouseCallback( "Mark Calibration Board",calib_board_corners_mouse_callback, (void*) &corners);

	bool ok = false;

	while(!ok)
	{
		corners.clear();
		cv::resizeWindow("Mark Calibration Board",800,600);

		int curNumOfCorners=0;
			
		cv::Mat img_copy ;
		img.copyTo(img_copy);

		system("cls");

		std::cout<<"Please click on the 4 extrime corners of the board, starting from the top left corner\n";

		cv::Point2f rectSize(20,20);

		while(corners.size()<4)
		{
			//draw selected corners and conection lines
			if(curNumOfCorners<corners.size())
			{
				int s = corners.size();
					
				cv::rectangle(img_copy,	corners[s-1] - rectSize,corners[s-1] + rectSize,cvScalar(0,0,255),3);
				
				if(!(corners.size()==1))
				{
					cv::line(img_copy, corners[s-1],corners[s-2],cvScalar(0,0,255),3);
				}
				
				curNumOfCorners++;
				
			}

			cv::imshow("Mark Calibration Board", img_copy);
			cv::waitKey(2);
		}

		//Draw corners and lines		
		cv::rectangle( img_copy,	corners[3] - rectSize, corners[3] + rectSize, cvScalar(0,0,255), 3);
		cv::line(img_copy, corners[3],corners[2],cvScalar(0,0,255),10);
		cv::line(img_copy, corners[3],corners[0],cvScalar(0,0,255),10);
		
		system("cls");
		std::cout<<"Press 'Enter' to continue or 'ESC' to select a new area!\n";

		int key = 0;

		//wait for enter or esc key press
		while( key!=27 && key!=13 )
		{
			cv::imshow("Mark Calibration Board", img_copy );
			key = cv::waitKey();
		}

		//if enter set ok as true to stop the loop or repeat the selection process
		if(key == 13)
			ok = true;
		else
			ok = false;

		img_copy.release();
			
	}

	cv::destroyWindow("Mark Calibration Board");
		
	return corners;
}
void drawOutsideOfRectangle(cv::Mat img,cv::vector<cv::Point2f> rectanglePoints, float color)
{

	std::vector<cv::Point> corners;
	for(int i=0; i<rectanglePoints.size(); i++)
	{
		corners.push_back(rectanglePoints[i]);
	}

	cv::Mat mask(img.size(),img.type());
	cv::Mat background(img.size(),img.type());
	
	mask =  1;
	cv::fillConvexPoly(mask, corners ,cv::Scalar(0));

	background = color;
	background.copyTo(img,mask);
	
}
float markWhite(cv::Mat img)
{
	
		float white;
		cv::namedWindow("Mark White",CV_WINDOW_NORMAL);
		cv::resizeWindow("Mark White",800,600);

		cv::Scalar point;

		// Set a mouse callback
		cv::setMouseCallback( "Mark White",image_point_return, (void*) &point);

		bool ok = false;
		
		while(!ok)
		{
			cv::Mat img_copy;
			img.copyTo(img_copy);
			
			cv::resizeWindow("Mark White",800,600);
			
			int pointsCount=0;
			point.val[2]=0;
			
			while(pointsCount==0)
			{
				if(point.val[2]==1)
				{
					cv::rectangle(img_copy, cvPoint(point.val[0]-10,point.val[1]-10),cvPoint(point.val[0]+10,point.val[1]+10),cvScalar(0,0,255),3);
					
					white = img.at<uchar>(point.val[1],point.val[0]);
					
					pointsCount++;
					point.val[2]=0;
				}

				cv::imshow("Mark White", img_copy );
				cv::waitKey(2);
			}
							

			int key = 0;

			while(key != 27 && key != 13)
			{
				cv::imshow("Mark White", img_copy );
				key=cv::waitKey();
			}

			if(key==13)
				ok=true;
			else
				ok=false;

			img_copy.release();
		}

		cvDestroyWindow("Mark White");
		

		return white;
}
bool findCornersInCamImg(cv::Mat img,cv::vector<cv::Point2f> *camCorners,cv::vector<cv::Point3f> *objCorners, cv::Size squareSize)
{

	//copy camera img
	cv::Mat img_grey;
	cv::Mat img_copy;
	img.copyTo(img_copy);

	int numOfCornersX;
	int numOfCornersY;

	bool found=false;

	//find the corners
	while(!found)
	{
		//convert the copy to gray
		cv::cvtColor( img, img_grey, CV_RGB2GRAY );
		img.copyTo(img_copy);

		//ask user to mark 4 corners of the checkboard
		cv::vector<cv::Point2f> chessBoardCorners = manualMarkCheckBoard(img_copy);

		//ask user to mark a white point on checkboard
		float color = markWhite(img_grey);

		drawOutsideOfRectangle(img_grey,chessBoardCorners, color);

		//show img to user
		cv::namedWindow("Calibration",CV_WINDOW_NORMAL);
		cv::resizeWindow("Calibration",800,600);

		cv::imshow("Calibration",img_grey);
		cv::waitKey(20);

		system("cls");

		//ask the number of squares in img
		std::cout<<"Give number of squares on x axis: ";
		std::cin>>numOfCornersX;
		std::cout<<"Give number of squares on y axis: ";
		std::cin>>numOfCornersY;

		if(numOfCornersX<=0 || numOfCornersY<=0)
			break;

		if(numOfCornersX<=3 || numOfCornersY<=3)
		{
			std::cout<<"Board size must be >3\n";
			continue;
		}

		numOfCornersX--;
		numOfCornersY--;
		
		
		found=cv::findChessboardCorners(img_grey, cvSize(numOfCornersX,numOfCornersY), *camCorners, CV_CALIB_CB_ADAPTIVE_THRESH );

		std::cout<<"found = "<<camCorners->size()<<"\n";

		cv::drawChessboardCorners(img_copy, cvSize(numOfCornersX,numOfCornersY), *camCorners, found);

		int key = cv::waitKey(1);

		if(key==13)
			break;

		std::cout<<"\nPress 'Enter' to continue or 'ESC' to repeat the procedure.\n";

		while(found)
		{
			cv::imshow("Calibration", img_copy );

			key = cv::waitKey(1);

			if(key==27)
				found=false;
			if(key==13)
			break;
		}
	//if corners found find subPixel
	if(found)
	{

		//convert the copy to gray
		cv::cvtColor( img, img_grey, CV_RGB2GRAY );

		//find sub pix of the corners
		cv::cornerSubPix(img_grey, *camCorners, cvSize(20,20), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));

		system("cls");

		if(squareSize.height == 0)
		{
			std::cout<<"Give square height in mm: ";
			std::cin>>squareSize.height;

			std::cout<<"Give square width in mm: ";
			std::cin>>squareSize.width;
		}

		for(int i=0; i<numOfCornersY ; i++)
		{
			for(int j=0; j<numOfCornersX; j++)
			{
				cv::Point3f p;
				p.x = j*squareSize.width;
				p.y = i*squareSize.height;
				p.z = 0;
				objCorners->push_back(p);
			}
		}

	}

	cv::destroyWindow("Calibration");
	
	
	return found;
}

	
void Calibrator::Calibrate(FileReader *cam, const std::string& calibImgsDir, const std::string& calibFile)
{
    //Load calibration images
    cam->loadImages(calibImgsDir);

    //Extract corners
	cv::vector<cv::vector<cv::Point2f>> imgBoardCornersCam;
	cv::vector<cv::vector<cv::Point3f>> objBoardCornersCam;
    imgBoardCornersCam.clear();
    objBoardCornersCam.clear();
    for (size_t i=0; i<cam->getNumFrames(); i++)
    {
		cv::vector<cv::Point2f> cCam;
		cv::vector<cv::Point3f> cObj;
        auto &img = cam->getNextFrame();
		findCornersInCamImg(img, &imgBoardCornersCam, &objBoardCornersCam, cv::Size(20, 20));
    }
}
} // namespace SLS

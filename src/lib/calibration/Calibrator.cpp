#include <calibration/Calibrator.hpp>
#include <thread>

namespace SLS
{
    std::condition_variable Calibrator::cv;
    bool Calibrator::closeAsynImg=false;
    // Callbacks !!
    void calib_board_corners_mouse_callback( int event, int x, int y, int flags, void* param )
    {

        cv::vector<cv::Point2f> *corners= (cv::vector<cv::Point2f>*) param;

        switch( event )
        {
            case CV_EVENT_LBUTTONDOWN:
                if(corners->size() < 4)
                    corners->push_back(cv::Point(x,y));
                else
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

    cv::vector<cv::Point2f>  Calibrator::manualMarkCheckBoard(cv::Mat img)
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

            //system("clear");

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
                //Showing in the loop
                cv::waitKey(5);
            }

            //Draw corners and lines		
            cv::rectangle( img_copy,	corners[3] - rectSize, corners[3] + rectSize, cvScalar(0,0,255), 3);
            cv::line(img_copy, corners[3],corners[2],cvScalar(0,0,255),10);
            cv::line(img_copy, corners[3],corners[0],cvScalar(0,0,255),10);

            //system("clear");
            std::cout<<"Press 'n' to continue or 'r' to select a new area!\n";

            int key = 0;

            //wait for enter or esc key press
            while( key!='n' && key!='r' )
            {
                cv::imshow("Mark Calibration Board", img_copy );
                key = cv::waitKey(0);
            }

            //if enter set ok as true to stop the loop or repeat the selection process
            if(key == 'n')
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
        for(unsigned i=0; i<rectanglePoints.size(); i++)
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
    float markWhite(const cv::Mat &img)
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
            cv::Mat img_copy=img.clone();
            cv::resizeWindow("Mark White",800,600);

            int pointsCount=0;
            point.val[2]=0;
            while(pointsCount<1)
            {
                if(point.val[2]==1)
                {
                    cv::rectangle(img_copy, cvPoint(point.val[0]-10,point.val[1]-10),cvPoint(point.val[0]+10,point.val[1]+10),cvScalar(0,0,255),3);

                    white = img.at<uchar>(point.val[1],point.val[0]);

                    pointsCount++;
                    point.val[2]=0;
                }
                cv::imshow("Mark White", img_copy);
                cv::waitKey(5);
            }


            int key = 0;

            while(key != 'n' && key != 'r')
            {
                cv::imshow("Mark White", img_copy );
                key=cv::waitKey();
            }

            if(key=='n')
                ok=true;
            else
                ok=false;

            img_copy.release();
        }

        cvDestroyWindow("Mark White");


        return white;
    }
    
    bool Calibrator::findCornersInCamImg(const cv::Mat &img,cv::vector<cv::Point2f> &camCorners,cv::vector<cv::Point3f> &objCorners, cv::Size squareSize)
    {

        cv::Mat img_copy=img.clone();   // keep a cpy of it
        cv::Mat img_grey;
        //copy camera img
        int numOfCornersX;
        int numOfCornersY;
        bool found=false;

        //find the corners
        while(!found)
        {
            img_grey=img.clone();
            //ask user to mark 4 corners of the checkerboard
            cv::vector<cv::Point2f> chessBoardCorners = manualMarkCheckBoard(img_copy);

            //ask user to mark a white point on checkerboard
            float color = markWhite(img_grey);

            // Modify rectangle
            drawOutsideOfRectangle(img_grey,chessBoardCorners, color);

            //show img to user
            // Create an async task to show image
            cv::namedWindow("Calibration",CV_WINDOW_NORMAL);
            cv::resizeWindow("Calibration",800,600);
            closeAsynImg = false;
            // Kick the thread
            std::thread imgAsync( showImgAsync, img_grey, "Calibration");

            system("clear");
            //ask the number of squares in img
            std::cout<<"Give number of squares on x axis: ";
            std::cin>>numOfCornersX;
            std::cout<<"Give number of squares on y axis: ";
            std::cin>>numOfCornersY;

            // Close the thread
            closeAsynImg = true;
            // Notify it
            cv.notify_one();
            // Wait for it to finish
            imgAsync.join();

            if(numOfCornersX<=0 || numOfCornersY<=0)
                break;

            if(numOfCornersX<=3 || numOfCornersY<=3)
            {
                std::cout<<"Board size must be >3\n";
                continue;
            }

            numOfCornersX--;
            numOfCornersY--;


            found=cv::findChessboardCorners(img_grey, cvSize(numOfCornersX,numOfCornersY), camCorners, CV_CALIB_CB_ADAPTIVE_THRESH );

            std::cout<<"found = "<<camCorners.size()<<"\n";
            if (!found)
            {
                std::cout<<"DEBUGGGGGGGGGGGGGGGGGGGGG\n";
                std::cout<<numOfCornersX<<"\t"<<numOfCornersY<<std::endl;
                std::cout<<camCorners.size()<<std::endl;
                imshow("DEBUG", img_grey);
                cv::waitKey(0);
            }
                


            int key = cv::waitKey(5);

            if(key=='n')
                break;

            std::cout<<"\nPress 'Enter' to continue or 'ESC' to repeat the procedure.\n";

            while(found)
            {
                cv::drawChessboardCorners(img_copy, cvSize(numOfCornersX,numOfCornersY), camCorners, found);
                cv::imshow("Calibration", img_copy );

                key = cv::waitKey(0);

                if(key=='r')
                    found=false;
                if(key=='n')
                    break;
            }
            if (!found)
            {
                std::cout<<"No squres found, do it again ...\n";
                cv::destroyWindow("Calibration");
                img_grey.release();
            }
        }
        //if corners found find subPixel
        if(found)
        {
            //find sub pix of the corners
            cv::cornerSubPix(img_grey, camCorners, cvSize(20,20), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));

            system("clear");

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
                    objCorners.push_back(p);
                }
            }

        }
        img_grey.release();
        img_copy.release();

        cv::destroyWindow("Calibration");
        return found;
    }


    void Calibrator::Calibrate(FileReader *cam, const std::string& calibImgsDir, const std::string& calibFile)
    {
        //Value to generate
        cv::Mat camMatrix;
        cv::Mat distortion;
        cv::Mat rotationMatrix;
        cv::Mat translationVector;
        cv::Size squareSize(33, 33);
        cv::Size camImageSize;

        //Load calibration images
        cam->loadImages(calibImgsDir);
        size_t width, height;
        cam->getResolution(width, height); 
        camImageSize.height = height;
        camImageSize.width = width;


        //Extract corners
        cv::vector<cv::vector<cv::Point2f>> imgBoardCornersCam;
        cv::vector<cv::vector<cv::Point3f>> objBoardCornersCam;
        imgBoardCornersCam.clear();
        objBoardCornersCam.clear();
        for (size_t i=0; i<cam->getNumFrames()-1; i++)
        {
            cv::vector<cv::Point2f> cCam;
            cv::vector<cv::Point3f> cObj;
            auto img = cam->getNextFrame().clone();
            findCornersInCamImg(img, cCam, cObj, squareSize);
            if (cCam.size())
            {
                imgBoardCornersCam.push_back(cCam);
                objBoardCornersCam.push_back(cObj);
            }
        }
        cv::vector<cv::Mat> camRotationVectors;
        cv::vector<cv::Mat> camTranslationVectors;

        // Find intrinsic
        cv::calibrateCamera(objBoardCornersCam,imgBoardCornersCam,camImageSize,camMatrix, distortion, camRotationVectors,camTranslationVectors,0,
		cv::TermCriteria( (cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 30, DBL_EPSILON) );

        // Find extrinsic
        auto extImg = cam->getNextFrame().clone();
        cv::vector<cv::Point2f> imgPoints;
        cv::vector<cv::Point3f> objPoints3D;
        findCornersInCamImg(extImg, imgPoints, objPoints3D, squareSize );
        cv::Mat rVec;

        //find extrinsics rotation & translation
        bool r = cv::solvePnP(objPoints3D,imgPoints,camMatrix,distortion,rVec,translationVector);
        cv::Rodrigues(rVec,rotationMatrix);
        std::cout<<rotationMatrix<<"\n\n\n"<<translationVector<<"\n\n\n";

        //          Save calib data	
        cv::FileStorage fs(calibFile, cv::FileStorage::WRITE);

        fs << "Camera" << "{:";
        fs<< "Calibrated" << r << "Matrix" << camMatrix << "Distortion" << distortion<<"Translation"<<translationVector<<"Rotation"<<rotationMatrix;
        fs<<"Height" << camImageSize.height<<"Width" << camImageSize.width;
        fs<<"}";


        fs << "BoardSquare" << "{:";
        fs << "Height" << squareSize.height << "Width" << squareSize.width; 
        fs<<"}";

        fs << "ExtractedFeatures" << "{:";

        fs << "CameraImages" << "{:";

        int size = imgBoardCornersCam.size();
        fs << "NumberOfImgs" << size;

        for(int i=0; i<imgBoardCornersCam.size(); i++)
        {

            std::stringstream name;
            name << "Image" << i+1;
            fs<<name.str()<< "{:";

            fs<<"BoardCorners"<<imgBoardCornersCam[i];
            fs<<"ObjBoardCorners"<<objBoardCornersCam[i];

            fs<<"}";

        }
        fs<<"}";
        fs<<"}";

        fs.release();


    }
} // namespace SLS

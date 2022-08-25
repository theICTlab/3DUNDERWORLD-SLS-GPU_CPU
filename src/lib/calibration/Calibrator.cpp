#include <calibration/Calibrator.hpp>
#include <sstream>
#include <thread>

namespace SLS {

////////////////////////// Callbacks

/*! Mouse callback when drawing rectangles surround the checkerboard. Is called
 * in Calibrator::manualMarkCheckBoard()
 */
void calibBoardCornersMouseCallback(int event, int x, int y, int flags,
                                    void *param)
{
    std::vector<cv::Point2f> *corners = (std::vector<cv::Point2f> *)param;

    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            if (corners->size() < 4)
                corners->push_back(cv::Point(x, y));
            else
                break;
    }
}

/*! Mouse callback to mark a white block in the checkerboard, is called in
 * Calibrator::markWhite()
 */
void imagePointReturn(int event, int x, int y, int flags, void *param)
{
    cv::Scalar *point = (cv::Scalar *)param;
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            point->val[0] = x;
            point->val[1] = y;
            point->val[2] = 1;
            break;
    }
}

/*! Manually pick for corners of \p img image.
 * \param img Input image
 * \return An array of 4 picked corners
 */
std::vector<cv::Point2f> Calibrator::manualMarkCheckBoard(cv::Mat img)
{
    std::vector<cv::Point2f> corners;

    cv::namedWindow("Mark Calibration Board", cv::WINDOW_NORMAL);
    cv::resizeWindow("Mark Calibration Board", WINDOW_WIDTH, WINDOW_HEIGHT);
    // Set a mouse callback
    cv::setMouseCallback("Mark Calibration Board",
                         calibBoardCornersMouseCallback, (void *)&corners);

    bool ok = false;
    while (!ok) {
        corners.clear();
        cv::resizeWindow("Mark Calibration Board", WINDOW_WIDTH, WINDOW_HEIGHT);

        size_t curNumOfCorners = 0;

        cv::Mat img_copy;
        img.copyTo(img_copy);

        cv::Point2f rectSize(20, 20);

        while (corners.size() < 4) {
            // draw selected corners and conection lines
            if (curNumOfCorners < corners.size()) {
                int s = corners.size();

                cv::rectangle(img_copy, corners[s - 1] - rectSize,
                              corners[s - 1] + rectSize, cv::Scalar(0, 0, 255),
                              3);

                if (!(corners.size() == 1)) {
                    cv::line(img_copy, corners[s - 1], corners[s - 2],
                             cv::Scalar(0, 0, 255), 3);
                }

                curNumOfCorners++;
            }
            // cv::imshow("Mark Calibration Board", img_copy);
            std::ostringstream ss;
            ss << "Please click on the 4 extreme corners of the board starting "
                  "from the top left corner "
               << curNumOfCorners << "/4";
            showImgWithText_Block(img_copy, ss.str(), "Mark Calibration Board");
            // Showing in the loop
            cv::waitKey(5);
        }

        // Draw corners and lines
        cv::rectangle(img_copy, corners[3] - rectSize, corners[3] + rectSize,
                      cv::Scalar(0, 0, 255), 3);
        cv::line(img_copy, corners[3], corners[2], cv::Scalar(0, 0, 255), 10);
        cv::line(img_copy, corners[3], corners[0], cv::Scalar(0, 0, 255), 10);

        int key = 0;

        // wait for enter or esc key press
        while (key != 'n' && key != 'r') {
            // cv::imshow("Mark Calibration Board", img_copy );
            showImgWithText_Block(
                img_copy, "Press 'n' to continue or 'r' to select a new area!",
                "Mark Calibration Board");
            key = cv::waitKey(0);
        }

        // if enter set ok as true to stop the loop or repeat the selection
        // process
        if (key == 'n')
            ok = true;
        else
            ok = false;
        img_copy.release();
    }

    cv::destroyWindow("Mark Calibration Board");

    return corners;
}

void drawOutsideOfRectangle(cv::Mat img,
                            std::vector<cv::Point2f> rectanglePoints,
                            float color)
{
    std::vector<cv::Point> corners;
    for (unsigned i = 0; i < rectanglePoints.size(); i++) {
        corners.push_back(rectanglePoints[i]);
    }

    cv::Mat mask(img.size(), img.type());
    cv::Mat background(img.size(), img.type());

    mask = 1;
    cv::fillConvexPoly(mask, corners, cv::Scalar(0));

    background = color;
    background.copyTo(img, mask);
}

float Calibrator::markWhite(const cv::Mat &img)
{
    float white = 0.0;
    cv::namedWindow("Mark White", cv::WINDOW_NORMAL);
    cv::resizeWindow("Mark White", WINDOW_WIDTH, WINDOW_HEIGHT);

    cv::Scalar point;

    // Set a mouse callback
    cv::setMouseCallback("Mark White", imagePointReturn, (void *)&point);

    bool ok = false;

    while (!ok) {
        cv::Mat img_copy = img.clone();
        cv::resizeWindow("Mark White", WINDOW_WIDTH, WINDOW_HEIGHT);

        int pointsCount = 0;
        point.val[2] = 0;
        while (pointsCount < 1) {
            if (point.val[2] == 1) {
                cv::rectangle(img_copy,
                              cv::Point(point.val[0] - 10, point.val[1] - 10),
                              cv::Point(point.val[0] + 10, point.val[1] + 10),
                              cv::Scalar(0, 0, 255), 3);

                white = img.at<uchar>(point.val[1], point.val[0]);

                pointsCount++;
                point.val[2] = 0;
            }
            showImgWithText_Block(img_copy, "Click mouse on a white area",
                                  "Mark White");
            cv::waitKey(5);
        }

        int key = 0;

        while (key != 'n' && key != 'r') {
            showImgWithText_Block(
                img_copy, "Press 'n' to continue or 'r' to select a new point!",
                "Mark White");
            key = cv::waitKey();
        }

        if (key == 'n')
            ok = true;
        else
            ok = false;

        img_copy.release();
    }

    cv::destroyWindow("Mark White");

    return white;
}

bool Calibrator::findCornersInCamImg(const cv::Mat &img,
                                     std::vector<cv::Point2f> &camCorners,
                                     std::vector<cv::Point3f> &objCorners,
                                     cv::Size squareSize)
{
    cv::Mat img_copy = img.clone();  // keep a cpy of it
    cv::Mat img_grey;
    // copy camera img
    int numOfCornersX;
    int numOfCornersY;
    bool found = false;

    // find the corners
    while (!found) {
        img_grey = img.clone();
        // ask user to mark 4 corners of the checkerboard
        std::vector<cv::Point2f> chessBoardCorners =
            manualMarkCheckBoard(img_copy);

        // ask user to mark a white point on checkerboard
        float color = markWhite(img_grey);

        // Modify rectangle
        drawOutsideOfRectangle(img_grey, chessBoardCorners, color);

        // show img to user
        // Create an async task to show image
        cv::namedWindow("Calibration", cv::WINDOW_NORMAL);
        cv::resizeWindow("Calibration", WINDOW_WIDTH, WINDOW_HEIGHT);
        // closeAsynImg = false;
        numOfCornersX = 5;
        numOfCornersY = 5;
        cv::createTrackbar("x", "Calibration", &numOfCornersX, 10);
        cv::createTrackbar("y", "Calibration", &numOfCornersY, 10);
        showImgWithText_Block(img_grey,
                              "Select number of squares on x and y axis on the "
                              "trackbar and press any key",
                              "Calibration");
        cv::waitKey(10);

//        if (numOfCornersX <= 0 || numOfCornersY <= 0) break;

//        if (numOfCornersX <= 3 || numOfCornersY <= 3) {
//            std::cout << "Board size must be >3\n";
//            continue;
//        }

        numOfCornersX=10;
        numOfCornersY=7;

        numOfCornersX--;
        numOfCornersY--;


        found = cv::findChessboardCorners(
            img_grey, cv::Size(numOfCornersX, numOfCornersY), camCorners);

        std::cout << "found = " << camCorners.size() << "\n";

        int key = cv::waitKey(5);

        if (key == 'n') break;

        while (found) {
            cv::destroyWindow("Calibration");
            cv::namedWindow("Calibration", cv::WINDOW_NORMAL);
            cv::resizeWindow("Calibration", WINDOW_WIDTH, WINDOW_HEIGHT);
            cv::drawChessboardCorners(img_copy,
                                      cv::Size(numOfCornersX, numOfCornersY),
                                      camCorners, found);

            showImgWithText_Block(img_copy,
                                  "Press 'n' to continue or 'r' to redo!",
                                  "Calibration");
            cv::imshow("Calibration", img_copy);

            key = cv::waitKey(0);

            if (key == 'r') found = false;
            if (key == 'n') break;
        }
        if (!found) {
            std::cout << "No squares found, do it again ...\n";
            cv::destroyWindow("Calibration");
            img_grey.release();
        }
    }
    // if corners found find subPixel
    if (found) {
        // find sub pix of the corners
        cv::cornerSubPix(
            img_grey, camCorners, cv::Size(20, 20), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

        if (squareSize.height == 0) {
            std::cout << "Give square height in mm: ";
            std::cin >> squareSize.height;

            std::cout << "Give square width in mm: ";
            std::cin >> squareSize.width;
        }

        for (int i = 0; i < numOfCornersY; i++) {
            for (int j = 0; j < numOfCornersX; j++) {
                cv::Point3f p;
                p.x = j * squareSize.width;
                p.y = i * squareSize.height;
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

void Calibrator::Calibrate(ImageFileProcessor *cam, const std::string &calibImgsDir,
                           const std::string &calibFile)
{
    // Value to generate
    cv::Mat camMatrix;
    cv::Mat distortion;
    cv::Mat rotationMatrix;
    cv::Mat translationVector;
    cv::Size squareSize(33, 33);
    cv::Size camImageSize;

    // Load calibration images
    cam->loadImages(calibImgsDir, "", 1, 1, "JPG");
    size_t width, height;
    cam->getResolution(width, height);
    camImageSize.height = height;
    camImageSize.width = width;

    // Extract corners
    std::vector<std::vector<cv::Point2f>> imgBoardCornersCam;
    std::vector<std::vector<cv::Point3f>> objBoardCornersCam;
    imgBoardCornersCam.clear();
    objBoardCornersCam.clear();
    for (size_t i = 0; i < cam->getNumFrames() - 1; i++) {
        std::vector<cv::Point2f> cCam;
        std::vector<cv::Point3f> cObj;
        auto img = cam->getNextFrame().clone();
        findCornersInCamImg(img, cCam, cObj, squareSize);
        if (cCam.size()) {
            imgBoardCornersCam.push_back(cCam);
            objBoardCornersCam.push_back(cObj);
        }
    }
    std::vector<cv::Mat> camRotationVectors;
    std::vector<cv::Mat> camTranslationVectors;

    // Find intrinsic
    cv::calibrateCamera(
        objBoardCornersCam, imgBoardCornersCam, camImageSize, camMatrix,
        distortion, camRotationVectors, camTranslationVectors, 0,
        cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS),
                         30, DBL_EPSILON));

    // Find extrinsic
    auto extImg = cam->getNextFrame().clone();
    std::vector<cv::Point2f> imgPoints;
    std::vector<cv::Point3f> objPoints3D;
    findCornersInCamImg(extImg, imgPoints, objPoints3D, squareSize);
    cv::Mat rVec;

    // find extrinsics rotation & translation
    bool r = cv::solvePnP(objPoints3D, imgPoints, camMatrix, distortion, rVec,
                          translationVector);
    cv::Rodrigues(rVec, rotationMatrix);
    std::cout << rotationMatrix << "\n\n\n" << translationVector << "\n\n\n";

    //          Save calib data
    cv::FileStorage fs(calibFile, cv::FileStorage::WRITE);

    fs << "Camera"
       << "{:";
    fs << "Calibrated" << r << "Matrix" << camMatrix << "Distortion"
       << distortion << "Translation" << translationVector << "Rotation"
       << rotationMatrix;
    fs << "Height" << camImageSize.height << "Width" << camImageSize.width;
    fs << "}";

    fs << "BoardSquare"
       << "{:";
    fs << "Height" << squareSize.height << "Width" << squareSize.width;
    fs << "}";

    fs << "ExtractedFeatures"
       << "{:";

    fs << "CameraImages"
       << "{:";

    int size = imgBoardCornersCam.size();
    fs << "NumberOfImgs" << size;

    for (size_t i = 0; i < imgBoardCornersCam.size(); i++) {
        std::stringstream name;
        name << "Image" << i + 1;
        fs << name.str() << "{:";

        fs << "BoardCorners" << imgBoardCornersCam[i];
        fs << "ObjBoardCorners" << objBoardCornersCam[i];

        fs << "}";
    }
    fs << "}";
    fs << "}";

    fs.release();
}
}  // namespace SLS

#include <opencv2/opencv.hpp>
#include <iostream>
#include <IKMvs/CameraCalibration/Calibration.h>
using namespace std;
using namespace cv;
using namespace KTKR::MVS;
int main()
{
    // =================================================

    Mat image1 = imread("../asset/sanae_01.jpg", cv::IMREAD_COLOR);
    Mat image2 = imread("../asset/sanae_02.jpg", cv::IMREAD_COLOR);

    cv::resize(image1, image1, cv::Size(1200, 800));
    cv::resize(image2, image2, cv::Size(1200, 800));

    int minHessian = 700;
    vector<KeyPoint> keyPoint1, keyPoint2;
    Mat imageDesc1, imageDesc2;

    auto siftDetector = SIFT::create(0);
    siftDetector->detect(image1, keyPoint1);
    siftDetector->detect(image2, keyPoint2);
    siftDetector->compute(image1, keyPoint1, imageDesc1);
    siftDetector->compute(image2, keyPoint2, imageDesc2);

    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matchePoints;
    vector<DMatch> GoodMatchePoints;
    vector<Mat> train_desc(1, imageDesc2);
    matcher.add(train_desc);
    matcher.train();
    matcher.knnMatch(imageDesc1, matchePoints, 2);

    vector<KeyPoint> R_keypoint1, R_keypoint2;
    for (size_t i = 0; i < matchePoints.size(); i++)
    {
        R_keypoint1.push_back(keyPoint1[matchePoints[i][0].queryIdx]);
        R_keypoint2.push_back(keyPoint2[matchePoints[i][0].trainIdx]);
    }
    vector<Point2f> p1, p2;
    for (size_t i = 0; i < matchePoints.size(); i++)
    {
        p1.push_back(R_keypoint1[i].pt);
        p2.push_back(R_keypoint2[i].pt);
    }

    // =================================================

    CalibrationManager::Get()->ReadCalibrationParameters("../asset/calibration/caliberation_result.txt");
    auto cameraMatrix = CalibrationManager::Get()->GetCameraMatrix();
    auto essentialMat = cv::findEssentialMat(p1, p2, cameraMatrix);
    auto fundamentalMat = cv::findFundamentalMat(p1, p2, cv::FM_8POINT);
    auto homographyMat = cv::findHomography(p1, p2, cv::RANSAC, (3.0));

    Mat R, t;
    cv::recoverPose(essentialMat, p1, p2, cameraMatrix, R, t);

    std::cout << R << endl;
    std::cout << t << endl;

    // =================================================
    
    return 0;
}
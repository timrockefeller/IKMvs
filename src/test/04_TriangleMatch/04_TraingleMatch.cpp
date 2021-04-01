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
    vector<uchar> RansacStatus;
    Mat Fundamental = cv::findFundamentalMat(p1, p2, RansacStatus, cv::FM_RANSAC);

    // create new arrays (code cleaning required)
    vector<KeyPoint> RAW_keyPoint1, RAW_keyPoint2;

    size_t _idx = 0;
    vector<Point2f> gp1, gp2;

    for (size_t i = 0; i < matchePoints.size(); i++)
    {
        if (RansacStatus[i] != 0)
        {
            RAW_keyPoint1.push_back(R_keypoint1[i]);
            RAW_keyPoint2.push_back(R_keypoint2[i]);
            gp1.push_back(R_keypoint1[i].pt);
            gp2.push_back(R_keypoint2[i].pt);
            matchePoints[i][0].queryIdx = static_cast<int>(_idx);
            matchePoints[i][0].trainIdx = static_cast<int>(_idx);
            GoodMatchePoints.push_back(matchePoints[i][0]);
            _idx++;
        }
    }

    // =================================================

    CalibrationManager::Get()->ReadCalibrationParameters("../asset/calibration/caliberation_result.txt");
    auto cameraMatrix = CalibrationManager::Get()->GetCameraMatrix();
    auto essentialMat = cv::findEssentialMat(gp1, gp2, cameraMatrix);
    auto fundamentalMat = cv::findFundamentalMat(gp1, gp2, cv::FM_8POINT);
    auto homographyMat = cv::findHomography(gp1, gp2, cv::RANSAC, (3.0));

    Mat R, T;
    cv::recoverPose(essentialMat, gp1, gp2, cameraMatrix, R, T);

    std::cout << R << endl;
    std::cout << T << endl;

    // =================================================
    auto length = T.at<double>(0) * T.at<double>(0) + T.at<double>(1) * T.at<double>(1) + T.at<double>(2) * T.at<double>(2);

    Mat T1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0);

    Mat T2 = (cv::Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1),
              R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2));

    vector<Point2d> camPt1, camPt2;

    for (auto match : GoodMatchePoints)
    {
        camPt1.push_back(pixel2cam(gp1[match.queryIdx], cameraMatrix));
        camPt2.push_back(pixel2cam(gp2[match.trainIdx], cameraMatrix));
    }

    Mat pts_4d;

    cv::triangulatePoints(T1, T2, camPt1, camPt2, pts_4d);

    vector<Point3d> points;

    for (int i = 0; i < pts_4d.cols; i++)
    {
        Mat x = pts_4d.col(i);
        x /= x.at<double>(3);
        points.emplace_back(x.at<double>(0), x.at<double>(1), x.at<double>(2));
        std::cout << points.back() << endl;
    }

    return 0;
}
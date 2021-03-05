#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

using namespace cv::xfeatures2d;
int main()
{

    //     Mat img1 = imread("../asset/banana_2.jpg");    //右图
    //     Mat img2 = imread("../asset/banana_1.jpg");    //左图

    //     cv::resize(img1,img1,cv::Size(800,600));
    //     cv::resize(img2,img2,cv::Size(800,600));

    //     // SURF 特征检测与匹配
    //     int minHessian = 700;
    //     Ptr<SURF> detector = SURF::create(minHessian);
    //     Ptr<DescriptorExtractor> descriptor = SURF::create();
    //     FlannBasedMatcher matcher;
    //     Ptr<DescriptorMatcher> matcher1 = DescriptorMatcher::create("BruteForce");
    // //    BFMatcher matcher1(NORM_L2);

    //     std::vector<KeyPoint> keyPoint1, keyPoint2;
    //     Mat descriptors1, descriptors2;
    //     std::vector<DMatch> matches;

    //     // 检测特征点
    //     detector->detect(img1, keyPoint1);
    //     detector->detect(img2, keyPoint2);
    //     // 提取特征点描述子
    //     descriptor->compute(img1, keyPoint1, descriptors1);
    //     descriptor->compute(img2, keyPoint2, descriptors2);
    //     // 匹配图像中的描述子
    //     matcher1->match(descriptors1, descriptors2, matches);
    //     Mat img_keyPoint1, img_keyPoint2;
    //     drawKeypoints(img1, keyPoint1, img_keyPoint1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    //     imshow("keyPoint1 SURF", img_keyPoint1);
    //     drawKeypoints(img2, keyPoint2, img_keyPoint2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    //     imshow("keyPoint2 SURF", img_keyPoint2);

    //     Mat img_matches;
    //     drawMatches(img1, keyPoint1, img2, keyPoint2, matches, img_matches);
    //     imshow("img_matches", img_matches);

    //     cout << "keyPoint1.size = " << keyPoint1.size() << endl;
    //     cout << "keyPoint2.size = " << keyPoint2.size() << endl;
    //     cout << "descriptors1.size = " << descriptors1.size() << endl;
    //     cout << "descriptors1.size = " << descriptors2.size() << endl;
    //     cout << "matches.size = " << matches.size() << endl;
    // //    for (int i = 0; i < matches.size(); i++)
    // //        cout << matches[i].distance << ' ';
    // //    cout << endl;
    //     waitKey(0);

    Mat image01 = imread("../asset/airpods_01.JPG", 1);
    Mat image02 = imread("../asset/airpods_02.JPG", 1);

    cv::namedWindow("first_match", cv::WINDOW_AUTOSIZE);

    cv::resize(image01, image01, cv::Size(1200, 800));
    cv::resize(image02, image02, cv::Size(1200, 800));


    //灰度图转换
    Mat image1, image2;
    cvtColor(image01, image1, cv::COLOR_RGB2GRAY);
    cvtColor(image02, image2, cv::COLOR_RGB2GRAY);

    //提取特征点
    Ptr<SURF> surfDetector = SURF::create(2000); // 海塞矩阵阈值，在这里调整精度，值越大点越少，越精准
    vector<KeyPoint> keyPoint1, keyPoint2;
    surfDetector->detect(image1, keyPoint1);
    surfDetector->detect(image2, keyPoint2);

    //特征点描述，为下边的特征点匹配做准备
    Ptr<DescriptorExtractor> SurfDescriptor = SURF::create();
    Mat imageDesc1, imageDesc2;
    SurfDescriptor->compute(image1, keyPoint1, imageDesc1);
    SurfDescriptor->compute(image2, keyPoint2, imageDesc2);

    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matchePoints;
    vector<DMatch> GoodMatchePoints;

    vector<Mat> train_desc(1, imageDesc1);
    matcher.add(train_desc);
    matcher.train();

    matcher.knnMatch(imageDesc2, matchePoints, 2);
    cout << "total match points: " << matchePoints.size() << endl;

    // Lowe's algorithm,获取优秀匹配点
    for (int i = 0; i < matchePoints.size(); i++)
    {
        if (matchePoints[i][0].distance < 0.6 * matchePoints[i][1].distance)
        {
            GoodMatchePoints.push_back(matchePoints[i][0]);
        }
    }

    Mat first_match;
    drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
    imshow("first_match", first_match);
    waitKey();
    return 0;
}
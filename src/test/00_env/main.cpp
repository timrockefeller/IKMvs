#include <iostream>
#include <IKMvs/ImageQueue.h>
#include <IKMvs/SIFT/SIFTCore.h>
#include <opencv2/opencv.hpp>
using cv::Mat;
int main(int argc, char **argv)
{
    std::cout << "Well you made it! \n";
    Mat image;
    image = cv::imread("../asset/banana_2.JPG", 1);
    cv::resize(image, image, cv::Size(1200, 800));
    if (!image.data)
    {
        std::cout << "No image data \n";
        cv::waitKey(0);
        return -1;
    }
    cv::namedWindow("GoodJob", cv::WINDOW_AUTOSIZE);
    // imshow("GoodJob", image);
    // cv::waitKey(0);

    // SIFT
    std::cout << "Detecting Key Points\n";
    int kp_num{512};
    std::vector<cv::KeyPoint> kps;
    cv::Ptr<cv::SiftFeatureDetector> siftDetector = cv::SiftFeatureDetector::create();
    siftDetector->detect(image, kps);

    cv::drawKeypoints(image, kps, image);
    cv::imshow("GoodJob", image);
    std::cout << "KeyPoint count: " << kps.size() << "\n";
    auto centroid = KTKR::MVS::SIFTCore::GetCentroid(kps);
    std::cout << "Centroid point (x, y) = (" << centroid.x << ", " << centroid.y << ")\n";
    cv::waitKey(0);

    // SVD
    auto A = KTKR::MVS::SIFTCore::GetMatrixA(kps, centroid);
    // std::cout << cv::format(A, cv::Formatter::FMT_NUMPY) << std::endl;

    Mat w, u, vt, y;
    cv::SVD::compute(A, w, u, vt,cv::SVD::FULL_UV);
    std::cout << "---W---\n";
    std::cout << cv::format(w, cv::Formatter::FMT_NUMPY) << std::endl;
    std::cout << "---VT---\n";
    std::cout << cv::format(vt, cv::Formatter::FMT_NUMPY) << std::endl;

    // PCA
    cv::Point2f
        pm1{centroid},
        pm2{centroid + cv::Point2f(vt.at<float>(0, 0), vt.at<float>(0, 1))},
        pm3{centroid + cv::Point2f(vt.at<float>(1, 0), vt.at<float>(1, 1))};
    cv::circle(image, pm1, 5, cv::Scalar(255, 0, 0), -1);
    cv::circle(image, pm2, 5, cv::Scalar(0, 255, 0), -1);
    cv::circle(image, pm3, 5, cv::Scalar(0, 0, 255), -1);
    cv::imshow("GoodJob", image);
    cv::waitKey(0);

    return 0;
}
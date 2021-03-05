#include <iostream>
#include <IKMvs/ImageQueue.h>
#include <IKMvs/SIFT/SIFTCore.h>
#include <opencv2/opencv.hpp>
using cv::Mat;
int main(int argc, char **argv)
{
    std::cout << "Well you made it! \n";
    Mat image;
    image = cv::imread("../asset/banana_1.jpg", 1);
    cv::resize(image, image, cv::Size(1200, 800));
    if (!image.data)
    {
        std::cout << "No image data \n";
        cv::waitKey(0);
        return -1;
    }
    cv::namedWindow("GoodJob", cv::WindowFlags::WINDOW_AUTOSIZE);
    

    auto kps = KTKR::MVS::SIFTCore::GetKeyPoints(image, true);

    cv::imshow("GoodJob", image);
    cv::waitKey(0);

    return 0;
}
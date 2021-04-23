// ================================================= //
//                                                   //
//                  solvePnPRansac                   //
//                                                   //
// ================================================= //

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <IKMvs/CameraCalibration/Calibration.h>
#include <IKMvs/SfM/SfM.h>
using namespace std;
using namespace cv;
using namespace KTKR::MVS;

int main()
{
    const vector<string> images = {
        "../asset/sanae/1.JPG",
        "../asset/sanae/2.JPG",
        "../asset/sanae/3.JPG",
        "../asset/sanae/4.JPG",
        "../asset/sanae/5.JPG",
        "../asset/sanae/6.JPG"};

    SfM::Get()->Init();
    SfM::Get()->LoadImage(images);

    SfM::Get()->extractFeatures();
    SfM::Get()->createFeatureMatchMatrix();
    return 0;
}
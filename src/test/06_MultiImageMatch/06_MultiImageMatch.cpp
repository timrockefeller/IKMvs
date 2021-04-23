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
// #include <Ikit/STL/ILog.h>
using namespace std;
using namespace cv;
using namespace KTKR;
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
    SfM::Get()->LoadIntrinsics("../asset/calibration/caliberation_result.txt");
    SfM::Get()->LoadImage(images);

    SfM::Get()->extractFeatures();
    SfM::Get()->createFeatureMatchMatrix();
    SfM::Get()->findBaselineTriangulation();

    // TODO ...

#if _DEBUG
    Mat des;
    const int l = 2, r = 3;
    drawMatches(SfM::Get()->mImages[l], SfM::Get()->mImageFeatures[l].keyPoints,
                SfM::Get()->mImages[r], SfM::Get()->mImageFeatures[r].keyPoints,
                SfM::Get()->mFeatureMatchMatrix[l][r], des);
    imshow("Matches", des);
    ILog(LOG_DEBUG, LOG_DEBUG, "Image shown: match of ", l, " and ", r);
    waitKey();
#endif

    return 0;
}
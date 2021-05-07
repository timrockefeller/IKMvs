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
    // const vector<string> images = {
    //     "../asset/sanae/1.JPG",
    //     "../asset/sanae/2.JPG",
    //     "../asset/sanae/3.JPG",
    //     "../asset/sanae/4.JPG",
    //     "../asset/sanae/5.JPG",
    //     "../asset/sanae/6.JPG"};
    // const vector<string> images = {
    //     "../asset/sanae_01.jpg",
    //     "../asset/sanae_02.jpg"};

    //  const vector<string> images = {
    //     "../asset/crazyhorse/P1000965.JPG",
    //     "../asset/crazyhorse/P1000966.JPG",
    //     "../asset/crazyhorse/P1000967.JPG",
    //     "../asset/crazyhorse/P1000968.JPG",
    //     "../asset/crazyhorse/P1000969.JPG",
    //     "../asset/crazyhorse/P1000970.JPG",
    //     "../asset/crazyhorse/P1000971.JPG"};

    const vector<string> images = {
        // "../asset/sanaetable/IMG_7689.JPG",
        // "../asset/sanaetable/IMG_7690.JPG",
        // "../asset/sanaetable/IMG_7691.JPG",
        // "../asset/sanaetable/IMG_7692.JPG",
        "../asset/sanaetable/IMG_7693.JPG",
        "../asset/sanaetable/IMG_7694.JPG",
        // "../asset/sanaetable/IMG_7695.JPG"
    };

    SfM::Get()->Init();
    SfM::Get()->LoadIntrinsics("../asset/calibration/caliberation_result.txt");
    SfM::Get()->LoadImage(images, {1200, 800});

    SfM::Get()->extractFeatures();
    SfM::Get()->createFeatureMatchMatrix();
    SfM::Get()->findBaselineTriangulation();
    SfM::Get()->addMoreViewsToReconstruction();

    SfM::Get()->savePointCloudToPLY("../asset/06_result_sanae_3");
    // TODO ...

#if _DEBUG
    Mat des;
    const int l = 0, r = 1;
    drawMatches(SfM::Get()->mImages[l], SfM::Get()->mImageFeatures[l].keyPoints,
                SfM::Get()->mImages[r], SfM::Get()->mImageFeatures[r].keyPoints,
                SfM::Get()->mFeatureMatchMatrix[l][r], des);
    imshow("Matches", des);
    ILog(LOG_DEBUG, LOG_DEBUG, "Image shown: match of ", l, " and ", r);
    waitKey();
#endif

    return 0;
}
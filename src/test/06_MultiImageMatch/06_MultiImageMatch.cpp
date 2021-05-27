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
    SfM::Get()->Init();
    SfM::Get()->LoadIntrinsics("../asset/calibration/caliberation_result.txt");
    SfM::Get()->LoadImage("../asset/data_qinghuamen/imagedata", {1200, 800});

    SfM::Get()->extractFeatures();
    SfM::Get()->createFeatureMatchMatrix();
    SfM::Get()->findBaselineTriangulation();
    SfM::Get()->addMoreViewsToReconstruction();

    SfM::Get()->savePointCloudToPLY("../asset/06_result_qinghua");
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
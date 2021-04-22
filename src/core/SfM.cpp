#include <IKMvs/SfM/SfM.h>
#include <IKMvs/SfM/SfMUtil.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ostream>
using namespace std;
using namespace cv;
using namespace KTKR::MVS;

void KTKR::MVS::SfM::runSfM()
{
}
void KTKR::MVS::SfM::extractFeatures()
{
}
void KTKR::MVS::KeyPointsToPoints(const Keypoints &kps, Points2f &ps)
{
    ps.clear();
    for (const auto &kp : kps)
    {
        ps.push_back(kp.pt);
    }
}
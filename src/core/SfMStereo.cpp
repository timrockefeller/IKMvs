#include <IKMvs/SfM/SfMStereo.h>
#include <IKMvs/SfM/SfMUtil.h>
#include <IKMvs/Config.h>
using namespace KTKR::MVS;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int SfMStereo::findHomographyInlier(const Features &left,
                                    const Features &right,
                                    const Matching &matches)
{
    Features alignedLeft;
    Features alignedRight;
    GetAlignedPointsFromMatch(left, right, matches, alignedLeft, alignedRight);

    Mat inlierMask;
    Mat homography;
    if (matches.size() >= 4)
        homography = findHomography(alignedLeft.points, alignedRight.points,
                                    cv::RANSAC, RANSAC_THRESHOLD, inlierMask);

    if (matches.size() < 4 || homography.empty())
        return 0;

    return countNonZero(inlierMask);
}
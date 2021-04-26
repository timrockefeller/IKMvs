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

ErrorCode SfMStereo::findCameraMatricesFromMatch(const Intrinsics &intrinsics,
                                                 const Matching &matches,
                                                 const Features &featureL,
                                                 const Features &featureR,
                                                 Matching &prunedMatches,
                                                 cv::Matx34f &Pleft,
                                                 cv::Matx34f &Pright)
{
    if (intrinsics.K.empty())
    {
        cerr << "Intrinsics matrix(K) must be initialized." << endl;
        return ErrorCode::ERR_RUNTIME_ABORT;
    }

    Features alignedL, alignedR;
    GetAlignedPointsFromMatch(featureL, featureR, matches, alignedL, alignedR);

    Mat mask;
    auto essentialMat = cv::findEssentialMat(alignedL.points, alignedR.points, intrinsics.K, RANSAC, 0.999, 1.0, mask);

    Mat R, T;
    cv::recoverPose(essentialMat, alignedL.points, alignedR.points, intrinsics.K, R, T);

    Pleft = cv::Matx34f::eye();
    Pright = Matx34f(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0),
                     R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1),
                     R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2));

    // filter by mask
    prunedMatches.clear();
    for (size_t i = 0; i < mask.rows; i++)
        if (mask.at<uchar>(i))
            prunedMatches.push_back(matches[i]);

    return OK;
}
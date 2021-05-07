#include <IKMvs/SfM/SfMStereo.h>
#include <IKMvs/Config.h>
using namespace KTKR::MVS;
using namespace cv;
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
    if (matches.size() < 6)
    {

        cerr << "Need at least 6 matches to recover pose." << endl;
        return ErrorCode::ERR_DO_NOT_FIT;
    }
    Features alignedL, alignedR;
    GetAlignedPointsFromMatch(featureL, featureR, matches,
                              alignedL, alignedR);

    Mat mask;
    auto essentialMat = cv::findEssentialMat(alignedL.points, alignedR.points, intrinsics.K,
                                             RANSAC, 0.999, 1.0, mask);

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

ErrorCode SfMStereo::triangulateViews(const Intrinsics &intrinsics,
                                      const ImagePair imagePair,
                                      const Matching &matches,
                                      const Features &featureL,
                                      const Features &featureR,
                                      cv::Matx34f &Pleft,
                                      cv::Matx34f &Pright,
                                      PointCloud &pointCloud)
{
    if (intrinsics.K.empty())
    {
        cerr << "Intrinsics matrix(K) must be initialized." << endl;
        return ErrorCode::ERR_RUNTIME_ABORT;
    }

    vector<int> backRefL;
    vector<int> backRefR;
    Features alignedL;
    Features alignedR;
    GetAlignedPointsFromMatch(featureL, featureR, matches,
                              alignedL, alignedR,
                              backRefL, backRefR);

    Mat normPtsL;
    Mat normPtsR;
    undistortPoints(alignedL.points, normPtsL, intrinsics.K, intrinsics.distortion);
    undistortPoints(alignedR.points, normPtsR, intrinsics.K, intrinsics.distortion);

    // =================================================

    Mat pts_4d;
    cv::triangulatePoints(Pleft, Pright, normPtsL, normPtsR, pts_4d);
    Mat pts_3d;
    cv::convertPointsFromHomogeneous(pts_4d.t(), pts_3d);

    Mat rvecLeft;
    Rodrigues(Pleft.get_minor<3, 3>(0, 0), rvecLeft);
    Mat tvecLeft(Pleft.get_minor<3, 1>(0, 3).t());
    Points2f projectedOnLeft(alignedL.points.size());
    projectPoints(pts_3d, rvecLeft, tvecLeft, intrinsics.K, Mat(), projectedOnLeft);

    Mat rvecRight;
    Rodrigues(Pright.get_minor<3, 3>(0, 0), rvecRight);
    Mat tvecRight(Pright.get_minor<3, 1>(0, 3).t());
    Points2f projectedOnRight(alignedR.points.size());
    projectPoints(pts_3d, rvecRight, tvecRight, intrinsics.K, Mat(), projectedOnRight);

    // =================================================

    for (size_t i = 0; i < pts_3d.rows; i++)
    {
        //check if point reprojection error is small enough
        if (norm(projectedOnLeft[i] - alignedL.points[i]) > MIN_REPROJECTION_ERROR ||
            norm(projectedOnRight[i] - alignedR.points[i]) > MIN_REPROJECTION_ERROR)
        {
            continue;
        }

        Point3DInMap p;
        p.p = Point3f(pts_3d.at<float>(i, 0),
                      pts_3d.at<float>(i, 1),
                      pts_3d.at<float>(i, 2));

        //use back reference to point to original features in images
        p.originatingViews[imagePair.left] = backRefL[i];
        p.originatingViews[imagePair.right] = backRefR[i];

        pointCloud.push_back(p);
    }

    return OK;
}
ErrorCode SfMStereo::findCameraPoseFrom2D3DMatch(const Intrinsics &intrinsics,
                                                 const Image2D3DMatch &match,
                                                 cv::Matx34f &cameraPose)
{
    //Recover camera pose using 2D-3D correspondence
    Mat rvec, tvec;
    Mat inliers;
    try
    {
        solvePnPRansac(
            match.points3D,
            match.points2D,
            intrinsics.K,
            intrinsics.distortion,
            rvec,
            tvec,
            false,
            100,
            RANSAC_THRESHOLD,
            0.99,
            inliers);
    }
    catch (Exception e)
    {
        return ERR_DO_NOT_FIT;
    }
    //check inliers ratio and reject if too small
    if (((float)countNonZero(inliers) / (float)match.points2D.size()) < POSE_INLIERS_MINIMAL_RATIO)
    {
        cerr << "Inliers ratio is too small: " << countNonZero(inliers) << " / " << match.points2D.size() << endl;
        return ERR_DO_NOT_FIT;
    }

    Mat rotMat;
    Rodrigues(rvec, rotMat); //convert to a rotation matrix

    rotMat.copyTo(Mat(3, 4, CV_32FC1, cameraPose.val)(ROT));
    tvec.copyTo(Mat(3, 4, CV_32FC1, cameraPose.val)(TRA));

    return OK;
}
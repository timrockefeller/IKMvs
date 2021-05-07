#pragma once

namespace KTKR::MVS
{
    static const size_t SIFT_GAUSSKERN = 2;
    const double NN_MATCH_RATIO = 0.8;
    const double RANSAC_THRESHOLD = 10.0;        // RANSAC inlier threshold
    const float MIN_REPROJECTION_ERROR = 10.0f;  // Maximum 10-pixel allowed re-projection error
    const float MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE = 0.01f;
    const float MERGE_CLOUD_FEATURE_MIN_MATCH_DISTANCE = 20.0f;
    const int MIN_POINT_COUNT_FOR_HOMOGRAPHY = 100;
    const float POSE_INLIERS_MINIMAL_RATIO = 0.5f;
} // namespace KTKR::MVS

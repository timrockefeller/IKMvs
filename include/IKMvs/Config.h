#pragma once

namespace KTKR::MVS
{
    static const size_t SIFT_GAUSSKERN = 2;
    const double NN_MATCH_RATIO = 0.8;
    const double RANSAC_THRESHOLD = 10.0;       // RANSAC inlier threshold
    const float MIN_REPROJECTION_ERROR = 10.0f; // Maximum 10-pixel allowed re-projection error
    const float MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE = 0.01f;
    const float MERGE_CLOUD_FEATURE_MIN_MATCH_DISTANCE = 20.0f;
    const int MIN_POINT_COUNT_FOR_HOMOGRAPHY = 100;
    const float POSE_INLIERS_MINIMAL_RATIO = 0.5f;

    struct Config
    {

        enum ENUM_FeatureDetector // 特征提取算子
        {
            SIFT,
            SURF,
            ORB
        } FEATURE_DETECTOR = SIFT;

        enum ENUM_FeatureMatcher // 特征匹配算子
        {
            FLANN,
            BRUTE
        } FEATURE_MATCHER = FLANN;

        enum ENUM_MatchPairElector // 选择匹配对方案
        {
            BestHomography,
            Sequencial
        } MATCH_PAIR_ELECTOR = BestHomography;

        // RANSAC阈值
        double RANSAC_THRESHOLD = 10.0;

        // 重投影误差最大值（超过则放弃）
        float MIN_REPROJECTION_ERROR = 10.0f;

        // 点云融合阈值（小于该距离则算同一点）
        float MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE = 0.01f;

        // 特征融合阈值（小于该值则算同一点）
        float MERGE_CLOUD_FEATURE_MIN_MATCH_DISTANCE = 20.0f;

        // 计算单应性矩阵时匹配点数最小值
        int MIN_POINT_COUNT_FOR_HOMOGRAPHY = 100;

        // 位姿恢复时匹配特征使用率
        float POSE_INLIERS_MINIMAL_RATIO = 0.5f;

    } globalConfig;

    class MVSRuntime
    {
    public:
        Config config = globalConfig;
        virtual void SetConfig(Config cfg) { config = cfg; };
    };
} // namespace KTKR::MVS

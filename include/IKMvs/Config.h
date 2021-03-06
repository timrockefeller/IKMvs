#pragma once

namespace KTKR::MVS
{

    struct Config
    {

        enum ENUM_FeatureDetector // 特征提取算子
        {
            SIFT,
            SURF,
            ORB,
            KAZE,
            AKAZE,
        };
        int FEATURE_DETECTOR = SIFT;

        enum ENUM_FeatureMatcher // 特征匹配算子
        {
            FLANN,
            BRUTE
        };
        int FEATURE_MATCHER = FLANN;

        enum ENUM_MatchPairElector // 选择匹配对方案
        {
            BestHomography,
            Sequencial
        };
        int MATCH_PAIR_ELECTOR = BestHomography;

        //
        double NN_MATCH_RATIO = 0.8;

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
    };

    class MVSRuntime
    {
    public:
        Config config;
        virtual void SetConfig(Config cfg) { config = cfg; }
    };
} // namespace KTKR::MVS

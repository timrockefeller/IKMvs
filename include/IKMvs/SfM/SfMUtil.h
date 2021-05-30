#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
namespace KTKR::MVS
{
    using Keypoints = std::vector<cv::KeyPoint>;
    using Points2f = std::vector<cv::Point2f>;
    using Points3f = std::vector<cv::Point3f>;
    struct Features
    {
        Keypoints keyPoints;
        Points2f points;
        cv::Mat descriptors;
    };
    
    // matching points
    using Matching = std::vector<cv::DMatch>;
    // view i to view j
    using MatchMatrix = std::vector<std::vector<Matching>>;

    struct Image2D3DMatch
    {
        Points2f points2D;
        Points3f points3D;
    };
    using Images2D3DMatches = std::map<int, Image2D3DMatch>;

    struct ImagePair
    {
        size_t left, right;
        ImagePair() : left{0}, right{0} {};
        ImagePair(size_t l, size_t r) : left{l}, right{r} {};
    };

    struct Point3DInMap
    {
        // 3D point.
        cv::Point3f p;

        // A mapping from **image index** to **2D point index** in that image's list of features.
        std::map<int, int> originatingViews;
    };

    struct Point3DInMapRGB
    {
        Point3DInMap p;
        cv::Scalar rgb;
    };

    using PointCloud = std::vector<Point3DInMap>;
    using PointCloudRGB = std::vector<Point3DInMapRGB>;
    using Pose = cv::Matx34f;

    struct Intrinsics
    {
        cv::Mat K;
        cv::Mat Kinv;
        cv::Mat distortion;
    };

    ///Rotational element in a 3x4 matrix
    const cv::Rect ROT(0, 0, 3, 3);

    ///Translational element in a 3x4 matrix
    const cv::Rect TRA(3, 0, 1, 3);

    enum ErrorCode
    {
        OK = 0,
        ERR,
        ERR_FILE_OPENING,
        ERR_RUNTIME_ABORT,
        ERR_DO_NOT_FIT
    };

    /**
     * Convert Keypoints to Points2f
     * @param kps keypoints
     * @param ps  points
     */
    void KeyPointsToPoints(const Keypoints &kps, Points2f &ps);

    /**
     * Get the features for left and right images after keeping only the matched features and aligning them.
     * Alignment: i-th feature in left is a match to i-th feature in right.
     * @param leftFeatures       Left image features.
     * @param rightFeatures      Right image features.
     * @param matches            Matching over the features.
     * @param alignedLeft        Output: aligned left features.
     * @param alignedRight       Output: aligned right features.
     * @param leftBackReference  Output: back reference from aligned index to original index
     * @param rightBackReference Output: back reference from aligned index to original index
     */
    void GetAlignedPointsFromMatch(const Features &leftFeatures,
                                   const Features &rightFeatures,
                                   const Matching &matches,
                                   Features &alignedLeft,
                                   Features &alignedRight,
                                   std::vector<int> &leftBackReference,
                                   std::vector<int> &rightBackReference);

    /**
     * Get the features for left and right images after keeping only the matched features and aligning them.
     * Alignment: i-th feature in left is a match to i-th feature in right.
     * @param leftFeatures  Left image features.
     * @param rightFeatures Right image features.
     * @param matches       Matching over the features.
     * @param alignedLeft   Output: aligned left features.
     * @param alignedRight  Output: aligned right features.
     */
    void GetAlignedPointsFromMatch(const Features &leftFeatures,
                                   const Features &rightFeatures,
                                   const Matching &matches,
                                   Features &alignedLeft,
                                   Features &alignedRight);

} // namespace KTKR::MVS

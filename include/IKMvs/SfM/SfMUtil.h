#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
namespace KTKR::MVS
{
    using Keypoints = std::vector<cv::KeyPoint>;
    using Points2f = std::vector<cv::Point2f>;
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

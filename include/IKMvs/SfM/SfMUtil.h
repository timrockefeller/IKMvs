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
        ImagePair(size_t l, size_t r) : left{l}, right{r} {};
        size_t left, right;
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

    /**
     * Convert Keypoints to Points2f
     * @param kps keypoints
     * @param ps  points
     */
    void KeyPointsToPoints(const Keypoints &kps, Points2f &ps);

} // namespace KTKR::MVS

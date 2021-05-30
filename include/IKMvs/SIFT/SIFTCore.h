#pragma once
#include <IKit/STL/Singleton.h>
#include <IKMvs/Config.h>
#include <opencv2/opencv.hpp>
#include <vector>
namespace KTKR::MVS
{
    class SIFTCore : KTKR::Singleton<SIFTCore>
    {
    public:
        static const size_t GAUSSKERN = 2;

        // cv::Ptr<cv::SiftFeatureDetector> siftDetector;

        using KeyPoints = std::vector<cv::KeyPoint>;
        static KeyPoints GetKeyPoints(cv::Mat &image, bool draw = false)
        {
            // if (!siftDetector)
            //     siftDetector = cv::SiftFeatureDetector::create();
            auto siftDetector = cv::SiftFeatureDetector::create();
            KeyPoints kps;
            siftDetector->detect(image, kps);
            if (draw)
                cv::drawKeypoints(image, kps, image);
            return kps;
        }

        static cv::Point2f GetCentroid(const KeyPoints kps) noexcept
        {
            cv::Point2f centroid_sum{0, 0};
            if (kps.size() == 0)
                return centroid_sum;
            for (size_t i = 0; i < kps.size(); i++)
                centroid_sum += kps[i].pt;
            auto centroid = cv::Point2f(centroid_sum.x / kps.size(), centroid_sum.y / kps.size());
            return centroid;
        }
        static cv::Mat GetMatrixA(const KeyPoints kps, const cv::Point2f centroid) noexcept
        {
            auto rsl = cv::Mat(static_cast<int>(kps.size()), 2, CV_32F);
            for (size_t i = 0; i < kps.size(); i++)
            {
                rsl.at<float>(static_cast<int>(i), 0) = kps[i].pt.x - centroid.x;
                rsl.at<float>(static_cast<int>(i), 1) = kps[i].pt.y - centroid.y;
            }
            return rsl;
        }
    };

} // namespace KTKR::MVS

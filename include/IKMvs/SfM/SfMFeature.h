#pragma once
#include "SfMUtil.h"
#include <Ikit/STL/Singleton.h>
namespace KTKR::MVS
{
    class SfMFeature : public KTKR::Singleton<SfMFeature>
    {
    private:
        cv::Ptr<cv::Feature2D> mDetector;
        cv::Ptr<cv::DescriptorMatcher> mMatcher;

    public:
        SfMFeature();
        ~SfMFeature();

        Features extractFeatures(const cv::Mat &image);
        Matching matchFeatures(
            const Features &ls,
            const Features &rs);
    };
} // namespace KTKR::MVS

#pragma once
#include "SfMUtil.h"
#include "../Config.h"
#include <Ikit/STL/Singleton.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
namespace KTKR::MVS
{
    class SfMFeature : public MVSRuntime
    {

    public:
        SfMFeature();
        ~SfMFeature();
        
        Features extractFeatures(const cv::Mat &image);
        Matching matchFeatures(
            const Features &ls,
            const Features &rs);
    };
} // namespace KTKR::MVS

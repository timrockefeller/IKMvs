/**
 * 
 *  Single Process for SfM
 * 
 */
#pragma once
#include "SfMUtil.h"
#include "SfMStereo.h"
#include "SfMFeature.h"
#include <Ikit/STL/ILog.h>
#include <Ikit/STL/Singleton.h>
namespace KTKR::MVS
{

    enum ErrorCode
    {
        OK = 0,
        ERR,
        ERR_LOADING_IMAGE
    };

    class SfM : public KTKR::Singleton<SfM>
    {
    private:
        KTKR::DebugLogLevel _debugLevel;

        // datasets
        std::vector<cv::Mat> mImages;
        std::vector<Features> mImageFeatures;
        MatchMatrix mFeatureMatchMatrix;

    public:
        SfM() : _debugLevel{KTKR::DebugLogLevel::LOG_TRACE} {}
        ~SfM() = default;

        void Init();
        ErrorCode LoadImage(std::vector<std::string> paths);
        void runSfM();
        void extractFeatures();
        void createFeatureMatchMatrix();
    };
} // namespace KTKR::MVS
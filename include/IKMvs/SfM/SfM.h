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
#include <map>
namespace KTKR::MVS
{

    enum ErrorCode
    {
        OK = 0,
        ERR,
        ERR_FILE_OPENING,
        ERR_RUNTIME_ABORT
    };

    class SfM : public KTKR::Singleton<SfM>
    {

    public:
        SfM() : _debugLevel{KTKR::DebugLogLevel::LOG_TRACE} {}
        ~SfM() = default;

        void Init();
        ErrorCode LoadIntrinsics(const std::string &intrinsicsFilePath);
        ErrorCode LoadImage(std::vector<std::string> paths);
        void runSfM();
        void extractFeatures();
        void createFeatureMatchMatrix();
        void findBaselineTriangulation();

        std::map<float, ImagePair> sortViewsForBaseline();
        // private:
        KTKR::DebugLogLevel _debugLevel;

        // datasets
        std::vector<cv::Mat> mImages;
        std::vector<Features> mImageFeatures;
        MatchMatrix mFeatureMatchMatrix;
        Intrinsics mIntrinsics;
    };
} // namespace KTKR::MVS
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

        void adjustCurBundle();
        std::map<float, ImagePair> sortViewsForBaseline();
        ErrorCode savePointCloudToPLY(const std::string &prefix);

        // private:
        KTKR::DebugLogLevel _debugLevel;

        // datasets
        std::vector<cv::Mat> mImages;
        std::vector<Features> mImageFeatures;
        MatchMatrix mFeatureMatchMatrix;
        Intrinsics mIntrinsics;
        PointCloud mReconstructionCloud;
        std::vector<cv::Matx34f> mCameraPoses;
        std::set<int> mDoneViews;
        std::set<int> mGoodViews;
    };
} // namespace KTKR::MVS
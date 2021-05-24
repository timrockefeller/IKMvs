/**
 * 
 *  Single Process for SfM
 * 
 */
#pragma once
#include "SfMUtil.h"
#include "SfMStereo.h"
#include "SfMFeature.h"
#include "SfMBundleAdjustment.h"
#include "../Config.h"
#include <Ikit/STL/ILog.h>
#include <Ikit/STL/Singleton.h>
#include <map>
namespace KTKR::MVS
{

    class SfM : public KTKR::Singleton<SfM>, public MVSRuntime
    {

    public:
        SfM(); 
        ~SfM() = default;

        void Init();
        ErrorCode LoadIntrinsics(const std::string &intrinsicsFilePath);
        ErrorCode LoadImage(std::vector<std::string> paths, cv::Size2f = cv::Size2f(0, 0));
        void runSfM();
        void extractFeatures();
        void createFeatureMatchMatrix();
        void findBaselineTriangulation();
        void addMoreViewsToReconstruction();

        void adjustCurBundle();
        Images2D3DMatches find2D3DMatches();
        void mergeNewPointCloud(const PointCloud &cloud);
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

        SfMFeature sfmFeature;
        SfMStereo sfmStereo;
        SfMBundleAdjustment sfmBundleAdjustment;

        void SetConfig(Config cfg) override;
    };
} // namespace KTKR::MVS
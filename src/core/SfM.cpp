#include <IKMvs/SfM/SfM.h>
#include <IKMvs/SfM/SfMUtil.h>
#include <IKMvs/SfM/SfMFeature.h>
#include <IKMvs/SfM/SfMStereo.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Ikit/STL/ILog.h>
#include <ostream>
using namespace std;
using namespace cv;
using namespace KTKR::MVS;
// =================================================
void KTKR::MVS::SfM::Init()
{
#if _DEBUG
    _debugLevel = KTKR::LOG_DEBUG;
#else
    _debugLevel = KTKR::LOG_INFO;
#endif
}
ErrorCode KTKR::MVS::SfM::LoadImage(std::vector<string> paths)
{

    // clear original image
    if (!mImages.empty())
    {
        mImages.clear();
    }
    mImages.resize(paths.size());

    // load
    for (size_t i = 0; i < paths.size(); i++)
    {
        mImages[i] = imread(paths[i], cv::IMREAD_COLOR);
        if (!mImages[i].data)
        {
            ILog(this->_debugLevel, KTKR::LOG_INFO, "ERROR: Loading image from ", paths[i]);
            return ERR_LOADING_IMAGE;
        }
        // need resize?
        cv::resize(mImages[i], mImages[i], Size(1200, 800));
    }

    return OK;
}

void KTKR::MVS::SfM::runSfM()
{
}

void KTKR::MVS::SfM::extractFeatures()
{
    ILog(this->_debugLevel, KTKR::LOG_INFO, "=== Extract Features ===");
    
    mImageFeatures.resize(mImages.size());

    for (size_t i = 0; i < mImages.size(); i++)
    {
        mImageFeatures[i] = SfMFeature::Get()->extractFeatures(mImages[i]);
        ILog(this->_debugLevel, KTKR::LOG_DEBUG, "INFO: Extracted file: ", i);
    }
}

void KTKR::MVS::SfM::createFeatureMatchMatrix()
{
    ILog(this->_debugLevel, KTKR::LOG_INFO, "=== Create Feature Matrix ===");

    const size_t imageNum = mImages.size();
    mFeatureMatchMatrix.resize(imageNum, vector<Matching>(imageNum));

    vector<ImagePair> pairs;
    for (size_t i = 0; i < imageNum; i++)
        for (size_t j = i + 1; i < imageNum; j++)
            pairs.emplace_back(i, j);
}
// =================================================

// =================================================

void KTKR::MVS::KeyPointsToPoints(const Keypoints &kps, Points2f &ps)
{
    ps.clear();
    for (const auto &kp : kps)
    {
        ps.push_back(kp.pt);
    }
}
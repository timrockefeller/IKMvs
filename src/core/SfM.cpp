#include <ostream>
#include <mutex>
#include <thread>

#include <IKMvs/Config.h>
#include <Ikit/STL/ILog.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <IKMvs/SfM/SfM.h>
#include <IKMvs/SfM/SfMUtil.h>
#include <IKMvs/SfM/SfMFeature.h>
#include <IKMvs/SfM/SfMStereo.h>

#include <IKMvs/CameraCalibration/Calibration.h>

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

ErrorCode KTKR::MVS::SfM::LoadIntrinsics(const std::string &intrinsicsFilePath)
{
    if (CalibrationManager::Get()->ReadCalibrationParameters(intrinsicsFilePath))
    {
        mIntrinsics.K = CalibrationManager::Get()->GetCameraMatrix();
        mIntrinsics.Kinv = mIntrinsics.K.inv();
        mIntrinsics.distortion = CalibrationManager::Get()->GetDistortion();
        return OK;
    }
    return ERR_FILE_OPENING;
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
            return ERR_FILE_OPENING;
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
        ILog(this->_debugLevel, KTKR::LOG_DEBUG, "\tExtracted image id: ", i, " with ", mImageFeatures[i].points.size(), " keypoints.");
    }
}

void KTKR::MVS::SfM::createFeatureMatchMatrix()
{
    ILog(this->_debugLevel, KTKR::LOG_INFO, "=== Create Feature Matrix ===");

    const size_t numImages = mImages.size();
    mFeatureMatchMatrix.resize(numImages, vector<Matching>(numImages));

    vector<ImagePair> pairs;
    for (size_t i = 0; i < numImages; i++)
        for (size_t j = i + 1; j < numImages; j++)
            pairs.emplace_back(i, j);

    vector<thread> threads;
    const int numThreads = std::thread::hardware_concurrency() - 1;
    const int numPairsForThread = (numThreads > pairs.size()) ? 1 : (int)ceilf((float)(pairs.size()) / numThreads);

    mutex writeMutex;

    ILog(this->_debugLevel, KTKR::LOG_DEBUG, "Launch ", numThreads, " threads with ", numPairsForThread, " pairs per thread...");

    //invoke each thread with its pairs to process (if less pairs than threads, invoke only #pairs threads with 1 pair each)
    for (int threadId = 0; threadId < MIN(numThreads, static_cast<int>(pairs.size())); threadId++)
    {
        threads.push_back(thread([&, threadId] {
            const int startingPair = numPairsForThread * threadId;

            for (int j = 0; j < numPairsForThread; j++)
            {
                const int pairId = startingPair + j;

                //make sure threads don't overflow the pairs
                if (pairId >= pairs.size())
                    break;

                const ImagePair &pair = pairs[pairId];

                mFeatureMatchMatrix[pair.left][pair.right] = SfMFeature::Get()->matchFeatures(mImageFeatures[pair.left], mImageFeatures[pair.right]);

                writeMutex.lock();
                ILog(this->_debugLevel, KTKR::LOG_DEBUG, "\tThread ", threadId, ": Match (pair ", pairId, ") ", pair.left, ", ", pair.right, ": ", mFeatureMatchMatrix[pair.left][pair.right].size(), " matched features");
                writeMutex.unlock();
            }
        }));
    }
    for (auto &t : threads)
        t.join();
}

void KTKR::MVS::SfM::findBaselineTriangulation()
{
    ILog(this->_debugLevel, KTKR::LOG_INFO, "=== Find Baseline Triangulation ===");

    ILog(this->_debugLevel, KTKR::LOG_DEBUG, "--- Sort views by homography inliers");
    auto pairsHomographyInliers = sortViewsForBaseline();

    Matx34f Pleft = Matx34f::eye();
    Matx34f Pright = Matx34f::eye();
    PointCloud pointCloud;
    ILog(this->_debugLevel, KTKR::LOG_DEBUG, "--- Try views in triangulation");

    for (auto &imagePair : pairsHomographyInliers)
    {
        ILog(this->_debugLevel, KTKR::LOG_DEBUG, "Trying ", imagePair.second, " ratio: ", imagePair.first);

        auto i = imagePair.second.left;
        auto j = imagePair.second.right;

        // =================================================

        ILog(this->_debugLevel, KTKR::LOG_TRACE, "--- Find camera matrices");

        Matching prunedMatching;

        auto rsl = SfMStereo::Get()->findCameraMatricesFromMatch(
            mIntrinsics,
            mFeatureMatchMatrix[i][j],
            mImageFeatures[i],
            mImageFeatures[j],
            prunedMatching,
            Pleft,
            Pright);
        if (rsl != OK)
        {
            ILog(this->_debugLevel, KTKR::LOG_WARN, "Error obtaining stereo view in ", imagePair.second, ", skip.");
            continue;
        }

        auto poseInliersRatio = static_cast<float>(prunedMatching.size()) / static_cast<float>(mFeatureMatchMatrix[i][j].size());
        ILog(this->_debugLevel, KTKR::LOG_TRACE, "pose inliers ratio ", poseInliersRatio);
        if (poseInliersRatio < POSE_INLIERS_MINIMAL_RATIO)
        {
            ILog(this->_debugLevel, KTKR::LOG_TRACE, "insufficient pose inliers. skip.");
            continue;
        }
        mFeatureMatchMatrix[i][j] = prunedMatching;

        // =================================================

        ILog(this->_debugLevel, KTKR::LOG_DEBUG, "--- Triangulate from stereo views: ", imagePair.second);

        // TODO
    }
}

map<float, ImagePair> KTKR::MVS::SfM::sortViewsForBaseline()
{
    ILog(this->_debugLevel, KTKR::LOG_INFO, "--- Find Views Homography Inliers");
    map<float, ImagePair> matchesSizes;

    const size_t numImages = mImages.size();
    for (size_t i = 0; i < numImages - 1; i++)
    {
        for (size_t j = i + 1; j < numImages; j++)
        {
            if (mFeatureMatchMatrix[i][j].size() < MIN_POINT_COUNT_FOR_HOMOGRAPHY)
            {
                //Not enough points in matching
                matchesSizes.emplace(std::piecewise_construct,
                                     std::forward_as_tuple(1.0f),
                                     std::forward_as_tuple(i, j));
                continue;
            }

            const auto numInliers = SfMStereo::Get()->findHomographyInlier(mImageFeatures[i], mImageFeatures[j], mFeatureMatchMatrix[i][j]);
            const auto inliersRatio = static_cast<float>(numInliers) / static_cast<float>(mFeatureMatchMatrix[i][j].size());
            matchesSizes[inliersRatio] = {i, j};
            ILog(this->_debugLevel, KTKR::LOG_DEBUG, "Homography inliers ratio: ", i, ", ", j, " : ", inliersRatio);
        }
    }
    return matchesSizes;
}

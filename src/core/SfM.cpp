#include <IKMvs/SfM/SfM.h>
#include <IKMvs/SfM/SfMUtil.h>
#include <IKMvs/SfM/SfMFeature.h>
#include <IKMvs/SfM/SfMStereo.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Ikit/STL/ILog.h>
#include <ostream>
#include <mutex>
#include <thread>
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
        ILog(this->_debugLevel, KTKR::LOG_DEBUG, "\tExtracted file: ", i);
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
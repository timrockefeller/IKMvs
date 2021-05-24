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
#include <IKMvs/SfM/SfMBundleAdjustment.h>

#include <IKMvs/CameraCalibration/Calibration.h>

using namespace std;
using namespace cv;
using namespace KTKR::MVS;

// =================================================

KTKR::MVS::SfM::SfM() : _debugLevel{KTKR::DebugLogLevel::LOG_TRACE}
{
    sfmFeature = SfMFeature();
    sfmStereo = SfMStereo();
    sfmBundleAdjustment = SfMBundleAdjustment();
}

void KTKR::MVS::SfM::Init()
{
#if _DEBUG
    _debugLevel = KTKR::LOG_DEBUG;
#else
    _debugLevel = KTKR::LOG_INFO;
#endif
}

void KTKR::MVS::SfM::SetConfig(Config cfg)
{
    this->config = cfg;
    sfmFeature.SetConfig(cfg);
    sfmStereo.SetConfig(cfg);
    sfmBundleAdjustment.SetConfig(cfg);
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

ErrorCode KTKR::MVS::SfM::LoadImage(std::vector<string> paths, cv::Size2f size)
{

    // clear original image
    if (!mImages.empty())
    {
        mImages.clear();
    }
    mImages.resize(paths.size());
    mCameraPoses.resize(paths.size());

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
        if (size.height != 0 && size.width != 0)
            cv::resize(mImages[i], mImages[i], size);
    }

    return OK;
}

void KTKR::MVS::SfM::runSfM()
{
    extractFeatures();
    createFeatureMatchMatrix();
    findBaselineTriangulation();
    addMoreViewsToReconstruction();
}

void KTKR::MVS::SfM::extractFeatures()
{
    ILog(this->_debugLevel, KTKR::LOG_INFO, "=== Extract Features ===");

    mImageFeatures.resize(mImages.size());

    for (size_t i = 0; i < mImages.size(); i++)
    {
        mImageFeatures[i] = sfmFeature.extractFeatures(mImages[i]);
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

                mFeatureMatchMatrix[pair.left][pair.right] = sfmFeature.matchFeatures(mImageFeatures[pair.left], mImageFeatures[pair.right]);

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
        ILog(this->_debugLevel, KTKR::LOG_DEBUG, "Trying ", imagePair.second.left, ", ", imagePair.second.right, " ratio: ", imagePair.first);

        auto i = imagePair.second.left;
        auto j = imagePair.second.right;

        // =================================================

        ILog(this->_debugLevel, KTKR::LOG_TRACE, "--- Find camera matrices");

        Matching prunedMatching;

        auto rsl = sfmStereo.findCameraMatricesFromMatch(
            mIntrinsics,
            mFeatureMatchMatrix[i][j],
            mImageFeatures[i],
            mImageFeatures[j],
            prunedMatching,
            Pleft,
            Pright);
        if (rsl != OK)
        {
            ILog(this->_debugLevel, KTKR::LOG_WARN, "Error obtaining stereo view in ", imagePair.second.left, ", ", imagePair.second.right, ". skip.");
            continue;
        }

        auto poseInliersRatio = static_cast<float>(prunedMatching.size()) / static_cast<float>(mFeatureMatchMatrix[i][j].size());
        ILog(this->_debugLevel, KTKR::LOG_TRACE, "pose inliers ratio ", poseInliersRatio);
        if (poseInliersRatio < config.POSE_INLIERS_MINIMAL_RATIO)
        {
            ILog(this->_debugLevel, KTKR::LOG_TRACE, "insufficient pose inliers. skip.");
            continue;
        }
        mFeatureMatchMatrix[i][j] = prunedMatching;

        // =================================================

        ILog(this->_debugLevel, KTKR::LOG_DEBUG, "--- Triangulate from stereo views: ", imagePair.second.left, ", ", imagePair.second.right);

        rsl = sfmStereo.triangulateViews(
            mIntrinsics,
            imagePair.second,
            mFeatureMatchMatrix[i][j],
            mImageFeatures[i],
            mImageFeatures[j],
            Pleft,
            Pright,
            pointCloud);
        if (rsl != OK)
        {
            ILog(this->_debugLevel, KTKR::LOG_WARN, "Error triangulating: ", imagePair.second.left, ", ", imagePair.second.right, ". skip.");
            continue;
        }
        mReconstructionCloud = pointCloud;
        mCameraPoses[i] = Pleft;
        mCameraPoses[j] = Pright;
        mDoneViews.insert(i);
        mDoneViews.insert(j);
        mGoodViews.insert(i);
        mGoodViews.insert(j);

        adjustCurBundle();

        break; // only first two image need to calculate in this turn
        // TODO in sequence
    }
}

void SfM::adjustCurBundle()
{
    sfmBundleAdjustment.adjustBundle(
        mReconstructionCloud,
        mCameraPoses,
        mIntrinsics,
        mImageFeatures);
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
            if (mFeatureMatchMatrix[i][j].size() < config.MIN_POINT_COUNT_FOR_HOMOGRAPHY)
            {
                //Not enough points in matching
                matchesSizes.emplace(std::piecewise_construct,
                                     std::forward_as_tuple(1.0f),
                                     std::forward_as_tuple(i, j));
                continue;
            }

            const auto numInliers = sfmStereo.findHomographyInlier(mImageFeatures[i], mImageFeatures[j], mFeatureMatchMatrix[i][j]);
            const auto inliersRatio = static_cast<float>(numInliers) / static_cast<float>(mFeatureMatchMatrix[i][j].size());
            matchesSizes[inliersRatio] = {i, j};
            ILog(this->_debugLevel, KTKR::LOG_DEBUG, "Homography inliers ratio: ", i, ", ", j, " : ", inliersRatio);
        }
    }
    return matchesSizes;
}

void KTKR::MVS::SfM::addMoreViewsToReconstruction()
{
    ILog(this->_debugLevel, KTKR::LOG_INFO, "--- Add views");

    while (mDoneViews.size() != mImages.size())
    {
        ErrorCode rsl = OK;
        size_t bestView;
        size_t bestNumMatches = 0;

        // =================================================
        // step A. Find the best view to add, according to the largest number of 2D-3D corresponding points
        Images2D3DMatches matches2D3D = find2D3DMatches();

        for (const auto &match2D3D : matches2D3D)
        {
            if (match2D3D.second.points2D.size() == 0 || match2D3D.second.points3D.size() == 0)
            {
                mDoneViews.insert(match2D3D.first);
                rsl = ErrorCode::ERR_RUNTIME_ABORT;
            }
            const size_t numMatches = match2D3D.second.points2D.size();
            if (numMatches > bestNumMatches)
            {
                bestView = match2D3D.first;
                bestNumMatches = numMatches;
            }
        }
        // step B. Sequenly by adding `cur` flag in order to fit video or continously image queue.
        // TODO

        // =================================================
        if (rsl != OK)
            continue;

        ILog(this->_debugLevel, KTKR::LOG_DEBUG, "Best view ", bestView, " has ", bestNumMatches, " matches");
        ILog(this->_debugLevel, KTKR::LOG_DEBUG, "Adding ", bestView, " to existing ", Mat(vector<int>(mGoodViews.begin(), mGoodViews.end())).t());

        mDoneViews.insert(bestView);

        //recover the new view camera pose
        Matx34f newCameraPose;
        rsl = sfmStereo.findCameraPoseFrom2D3DMatch(
            mIntrinsics,
            matches2D3D[bestView],
            newCameraPose);
        if (rsl != OK)
        {
            ILog(this->_debugLevel, KTKR::LOG_WARN, "Cannot recover camera pose for view ", bestView, ". skip.");
            continue;
        }

        mCameraPoses[bestView] = newCameraPose;

        // match with other good view
        bool anyViewSuccess = false;
        for (const auto goodView : mGoodViews)
        {
            size_t viewIdxL = (goodView < bestView) ? goodView : bestView;
            size_t viewIdxR = (goodView < bestView) ? bestView : goodView;

            Matx34f Pleft = Matx34f::eye();
            Matx34f Pright = Matx34f::eye();
            Matching prunedMatching;
            auto rsl = sfmStereo.findCameraMatricesFromMatch(
                mIntrinsics,
                mFeatureMatchMatrix[viewIdxL][viewIdxR],
                mImageFeatures[viewIdxL],
                mImageFeatures[viewIdxR],
                prunedMatching,
                Pleft,
                Pright);

            PointCloud pointCloud;
            mFeatureMatchMatrix[viewIdxL][viewIdxR] = prunedMatching;
            if (rsl == OK)
            {
                //triangulate the matching points
                rsl = sfmStereo.triangulateViews(
                    mIntrinsics,
                    {viewIdxL, viewIdxR},
                    mFeatureMatchMatrix[viewIdxL][viewIdxR],
                    mImageFeatures[viewIdxL],
                    mImageFeatures[viewIdxR],
                    mCameraPoses[viewIdxL],
                    mCameraPoses[viewIdxR],
                    pointCloud);
            }

            if (rsl == OK)
            {
                ILog(this->_debugLevel, KTKR::LOG_DEBUG, "Merge triangulation between ", viewIdxL, " and ", viewIdxR,
                     " (# matching pts = ", (mFeatureMatchMatrix[viewIdxL][viewIdxR].size()), ") ");
                mergeNewPointCloud(pointCloud);
                anyViewSuccess = true;
            }
            else
            {
                ILog(this->_debugLevel, KTKR::LOG_WARN, "Failed to triangulate ", viewIdxL, " and ", viewIdxR);
            }
        }
        //Adjust bundle if any additional view was added
        if (anyViewSuccess)
            adjustCurBundle();
        mGoodViews.insert(bestView);
    }
}

void SfM::mergeNewPointCloud(const PointCloud &cloud)
{
    const size_t numImages = mImages.size();
    MatchMatrix mergeMatchMatrix;
    mergeMatchMatrix.resize(numImages, vector<Matching>(numImages));

    size_t newPoints = 0;
    size_t mergedPoints = 0;

    for (const Point3DInMap &p : cloud)
    {
        const Point3f newPoint = p.p; //new 3D point

        bool foundAnyMatchingExistingViews = false;
        bool foundMatching3DPoint = false;
        for (Point3DInMap &existingPoint : mReconstructionCloud)
        {
            if (norm(existingPoint.p - newPoint) < config.MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE)
            {
                //This point is very close to an existing 3D cloud point
                foundMatching3DPoint = true;

                //Look for common 2D features to confirm match
                for (const auto &newKv : p.originatingViews)
                {
                    //kv.first = new point's originating view
                    //kv.second = new point's view 2D feature index

                    for (const auto &existingKv : existingPoint.originatingViews)
                    {
                        //existingKv.first = existing point's originating view
                        //existingKv.second = existing point's view 2D feature index

                        bool foundMatchingFeature = false;

                        const bool newIsLeft = newKv.first < existingKv.first;
                        const int leftViewIdx = (newIsLeft) ? newKv.first : existingKv.first;
                        const int leftViewFeatureIdx = (newIsLeft) ? newKv.second : existingKv.second;
                        const int rightViewIdx = (newIsLeft) ? existingKv.first : newKv.first;
                        const int rightViewFeatureIdx = (newIsLeft) ? existingKv.second : newKv.second;

                        const Matching &matching = mFeatureMatchMatrix[leftViewIdx][rightViewIdx];
                        for (const DMatch &match : matching)
                        {
                            if (match.queryIdx == leftViewFeatureIdx &&
                                match.trainIdx == rightViewFeatureIdx &&
                                match.distance < config.MERGE_CLOUD_FEATURE_MIN_MATCH_DISTANCE)
                            {

                                mergeMatchMatrix[leftViewIdx][rightViewIdx].push_back(match);

                                //Found a 2D feature match for the two 3D points - merge
                                foundMatchingFeature = true;
                                break;
                            }
                        }

                        if (foundMatchingFeature)
                        {
                            //Add the new originating view, and feature index
                            existingPoint.originatingViews[newKv.first] = newKv.second;

                            foundAnyMatchingExistingViews = true;
                        }
                    }
                }
            }
            if (foundAnyMatchingExistingViews)
            {
                mergedPoints++;
                break; //Stop looking for more matching cloud points
            }
        }

        if (!foundAnyMatchingExistingViews && !foundMatching3DPoint)
        {
            //This point did not match any existing cloud points - add it as new.
            mReconstructionCloud.push_back(p);
            newPoints++;
        }
    }
    ILog(this->_debugLevel, KTKR::LOG_DEBUG, " adding: ", cloud.size(), " (new: ", newPoints, ", merged: ", mergedPoints, ")");
}

ErrorCode KTKR::MVS::SfM::savePointCloudToPLY(const std::string &prefix)
{
    ILog(this->_debugLevel, KTKR::LOG_INFO, "Saving result reconstruction with prefix: ", prefix);

    ofstream ofs(prefix + "_points.ply");
    if (ofs.fail())
        return ERR_FILE_OPENING;

    // header
    ofs << "ply                 " << endl
        << "format ascii 1.0    " << endl
        << "element vertex " << mReconstructionCloud.size() << endl
        << "property float x    " << endl
        << "property float y    " << endl
        << "property float z    " << endl
        << "property uchar red  " << endl
        << "property uchar green" << endl
        << "property uchar blue " << endl
        << "end_header          " << endl;

    for (const auto &p : mReconstructionCloud)
    {
        // get color from first originating view
        auto originatingView = p.originatingViews.begin();
        const int viewIdx = originatingView->first;
        Point2f p2d = mImageFeatures[viewIdx].points[originatingView->second];
        Vec3b pointColor = mImages[viewIdx].at<Vec3b>(p2d);
        // write vertex
        ofs << p.p.x << " " << p.p.y << " " << p.p.z << " " << (int)pointColor(2) << " " << (int)pointColor(1) << " " << (int)pointColor(0) << " " << endl;
    }
    ofs.close();
    ILog(this->_debugLevel, KTKR::LOG_INFO, "Saved.");
    return OK;
}

Images2D3DMatches KTKR::MVS::SfM::find2D3DMatches()
{
    Images2D3DMatches matches;

    //scan all not-done views
    for (size_t viewIdx = 0; viewIdx < mImages.size(); viewIdx++)
    {
        if (mDoneViews.find(viewIdx) != mDoneViews.end())
        {
            continue; //skip done views
        }

        Image2D3DMatch match2D3D;

        //scan all cloud 3D points
        for (const Point3DInMap &cloudPoint : mReconstructionCloud)
        {
            bool found2DPoint = false;

            //scan all originating views for that 3D point
            for (const auto &origViewAndPoint : cloudPoint.originatingViews)
            {
                //check for 2D-2D matching via the match matrix
                const int originatingViewIndex = origViewAndPoint.first;
                const int originatingViewFeatureIndex = origViewAndPoint.second;

                //match matrix is upper-triangular (not symmetric) so the left index must be the smaller one
                const int leftViewIdx = (originatingViewIndex < viewIdx) ? originatingViewIndex : viewIdx;
                const int rightViewIdx = (originatingViewIndex < viewIdx) ? viewIdx : originatingViewIndex;

                //scan all 2D-2D matches between originating view and new view
                for (const DMatch &m : mFeatureMatchMatrix[leftViewIdx][rightViewIdx])
                {
                    int matched2DPointInNewView = -1;
                    if (originatingViewIndex < viewIdx)
                    { //originating view is 'left'
                        if (m.queryIdx == originatingViewFeatureIndex)
                        {
                            matched2DPointInNewView = m.trainIdx;
                        }
                    }
                    else
                    { //originating view is 'right'
                        if (m.trainIdx == originatingViewFeatureIndex)
                        {
                            matched2DPointInNewView = m.queryIdx;
                        }
                    }
                    if (matched2DPointInNewView >= 0)
                    {
                        //This point is matched in the new view
                        const Features &newViewFeatures = mImageFeatures[viewIdx];
                        match2D3D.points2D.push_back(newViewFeatures.points[matched2DPointInNewView]);
                        match2D3D.points3D.push_back(cloudPoint.p);
                        found2DPoint = true;
                        break;
                    }
                }

                if (found2DPoint)
                {
                    break;
                }
            }
        }

        matches[viewIdx] = match2D3D;
    }

    return matches;
}
#include <IKMvs/SfM/SfMFeature.h>
#include <IKMvs/Config.h>
using namespace KTKR::MVS;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
SfMFeature::SfMFeature()
{

    // TODO: make it easy to modify
    mDetector = SIFT::create();
    mMatcher = DescriptorMatcher::create("BruteForce-Hamming");
}

SfMFeature::~SfMFeature()
{
}

Features SfMFeature::extractFeatures(const Mat &image)
{
    Features features;
    mDetector->detectAndCompute(image, noArray(), features.keyPoints, features.descriptors);
    KeyPointsToPoints(features.keyPoints, features.points);
    return features;
}

Matching SfMFeature::matchFeatures(const Features &ls, const Features &rs)
{
    vector<Matching> matchedPoints;
    Matching goodMatchedPoints;

    // TODO: make it easy to modify
    auto matcher = DescriptorMatcher::create("FlannBased");
    matcher->knnMatch(ls.descriptors, rs.descriptors, matchedPoints, 2);

    //ransac filter
    vector<uchar> RansacStatus(ls.points.size());
    vector<Point2f> lsP{matchedPoints.size()}, rsP{matchedPoints.size()};
    for (int i = 0; i < matchedPoints.size(); i++)
    {
        lsP[i] = ls.points[matchedPoints[i][0].queryIdx];
        rsP[i] = rs.points[matchedPoints[i][0].trainIdx];
    }
    cv::findFundamentalMat(lsP, rsP, RansacStatus, FM_RANSAC, config.RANSAC_THRESHOLD);

    // prune matching between the ratio test
    for (size_t i = 0; i < matchedPoints.size(); i++)
    {
        if (RansacStatus[i] != 0 && matchedPoints[i][0].distance < config.NN_MATCH_RATIO * matchedPoints[i][1].distance)
        {
            goodMatchedPoints.push_back(matchedPoints[i][0]);
        }
    }
    return goodMatchedPoints;
}
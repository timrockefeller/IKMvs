#include <IKMvs/SfM/SfMFeature.h>
#include <IKMvs/Config.h>
using namespace KTKR::MVS;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
SfMFeature::SfMFeature()
{

    // TODO: make it easy to modify
    // mDetector = SIFT::create();
    // mMatcher = DescriptorMatcher::create("BruteForce-Hamming");
}

SfMFeature::~SfMFeature()
{
}

Features SfMFeature::extractFeatures(const Mat &image)
{
    Features features;
    cv::Ptr<Feature2D> mDetector;
    switch (config.FEATURE_DETECTOR)
    {
    case Config::ENUM_FeatureDetector::SIFT:
        mDetector = SIFT::create();
        break;
    case Config::ENUM_FeatureDetector::SURF:
        mDetector = SURF::create();
        break;
    case Config::ENUM_FeatureDetector::KAZE:
        mDetector = KAZE::create();
        break;
    case Config::ENUM_FeatureDetector::AKAZE:
        mDetector = AKAZE::create();
        break;
    default:
        break;
    }
    mDetector->detectAndCompute(image, noArray(), features.keyPoints, features.descriptors);
    KeyPointsToPoints(features.keyPoints, features.points);
    return features;
}

Matching SfMFeature::matchFeatures(const Features &ls, const Features &rs)
{
    vector<Matching> matchedPoints;
    Matching goodMatchedPoints;

    Ptr<DescriptorMatcher> matcher;
    if (config.FEATURE_DETECTOR == Config::ENUM_FeatureDetector::ORB)
    {
        matcher = DescriptorMatcher::create("BruteForce-Hamming");
    }
    else
    {
        matcher = DescriptorMatcher::create("FlannBased");
    }
    matcher->knnMatch(ls.descriptors, rs.descriptors, matchedPoints, 2);
    cout<<"u";
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
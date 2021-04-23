#include <IKMvs/SfM/SfMUtil.h>

using namespace std;
using namespace cv;
using namespace KTKR::MVS;

void KTKR::MVS::KeyPointsToPoints(const Keypoints &kps, Points2f &ps)
{
    ps.clear();
    for (const auto &kp : kps)
    {
        ps.push_back(kp.pt);
    }
}
void KTKR::MVS::GetAlignedPointsFromMatch(const Features &leftFeatures,
                                          const Features &rightFeatures,
                                          const Matching &matches,
                                          Features &alignedLeft,
                                          Features &alignedRight)
{
    vector<int> leftBackReference, rightBackReference;
    GetAlignedPointsFromMatch(
        leftFeatures,
        rightFeatures,
        matches,
        alignedLeft,
        alignedRight,
        leftBackReference,
        rightBackReference);
}
void KTKR::MVS::GetAlignedPointsFromMatch(const Features &leftFeatures,
                                          const Features &rightFeatures,
                                          const Matching &matches,
                                          Features &alignedLeft,
                                          Features &alignedRight,
                                          std::vector<int> &leftBackReference,
                                          std::vector<int> &rightBackReference)
{
    alignedLeft.keyPoints.clear();
    alignedRight.keyPoints.clear();
    alignedLeft.descriptors = cv::Mat();
    alignedRight.descriptors = cv::Mat();

    for (unsigned int i = 0; i < matches.size(); i++)
    {
        alignedLeft.keyPoints.push_back(leftFeatures.keyPoints[matches[i].queryIdx]);
        alignedLeft.descriptors.push_back(leftFeatures.descriptors.row(matches[i].queryIdx));
        alignedRight.keyPoints.push_back(rightFeatures.keyPoints[matches[i].trainIdx]);
        alignedRight.descriptors.push_back(rightFeatures.descriptors.row(matches[i].trainIdx));
        leftBackReference.push_back(matches[i].queryIdx);
        rightBackReference.push_back(matches[i].trainIdx);
    }

    KeyPointsToPoints(alignedLeft.keyPoints, alignedLeft.points);
    KeyPointsToPoints(alignedRight.keyPoints, alignedRight.points);
}
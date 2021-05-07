#pragma once
#include <Ikit/STL/Singleton.h>
#include "SfMUtil.h"

namespace KTKR::MVS
{

    class SfMStereo : public KTKR::Singleton<SfMStereo>
    {
    public:
        int findHomographyInlier(const Features &left,
                                 const Features &right,
                                 const Matching &matches);

        ErrorCode findCameraMatricesFromMatch(const Intrinsics &intrinsics,
                                              const Matching &matches,
                                              const Features &featureL,
                                              const Features &featureR,
                                              Matching &prunedMatches,
                                              cv::Matx34f &Pleft,
                                              cv::Matx34f &Pright);

        ErrorCode triangulateViews(const Intrinsics &intrinsics,
                                   const ImagePair imagePair,
                                   const Matching &matches,
                                   const Features &featureL,
                                   const Features &featureR,
                                   cv::Matx34f &Pleft,
                                   cv::Matx34f &Pright,
                                   PointCloud &PointCloud);

        ErrorCode findCameraPoseFrom2D3DMatch(const Intrinsics &intrinsics,
                                              const Image2D3DMatch &match,
                                              cv::Matx34f &cameraPose);
    };
} // namespace KTKR::MVS
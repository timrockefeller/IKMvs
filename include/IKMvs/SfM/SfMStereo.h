#pragma once
#include <Ikit/STL/Singleton.h>
#include "SfM.h"

#define COOL !

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
    };
} // namespace KTKR::MVS
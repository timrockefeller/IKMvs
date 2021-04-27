#pragma once
#include "SfMUtil.h"
#include <Ikit/STL/Singleton.h>

namespace KTKR::MVS
{
    class SfMBundleAdjustment : public KTKR::Singleton<SfMBundleAdjustment>
    {
    public:
        void adjustBundle(
            PointCloud &pointCloud,
            std::vector<Pose> &cameraPoses,
            Intrinsics &intrinsics,
            const std::vector<Features> &image2dFeatures);
    };

} // namespace KTKR::MVS

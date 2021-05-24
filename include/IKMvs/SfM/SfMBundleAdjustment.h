#pragma once
#include "SfMUtil.h"
#include "../Config.h"
#include <Ikit/STL/Singleton.h>

namespace KTKR::MVS
{
    class SfMBundleAdjustment : public MVSRuntime
    {
    public:
        void adjustBundle(
            PointCloud &pointCloud,
            std::vector<Pose> &cameraPoses,
            Intrinsics &intrinsics,
            const std::vector<Features> &image2dFeatures);
    };

} // namespace KTKR::MVS

#pragma once
#include <Ikit/STL/Singleton.h>
#include "SfM.h"

#define COOL !

namespace KTKR::MVS
{

    class SfMStereo : public KTKR::Singleton<SfMStereo>
    {
    public:
        void findBaselineTriangulation()
        {
        }

        int findHomographyInlier(const Features &left,
                                 const Features &right,
                                 const Matching &matches);
    };
} // namespace KTKR::MVS
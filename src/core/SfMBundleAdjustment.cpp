#include <IKMvs/SfM/SfMBundleAdjustment.h>
#include <IKMvs/Config.h>
#include <Ikit/STL/ILog.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
using namespace std;
using namespace cv;
using namespace KTKR;
using namespace KTKR::MVS;

namespace KTKR::Private
{
    void initLogging()
    {
        google::InitGoogleLogging("SFM");
    }
    std::once_flag initLoggingFlag;
}

void SfMBundleAdjustment::adjustBundle(
    PointCloud &pointCloud,
    std::vector<Pose> &cameraPoses,
    Intrinsics &intrinsics,
    const std::vector<Features> &image2dFeatures)
{
    std::call_once(Private::initLoggingFlag, Private::initLogging);

    ceres::Problem problem;


}
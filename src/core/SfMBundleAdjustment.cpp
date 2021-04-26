#include <IKMvs/SfM/SfMBundleAdjustment.h>
#include <IKMvs/Config.h>
#include <Ikit/STL/ILog.h>

using namespace std;
using namespace cv;
using namespace KTKR;
using namespace KTKR::MVS;

void adjustBundle(
    PointCloud &pointCloud,
    std::vector<Pose> &cameraPoses,
    Intrinsics &intrinsics,
    const std::vector<Features> &image2dFeatures)
{
}
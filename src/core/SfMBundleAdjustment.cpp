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

struct ReprojectCost
{
    cv::Point2d observation;

    ReprojectCost(cv::Point2d &observation)
        : observation(observation)
    {
    }

    template <typename T>
    bool operator()(const T *const intrinsic, // (1x4)
                    const T *const extrinsic, // (1x6)
                    const T *const pos3d,     // (1x3)
                    T *residuals) const
    {
        const T *r = extrinsic;
        const T *t = &extrinsic[3];

        T pos_proj[3];
        ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

        // Apply the camera translation
        pos_proj[0] += t[0];
        pos_proj[1] += t[1];
        pos_proj[2] += t[2];

        const T x = pos_proj[0] / pos_proj[2];
        const T y = pos_proj[1] / pos_proj[2];

        const T fx = intrinsic[0];
        const T fy = intrinsic[1];
        const T cx = intrinsic[2];
        const T cy = intrinsic[3];

        // Apply intrinsic
        const T u = fx * x + cx;
        const T v = fy * y + cy;

        residuals[0] = u - T(observation.x);
        residuals[1] = v - T(observation.y);

        return true;
    }

    static ceres::CostFunction *Create(cv::Point2d &observation)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(
            new ReprojectCost(observation)));
    }
};

void SfMBundleAdjustment::adjustBundle(
    PointCloud &pointCloud,
    std::vector<Pose> &cameraPoses,
    Intrinsics &intrinsics,
    const std::vector<Features> &image2dFeatures)
{

    // =================================================
    // Preproc
    Matx intrinsics_simple(Matx41d(intrinsics.K.at<double>(0, 0),
                                   intrinsics.K.at<double>(1, 1),
                                   intrinsics.K.at<double>(0, 2),
                                   intrinsics.K.at<double>(1, 2)));

    // Convert camera pose parameters from [R|t] (3x4) to [Angle-Axis (3), Translation (3)] (1x6)
    using CameraVector = cv::Matx<double, 1, 6>;

    vector<CameraVector> cameraPoses6d;
    cameraPoses6d.reserve(cameraPoses.size());
    for (size_t i = 0; i < cameraPoses.size(); i++)
    {
        const auto &pose = cameraPoses[i];

        if (pose(0, 0) == 0 && pose(1, 1) == 0 && pose(2, 2) == 0)
        {
            // This camera pose is empty, it should not be used in the optimization
            cameraPoses6d.push_back(CameraVector());
            continue;
        }
        Vec3f t{pose(0, 3), pose(1, 3), pose(2, 3)};
        auto R = pose.get_minor<3, 3>(0, 0);
        float angleAxis[3];
        ceres::RotationMatrixToAngleAxis<float>(R.t().val, angleAxis);

        cameraPoses6d.emplace_back(angleAxis[0], angleAxis[1], angleAxis[2],
                                   t(0), t(1), t(2));
    }

    // =================================================
    // Solve
    std::call_once(Private::initLoggingFlag, Private::initLogging);
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(4);
    vector<cv::Vec3d> points3d(pointCloud.size());

    for (int i = 0; i < pointCloud.size(); i++)
    {
        const Point3DInMap &p = pointCloud[i];
        points3d[i] = cv::Vec3d(p.p.x, p.p.y, p.p.z);

        for (const auto &kv : p.originatingViews)
        {
            //kv.first  = camera index
            //kv.second = 2d feature index
            Point2d p2d = image2dFeatures[kv.first].points[kv.second];

            //subtract center of projection, since the optimizer doesn't know what it is
            p2d.x -= intrinsics.K.at<double>(0, 2); // cx
            p2d.y -= intrinsics.K.at<double>(1, 2); // cy

            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            ceres::CostFunction *cost_function = ReprojectCost::Create(p2d);

            problem.AddResidualBlock(cost_function,
                                     loss_function, //NULL /* squared loss   ` `*/,
                                     intrinsics_simple.val,
                                     cameraPoses6d[kv.first].val,
                                     points3d[i].val);
        }
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
#ifdef _DEBUG
    options.minimizer_progress_to_stdout = true;
#else
    options.minimizer_progress_to_stdout = false;
#endif
    options.logging_type = ceres::LoggingType::SILENT;
    options.preconditioner_type = ceres::JACOBI;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    // options.max_num_iterations = 500;
    // options.eta = 1e-2;
    // options.max_solver_time_in_seconds = 10;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable())
        std::cout << "Bundle Adjustment failed." << std::endl;
    else
        std::cout << summary.BriefReport() << "\n";

    // recover
    //Implement the optimized camera poses and 3D points back into the reconstruction
    for (size_t i = 0; i < cameraPoses.size(); i++)
    {
        Pose &pose = cameraPoses[i];
        Pose poseBefore = pose;

        if (pose(0, 0) == 0 && pose(1, 1) == 0 && pose(2, 2) == 0)
        {
            //This camera pose is empty, it was not used in the optimization
            continue;
        }

        //Convert optimized Angle-Axis back to rotation matrix
        double rotationMat[9] = {0};
        ceres::AngleAxisToRotationMatrix(cameraPoses6d[i].val, rotationMat);

        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                pose(c, r) = rotationMat[r * 3 + c]; //`rotationMat` is col-major...
            }
        }

        //Translation
        pose(0, 3) = cameraPoses6d[i](3);
        pose(1, 3) = cameraPoses6d[i](4);
        pose(2, 3) = cameraPoses6d[i](5);
    }

    for (int i = 0; i < pointCloud.size(); i++)
    {
        pointCloud[i].p.x = points3d[i](0);
        pointCloud[i].p.y = points3d[i](1);
        pointCloud[i].p.z = points3d[i](2);
    }
}
#pragma once
#include <Ikit/STL/Singleton.h>
#include "SfM.h"

#define COOL !

namespace KTKR::MVS
{

    class SfMStereo : public KTKR::Singleton<SfMStereo>
    {
    public:
        /**
         * Find the amount of inlier points in a homography between 2 views.
         * @param left      Left image features
         * @param right     Right image features
         * @param matches   Matching between the features
         * @return          number of inliers.
         */
        int findHomographyInlier(const Features &left,
                                 const Features &right,
                                 const Matching &matches);
        /**
         * Find camera matrices (3x4 poses) from stereo point matching.
         * @param intrinsics        Camera intrinsics (assuming both cameras have the same parameters)
         * @param matches           Matching between the features
         * @param featureMatching   Matching between left and right features
         * @param featuresLeft      Features in left image
         * @param featuresRight     Features in right image
         * @param prunedMatches     Output: matching after pruning using essential matrix
         * @param Pleft             Output: left image matrix (3x4)
         * @param Pright            Output: right image matrix (3x4)
         * @return                  `OK` on success.
         */
        ErrorCode findCameraMatricesFromMatch(const Intrinsics &intrinsics,
                                              const Matching &matches,
                                              const Features &featureL,
                                              const Features &featureR,
                                              Matching &prunedMatches,
                                              cv::Matx34f &Pleft,
                                              cv::Matx34f &Pright);
        /**
         * Triangulate (recover 3D locations) from point matching.
         * @param intrinsics    Camera intrinsics (assuming both cameras have the same parameters)
         * @param imagePair     Indices of left and right views
         * @param matches       Matching between the features
         * @param leftFeatures  Left image features
         * @param rightFeatures Right image features
         * @param Pleft         Left camera matrix
         * @param Pright        Right camera matrix
         * @param pointCloud    Output: point cloud with image associations
         * @return              `OK` on success.
         */
        ErrorCode triangulateViews(const Intrinsics &intrinsics,
                                   const ImagePair imagePair,
                                   const Matching &matches,
                                   const Features &featureL,
                                   const Features &featureR,
                                   cv::Matx34f &Pleft,
                                   cv::Matx34f &Pright,
                                   PointCloud &PointCloud);
    };
} // namespace KTKR::MVS
// =============================================================================
// 05 - ICP Registration — Aligning a Scanned Cloud to its Reference
// =============================================================================
//
// PURPOSE:
//   Load the reference and scanned keyed-cylinder clouds from example 01, apply ICP
//   to recover the rigid transform between them, and verify how close the
//   recovered transform is to the known ground-truth misalignment.
//
// KEY CONCEPTS:
//
//   ICP (Iterative Closest Point) — how it works
//     Repeat until convergence:
//       1. Correspondence: for each source point find its nearest target point.
//       2. Rejection: discard pairs whose distance > max_correspondence_dist.
//       3. Minimisation: find the rotation R and translation t that minimise
//          the sum of squared distances between all accepted pairs.
//          (Solved in closed form via SVD of the cross-covariance matrix.)
//       4. Transform: apply R, t to the source cloud.
//     The fitness score = mean squared distance over all accepted pairs.
//
//   ICP convergence criteria
//     transformation_epsilon  : stop if the incremental transform is smaller
//                               than this (Frobenius norm of ΔT).
//     euclidean_fitness_epsilon: stop if the fitness score improves by less
//                               than this between iterations.
//     max_iterations           : hard cap — always set this as a safety net.
//
//   Why ICP fails without a good initial alignment
//     ICP is a local optimiser.  If source and target are too far apart the
//     correspondences will be wrong from step 1 and it converges to a wrong
//     local minimum.  For large misalignments you need a global initialisation
//     first (FPFH + SampleConsensusPrerejective — see the ADIF project).
//
//   Reading the 4×4 transform matrix
//     │ R R R tx │     R = 3×3 rotation matrix
//     │ R R R ty │     t = translation vector (metres)
//     │ R R R tz │
//     │ 0 0 0  1 │     Multiply source point (as homogeneous column vector)
//                      on the right: p_target = T · p_source
//
// =============================================================================

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Geometry>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "pcl_viewer_utils.h"

// Pretty-print a 4×4 matrix with a label
void printMatrix(const std::string& label, const Eigen::Matrix4f& M) {
    std::cout << label << ":\n";
    for (int r = 0; r < 4; ++r) {
        std::cout << "  │";
        for (int c = 0; c < 4; ++c)
            std::cout << std::setw(10) << std::fixed << std::setprecision(5) << M(r, c);
        std::cout << " │\n";
    }
    std::cout << "\n";
}

// Extract ZYX Euler angles (degrees) from a rotation matrix
Eigen::Vector3f rotationToEulerDeg(const Eigen::Matrix3f& R) {
    Eigen::Vector3f eu = R.eulerAngles(2, 1, 0);  // Z, Y, X order
    return eu * (180.0f / static_cast<float>(M_PI));
}

float rotationErrorDeg(const Eigen::Matrix3f& expected,
                       const Eigen::Matrix3f& recovered) {
    Eigen::Matrix3f delta = expected.transpose() * recovered;
    float trace_term = std::clamp((delta.trace() - 1.0f) * 0.5f, -1.0f, 1.0f);
    return std::acos(trace_term) * 180.0f / static_cast<float>(M_PI);
}

int main() {
    // ── Load clouds produced by 01-Cloud_Generation ───────────────────────────
    auto reference = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                         new pcl::PointCloud<pcl::PointXYZ>());
    auto scanned   = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                         new pcl::PointCloud<pcl::PointXYZ>());

    if (pcl::io::loadPCDFile("data/reference_part.pcd", *reference) < 0 ||
        pcl::io::loadPCDFile("data/scanned_part.pcd",   *scanned)   < 0) {
        std::cerr << "Could not load PCD files.  Run example 01 first.\n";
        return 1;
    }
    std::cout << "Loaded reference : " << reference->size() << " pts\n";
    std::cout << "Loaded scanned   : " << scanned->size()   << " pts\n\n";

    // Ground-truth pose offset applied by example 01 to create the scanned cloud
    // (≈ 0.86° Z-rotation + sub-mm translation).
    // Since ICP aligns scanned → reference, the expected alignment transform is
    // the inverse of this generation transform.
    Eigen::Affine3f T_gt = Eigen::Affine3f::Identity();
    T_gt.rotate(Eigen::AngleAxisf(0.015f, Eigen::Vector3f::UnitZ()));
    T_gt.pretranslate(Eigen::Vector3f(0.003f, -0.002f, 0.001f));
    Eigen::Affine3f T_expected = T_gt.inverse();

    std::cout << "Expected alignment transform (inverse of example 01 offset):\n";
    Eigen::Vector3f eu_gt = rotationToEulerDeg(T_expected.rotation());
    std::cout << "  Rotation  Z/Y/X: " << eu_gt[0] << " / " << eu_gt[1]
              << " / " << eu_gt[2] << "  deg\n";
    std::cout << "  Translation    : "
              << T_expected.translation().transpose() * 1000 << "  mm\n\n";

    // ── Configure ICP ─────────────────────────────────────────────────────────
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    // Source = scanned (unknown pose).  Target = reference (fixed frame).
    icp.setInputSource(scanned);
    icp.setInputTarget(reference);

    // max_correspondence_distance: pairs farther apart than this are rejected.
    // Set to ~3× the expected misalignment magnitude.
    icp.setMaxCorrespondenceDistance(0.020f);   // 20 mm

    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-9);         // converge if ΔT is tiny
    icp.setEuclideanFitnessEpsilon(1e-8);       // converge if score stops improving

    // ── Run ICP ───────────────────────────────────────────────────────────────
    auto aligned = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                       new pcl::PointCloud<pcl::PointXYZ>());
    icp.align(*aligned);

    // ── Report ────────────────────────────────────────────────────────────────
    std::cout << "ICP result:\n";
    std::cout << "  Converged      : " << std::boolalpha << icp.hasConverged() << "\n";
    std::cout << "  Fitness score  : " << std::scientific << std::setprecision(4)
              << icp.getFitnessScore() << "  (mean squared dist, metres²)\n";
    std::cout << "  RMSE           : " << std::sqrt(icp.getFitnessScore()) * 1000
              << "  mm\n\n";

    Eigen::Matrix4f T_icp = icp.getFinalTransformation();
    printMatrix("ICP final transformation T", T_icp);
    printMatrix("Expected alignment transform", T_expected.matrix());

    // Decompose recovered transform
    Eigen::Matrix3f R_rec = T_icp.block<3,3>(0,0);
    Eigen::Vector3f t_rec = T_icp.block<3,1>(0,3);

    float rot_err_deg = rotationErrorDeg(T_expected.rotation(), R_rec);
    Eigen::Vector3f t_err = t_rec - T_expected.translation();

    std::cout << "Recovered vs expected alignment:\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Translation recovered : " << t_rec.transpose() * 1000 << " mm\n";
    std::cout << "  Translation expected  : "
              << T_expected.translation().transpose() * 1000 << " mm\n";
    std::cout << "  Translation error     : " << t_err.norm() * 1000 << " mm\n";
    std::cout << "  Rotation error angle  : " << rot_err_deg << " deg\n\n";

    // ── Visualise: before / after ─────────────────────────────────────────────
    if (!canLaunchViewer()) {
        printViewerSkipMessage("Visualisation");
        return 0;
    }

    try {
        pcl::visualization::PCLVisualizer viewer("05 - ICP Registration");
        int v1(0), v2(0);
        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        viewer.setBackgroundColor(0.08, 0.08, 0.08, v1);
        viewer.setBackgroundColor(0.08, 0.08, 0.08, v2);
        viewer.addText("Before ICP",  5, 15, 12, 1,1,1, "t1", v1);
        viewer.addText("After  ICP",  5, 15, 12, 1,1,1, "t2", v2);

        // Reference — white (both viewports)
        auto addRef = [&](int vp) {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
                w(reference, 200, 200, 200);
            viewer.addPointCloud<pcl::PointXYZ>(reference, w,
                                                "ref_" + std::to_string(vp), vp);
            viewer.setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
                "ref_" + std::to_string(vp));
        };
        addRef(v1); addRef(v2);

        // Scanned (misaligned) — red, left viewport
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            red(scanned, 220, 60, 60);
        viewer.addPointCloud<pcl::PointXYZ>(scanned, red, "scanned", v1);
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scanned");

        // Aligned — green, right viewport
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            green(aligned, 60, 220, 60);
        viewer.addPointCloud<pcl::PointXYZ>(aligned, green, "aligned", v2);
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aligned");

        viewer.addCoordinateSystem(0.04, "ax", 0);
        viewer.addText("White = reference   Red = scanned (shifted)   Green = ICP result",
                       5, 32, 11, 1,1,1, "leg", v1);
        setupInitialView(viewer);

        std::cout << "Left: reference (white) + scanned before ICP (red)\n";
        std::cout << "Right: reference (white) + scanned after ICP (green)\n";
        std::cout << "Press 'q' to quit.\n";
        while (!viewer.wasStopped())
            viewer.spinOnce(100);
    } catch (const std::exception& e) {
        std::cout << "Visualisation skipped: " << e.what() << "\n";
    }

    return 0;
}

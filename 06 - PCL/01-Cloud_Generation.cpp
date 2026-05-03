// =============================================================================
// 01 - Synthetic Part Generation — Building an Inspection Reference Cloud
// =============================================================================
//
// PURPOSE:
//   Generate a realistic synthetic "inspection part" — a cylinder with a flat
//   end cap and a small locator rib — that is used as input for all downstream
//   examples.
//   Also produce a "scanned" version with sensor noise and a simulated surface
//   defect (dent), representing what a real scanner would return.
//
// KEY CONCEPTS:
//
//   Parametric surface sampling
//     Cylinder surface : x = R·cos(θ), y = R·sin(θ), z ∈ [0, H]
//     Disk end cap     : x = r·cos(θ), y = r·sin(θ), z = 0, r ∈ [0, R]
//     Sampling density should be uniform — vary θ-spacing with r on the disk
//     so you don't over-sample near the centre.
//
//   Gaussian sensor noise (σ = 0.5 mm)
//     Represents the measurement uncertainty of a real structured-light or
//     time-of-flight scanner.  Adding it here lets later examples deal with
//     realistic, imperfect data rather than a pristine analytic surface.
//
//   Defect simulation — cosine-tapered dent
//     A patch of surface points is pushed inward along the local outward
//     normal by DENT_DEPTH, with a smooth cosine taper to zero at the edge.
//     This avoids a hard step that would create artificial high-frequency
//     features and confuse feature-based registration.
//
//   Rigid misalignment
//     The scanned cloud is rotated and translated by a small but non-trivial
//     amount.  The ICP example (05) must recover this transform.
//
// OUTPUT:
//   data/reference_part.pcd  — clean parametric model ("golden master")
//   data/scanned_part.pcd    — noisy + dented + misaligned version
//
// =============================================================================

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Geometry>
#include <filesystem>
#include <iostream>
#include <random>
#include <cmath>
#include "pcl_viewer_utils.h"

// ── Part geometry (metres) ────────────────────────────────────────────────────
constexpr float R            = 0.050f;   // 50 mm cylinder radius
constexpr float H            = 0.100f;   // 100 mm cylinder height
constexpr float NOISE_SIGMA  = 0.0005f;  // 0.5 mm 1-σ sensor noise
constexpr float DENT_DEPTH   = 0.003f;   // 3 mm pressed-in dent
constexpr float DENT_RADIUS  = 0.015f;   // 15 mm dent patch radius
constexpr float DENT_THETA   = 1.2f;     // angular position (radians)
constexpr float DENT_Z       = 0.055f;   // axial position
constexpr float KEY_WIDTH    = 0.010f;   // 10 mm locator rib width
constexpr float KEY_DEPTH    = 0.006f;   // 6 mm radial protrusion
constexpr float KEY_Z0       = 0.025f;   // rib starts 25 mm above base
constexpr float KEY_Z1       = 0.080f;   // rib ends   80 mm above base

// ── Build a clean cylinder + end cap + asymmetric locator rib ───────────────
pcl::PointCloud<pcl::PointXYZ>::Ptr buildCylinder(int theta_steps = 250,
                                                   int z_steps     = 125,
                                                   int disk_rings  = 20)
{
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                     new pcl::PointCloud<pcl::PointXYZ>());

    // Lateral surface
    for (int ti = 0; ti < theta_steps; ++ti) {
        float theta = 2.0f * static_cast<float>(M_PI) * ti / theta_steps;
        for (int zi = 0; zi <= z_steps; ++zi)
            cloud->push_back({R * std::cos(theta),
                              R * std::sin(theta),
                              H * zi / z_steps});
    }

    // End cap disk — vary ring density so inter-point spacing stays uniform
    for (int ri = 1; ri <= disk_rings; ++ri) {
        float r       = R * ri / disk_rings;
        int ring_pts  = std::max(6, static_cast<int>(
                            2.0f * static_cast<float>(M_PI) * r
                            / (R / disk_rings)));
        for (int ti = 0; ti < ring_pts; ++ti) {
            float theta = 2.0f * static_cast<float>(M_PI) * ti / ring_pts;
            cloud->push_back({r * std::cos(theta), r * std::sin(theta), 0.0f});
        }
    }

    // Locator rib on the +X side.
    // This intentionally breaks rotational symmetry so ICP can recover yaw
    // around the cylinder axis; a perfect cylinder is underconstrained.
    for (int zi = 0; zi <= 60; ++zi) {
        float z = KEY_Z0 + (KEY_Z1 - KEY_Z0) * zi / 60.0f;
        for (int yi = 0; yi <= 12; ++yi) {
            float y = -KEY_WIDTH * 0.5f + KEY_WIDTH * yi / 12.0f;
            // Outer face of rib
            cloud->push_back({R + KEY_DEPTH, y, z});
            // Side faces connecting rib to cylinder wall
            cloud->push_back({R + KEY_DEPTH * 0.5f, -KEY_WIDTH * 0.5f, z});
            cloud->push_back({R + KEY_DEPTH * 0.5f,  KEY_WIDTH * 0.5f, z});
        }
    }

    cloud->width  = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

// ── Add noise + dent + rigid misalignment ────────────────────────────────────
pcl::PointCloud<pcl::PointXYZ>::Ptr addDefects(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& ref)
{
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, NOISE_SIGMA);

    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                     new pcl::PointCloud<pcl::PointXYZ>(*ref));

    // Pre-compute dent geometry
    Eigen::Vector3f dent_centre(R * std::cos(DENT_THETA),
                                R * std::sin(DENT_THETA),
                                DENT_Z);
    // Outward radial normal at the dent location on the cylinder
    Eigen::Vector3f dent_normal(std::cos(DENT_THETA), std::sin(DENT_THETA), 0.0f);

    for (auto& p : cloud->points) {
        // 1) Gaussian sensor noise — applied to every point
        p.x += gauss(rng);
        p.y += gauss(rng);
        p.z += gauss(rng);

        // 2) Cosine-tapered dent — only within DENT_RADIUS of the dent centre
        Eigen::Vector3f pt(p.x, p.y, p.z);
        float dist = (pt - dent_centre).norm();
        if (dist < DENT_RADIUS) {
            // alpha = 1 at centre, 0 at edge (smooth falloff)
            float alpha = 0.5f * (1.0f + std::cos(
                              static_cast<float>(M_PI) * dist / DENT_RADIUS));
            p.x -= dent_normal.x() * DENT_DEPTH * alpha;
            p.y -= dent_normal.y() * DENT_DEPTH * alpha;
            p.z -= dent_normal.z() * DENT_DEPTH * alpha;
        }
    }

    // 3) Small rigid misalignment — simulates imperfect fixture seating.
    // Keep the angular offset modest so the downstream ICP lesson stays inside
    // the convergence basin of a local point-to-point registration.
    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.rotate(Eigen::AngleAxisf(0.015f, Eigen::Vector3f::UnitZ())); // ≈ 0.86°
    T.pretranslate(Eigen::Vector3f(0.003f, -0.002f, 0.001f));      // sub-mm
    pcl::transformPointCloud(*cloud, *cloud, T);

    return cloud;
}

int main() {
    std::filesystem::create_directories("data");

    // ── Reference (the "golden master") ──────────────────────────────────────
    auto reference = buildCylinder();
    std::cout << "Reference part : " << reference->size() << " points\n";
    pcl::io::savePCDFileASCII("data/reference_part.pcd", *reference);
    std::cout << "Saved → data/reference_part.pcd\n";

    // ── Scanned (noisy, dented, misaligned) ───────────────────────────────────
    auto scanned = addDefects(reference);
    std::cout << "Scanned part   : " << scanned->size()   << " points\n";
    pcl::io::savePCDFileASCII("data/scanned_part.pcd", *scanned);
    std::cout << "Saved → data/scanned_part.pcd\n\n";

    std::cout << "Part parameters:\n"
              << "  Radius         : " << R            * 1000 << " mm\n"
              << "  Height         : " << H            * 1000 << " mm\n"
              << "  Locator rib    : " << KEY_WIDTH    * 1000 << " mm wide, "
              << KEY_DEPTH * 1000 << " mm deep\n"
              << "  Noise sigma    : " << NOISE_SIGMA  * 1000 << " mm\n"
              << "  Dent depth     : " << DENT_DEPTH   * 1000 << " mm\n"
              << "  Dent radius    : " << DENT_RADIUS  * 1000 << " mm\n\n";

    // ── Side-by-side visualisation ────────────────────────────────────────────
    if (!canLaunchViewer()) {
        printViewerSkipMessage("Visualisation");
        return 0;
    }

    try {
        pcl::visualization::PCLVisualizer viewer("01 - Cloud Generation");
        int v1(0), v2(0);
        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        viewer.setBackgroundColor(0.1, 0.1, 0.1, v1);
        viewer.setBackgroundColor(0.1, 0.1, 0.1, v2);
        viewer.addText("Reference (clean)",             10, 18, 13, 1,1,1, "t1", v1);
        viewer.addText("Scanned (noise + dent + shift)",10, 18, 13, 1,1,1, "t2", v2);

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            col_ref(reference, 80, 200, 80);
        viewer.addPointCloud<pcl::PointXYZ>(reference, col_ref, "ref", v1);
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "ref");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            col_scan(scanned, 200, 80, 80);
        viewer.addPointCloud<pcl::PointXYZ>(scanned, col_scan, "scan", v2);
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scan");

        viewer.addCoordinateSystem(0.05, "ax", 0);
        setupInitialView(viewer);

        std::cout << "Left = reference  |  Right = scanned (rotate to see the dent)\n";
        std::cout << "Press 'q' to quit.\n";
        while (!viewer.wasStopped())
            viewer.spinOnce(100);
    } catch (const std::exception& e) {
        std::cout << "Visualisation skipped: " << e.what() << "\n";
    }

    return 0;
}

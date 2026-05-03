// =============================================================================
// 03 - Surface Normal Estimation & the KD-Tree
// =============================================================================
//
// PURPOSE:
//   Compute surface normals on a synthetic sphere — a geometry where the
//   correct answer is analytically known (normals point radially outward).
//   This lets us verify accuracy and understand the impact of neighbourhood
//   size on normal quality.
//
// KEY CONCEPTS:
//
//   KD-Tree (k-dimensional tree)
//     A binary space-partitioning tree.  Nearest-neighbour queries take
//     O(log N) on average instead of O(N) brute force.  Every PCL algorithm
//     that needs neighbourhood information — normals, FPFH features, ICP
//     correspondence search — uses a KD-tree internally.
//
//   Normal estimation (PCA-based)
//     For each point p, collect its K nearest neighbours.  Fit a plane to
//     those K points using PCA (find the eigenvector of the covariance matrix
//     with the smallest eigenvalue — that's the normal direction).
//     The ratio  λ_min / (λ_min + λ_mid + λ_max)  is the curvature scalar.
//
//   Neighbourhood size trade-off
//     Small K / small radius → captures fine details, but noisy normals.
//     Large K / large radius → smooth normals, but blurs sharp edges.
//     For inspection: K = 20–30 on a VoxelGrid-downsampled cloud is typical.
//
//   Why normals matter in inspection
//     1. ICP point-to-plane variant: converges ~3× faster than point-to-point.
//     2. FPFH features: encode local surface geometry using normal differences.
//     3. Signed deviation: sign(dot(displacement, normal)) tells you whether
//        the scanned surface is inside or outside the reference model.
//     4. Curvature: high curvature flags sharp features or noise spikes.
//
// =============================================================================

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "pcl_viewer_utils.h"

// Sample a sphere surface parametrically — normals should point radially out
pcl::PointCloud<pcl::PointXYZ>::Ptr buildSphere(float radius    = 0.05f,
                                                 int   phi_steps = 40,
                                                 int   theta_steps = 80)
{
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                     new pcl::PointCloud<pcl::PointXYZ>());
    for (int pi = 0; pi <= phi_steps; ++pi) {
        float phi = static_cast<float>(M_PI) * pi / phi_steps;   // 0 … π
        for (int ti = 0; ti < theta_steps; ++ti) {
            float theta = 2.0f * static_cast<float>(M_PI) * ti / theta_steps;
            cloud->push_back({
                radius * std::sin(phi) * std::cos(theta),
                radius * std::sin(phi) * std::sin(theta),
                radius * std::cos(phi)
            });
        }
    }
    cloud->width  = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

// Verify normals: for a unit sphere centred at origin each normal should be
// close to the normalised position vector of the point.
void evaluateNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr&  cloud,
                     const pcl::PointCloud<pcl::Normal>::Ptr&     normals,
                     float radius)
{
    double sum_angle_err = 0.0;
    int    nan_count     = 0;
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        const auto& p = cloud->points[i];
        const auto& n = normals->points[i];
        if (!std::isfinite(n.normal_x)) { ++nan_count; continue; }

        // Expected normal = radially outward unit vector
        Eigen::Vector3f expected(p.x / radius, p.y / radius, p.z / radius);
        Eigen::Vector3f computed(n.normal_x, n.normal_y, n.normal_z);
        // PCL flips normals toward viewpoint; use absolute angle
        float dot   = std::abs(expected.dot(computed));
        float angle = std::acos(std::min(1.0f, dot)) * 180.0f /
                      static_cast<float>(M_PI);
        sum_angle_err += angle;
    }
    std::size_t valid = cloud->size() - nan_count;
    std::cout << "  NaN normals   : " << nan_count << "\n";
    std::cout << "  Mean angle err: " << std::fixed << std::setprecision(2)
              << (valid ? sum_angle_err / valid : 0.0) << " deg"
              << "  (0 = perfect, <5 deg = good)\n";
}

int main() {
    auto sphere = buildSphere(0.05f, 40, 80);
    std::cout << "Sphere cloud: " << sphere->size() << " points\n\n";

    // ── Build the KD-Tree ─────────────────────────────────────────────────────
    // The search tree is shared — build it once, reuse across K values.
    auto tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(
                    new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(sphere);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(sphere);
    ne.setSearchMethod(tree);

    // ── Compare K = 8, 20, 60 ────────────────────────────────────────────────
    for (int K : {8, 20, 60}) {
        ne.setKSearch(K);
        auto normals = pcl::PointCloud<pcl::Normal>::Ptr(
                           new pcl::PointCloud<pcl::Normal>());
        ne.compute(*normals);

        std::cout << "K = " << std::setw(2) << K << " neighbours:\n";
        evaluateNormals(sphere, normals, 0.05f);

        // Curvature on a perfect sphere is constant — deviation from this
        // is a proxy for how much the neighbourhood blurs the surface.
        float sum_curv = 0.0f; int cnt = 0;
        for (const auto& n : normals->points) {
            if (std::isfinite(n.curvature)) { sum_curv += n.curvature; ++cnt; }
        }
        std::cout << "  Mean curvature: " << (cnt ? sum_curv / cnt : 0.0f)
                  << "  (lower/more-uniform = better for smooth surface)\n\n";
    }

    // ── Visualise with K = 20 ─────────────────────────────────────────────────
    ne.setKSearch(20);
    auto normals = pcl::PointCloud<pcl::Normal>::Ptr(
                       new pcl::PointCloud<pcl::Normal>());
    ne.compute(*normals);

    if (!canLaunchViewer()) {
        printViewerSkipMessage("Visualisation");
        return 0;
    }

    try {
        pcl::visualization::PCLVisualizer viewer("03 - Normal Estimation");
        viewer.setBackgroundColor(0.08, 0.08, 0.08);

        // Sphere surface — white
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            white(sphere, 220, 220, 220);
        viewer.addPointCloud<pcl::PointXYZ>(sphere, white, "sphere");
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sphere");

        // Normals — cyan lines (every 4th point, 15 mm long)
        viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(
            sphere, normals, 4, 0.015f, "normals");

        viewer.addCoordinateSystem(0.03, "ax", 0);
        viewer.addText("White = sphere surface   Cyan = normals (K=20, every 4th pt)",
                       5, 15, 12, 1, 1, 1, "legend");
        setupInitialView(viewer);

        std::cout << "White = sphere  |  Cyan = normals.\n"
                  << "Verify: each normal should point radially outward.\n"
                  << "Press 'q' to quit.\n";
        while (!viewer.wasStopped())
            viewer.spinOnce(100);
    } catch (const std::exception& e) {
        std::cout << "Visualisation skipped: " << e.what() << "\n";
    }

    return 0;
}

// =============================================================================
// 04 - RANSAC Plane Segmentation — Isolating a Part from its Fixture
// =============================================================================
//
// PURPOSE:
//   In a real inspection cell a scanner sees the part AND the fixture/table
//   it sits on.  Before you can measure the part you must remove the flat
//   surfaces.  This example shows how to do that iteratively with RANSAC.
//
// SCENE:
//   Floor plane  z = 0  (dominant feature — huge flat surface)
//   └── Cylinder sitting on the floor (the "part" we want to inspect)
//
// KEY CONCEPTS:
//
//   RANSAC (Random Sample Consensus)
//     1. Randomly pick the minimum number of points needed to define the model
//        (3 for a plane).
//     2. Count how many other points lie within distance_threshold of that
//        model — these are "inliers".
//     3. Repeat N times.  Keep the model with the most inliers.
//     4. Re-fit the model to ALL its inliers (least-squares polish).
//     The result is robust to outliers — even 40 % noise won't break it.
//
//   distance_threshold
//     The half-width of the inlier band around the model.
//     Too small → inliers miss noisy-but-valid surface points.
//     Too large → the plane absorbs points from curved surfaces nearby.
//     Rule of thumb: 2–3 × sensor noise sigma.
//
//   Iterative plane removal (multi-layer segmentation)
//     After extracting the dominant plane, remove its inliers from the cloud
//     and run RANSAC again to find the next plane.  Repeat until the remaining
//     cloud is too small or the inlier ratio drops below a threshold.
//     This isolates the part from a multi-surface fixture.
//
//   ExtractIndices
//     PCL's way of splitting a cloud into inliers / outliers given an index
//     set — more efficient than building a std::set and iterating manually.
//
// =============================================================================

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "pcl_viewer_utils.h"

// Build scene: floor + cylinder sitting on it + a few outlier noise points
pcl::PointCloud<pcl::PointXYZ>::Ptr buildScene() {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                     new pcl::PointCloud<pcl::PointXYZ>());

    // Floor — 400 × 400 mm flat grid at z = 0
    for (int x = -20; x <= 20; ++x)
        for (int y = -20; y <= 20; ++y)
            cloud->push_back({x * 0.010f, y * 0.010f, 0.0f});

    // Cylinder sitting on the floor (base at z = 0)
    for (int ti = 0; ti < 200; ++ti) {
        float theta = 2.0f * static_cast<float>(M_PI) * ti / 200;
        for (int zi = 0; zi <= 100; ++zi)
            cloud->push_back({0.04f * std::cos(theta),
                              0.04f * std::sin(theta),
                              0.10f * zi / 100.0f});
    }
    // Cylinder end cap
    for (int ri = 1; ri <= 8; ++ri) {
        float r = 0.04f * ri / 8.0f;
        int   n = std::max(6, static_cast<int>(2.0f * static_cast<float>(M_PI) * r / 0.005f));
        for (int ti = 0; ti < n; ++ti) {
            float theta = 2.0f * static_cast<float>(M_PI) * ti / n;
            cloud->push_back({r * std::cos(theta), r * std::sin(theta), 0.10f});
        }
    }

    // A few random noise points (multipath / stray reflections)
    for (int i = 0; i < 60; ++i)
        cloud->push_back({(i % 10 - 5) * 0.03f,
                          (i / 10 - 3) * 0.03f,
                          0.05f + (i % 7) * 0.01f});

    cloud->width  = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

int main() {
    auto scene = buildScene();
    std::cout << "Scene points: " << scene->size() << "\n\n";

    // Configure RANSAC segmentation (reuse the same object)
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    // distance_threshold = 3 mm — roughly 3× the 1 mm noise on the floor
    seg.setDistanceThreshold(0.003f);
    // Maximum inlier ratio to keep going — stop if a "plane" has < 5 % inliers
    constexpr float MIN_INLIER_RATIO = 0.05f;

    pcl::ExtractIndices<pcl::PointXYZ> extractor;

    // Working copy — will be progressively stripped of planes
    auto remaining = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                         new pcl::PointCloud<pcl::PointXYZ>(*scene));

    // Accumulate extracted planes for visualisation
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> planes;
    int pass = 0;

    while (remaining->size() > 100) {
        pcl::PointIndices::Ptr  inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients());

        seg.setInputCloud(remaining);
        seg.segment(*inliers, *coeff);

        float ratio = static_cast<float>(inliers->indices.size()) /
                      static_cast<float>(remaining->size());

        if (inliers->indices.empty() || ratio < MIN_INLIER_RATIO)
            break;

        ++pass;
        std::cout << "Pass " << pass << ":\n";
        std::cout << "  Plane   ax+by+cz+d=0 : ";
        for (float c : coeff->values) std::cout << std::setw(8) << std::fixed
                                                 << std::setprecision(4) << c << " ";
        std::cout << "\n";
        std::cout << "  Inliers : " << inliers->indices.size()
                  << " (" << std::setprecision(1) << ratio * 100 << "% of remaining)\n";

        // Extract inliers (the plane) into a separate cloud
        auto plane_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                               new pcl::PointCloud<pcl::PointXYZ>());
        extractor.setInputCloud(remaining);
        extractor.setIndices(inliers);
        extractor.setNegative(false);
        extractor.filter(*plane_cloud);
        planes.push_back(plane_cloud);

        // Remove inliers — keep the non-plane points for next iteration
        auto next = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                        new pcl::PointCloud<pcl::PointXYZ>());
        extractor.setNegative(true);
        extractor.filter(*next);
        remaining = next;

        std::cout << "  Remaining: " << remaining->size() << " pts\n\n";
    }

    std::cout << "After plane removal: " << remaining->size()
              << " points — this is the isolated part.\n\n";

    // ── Visualise ─────────────────────────────────────────────────────────────
    if (!canLaunchViewer()) {
        printViewerSkipMessage("Visualisation");
        return 0;
    }

    try {
        pcl::visualization::PCLVisualizer viewer("04 - Plane Segmentation");
        viewer.setBackgroundColor(0.08, 0.08, 0.08);

        // Colour planes progressively: first = grey, second = orange, …
        const std::vector<std::array<int,3>> plane_colours = {
            {140, 140, 140}, {210, 140, 50}, {50, 140, 210}
        };
        for (std::size_t i = 0; i < planes.size(); ++i) {
            auto [r, g, b] = plane_colours[i % plane_colours.size()];
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
                col(planes[i], r, g, b);
            viewer.addPointCloud<pcl::PointXYZ>(
                planes[i], col, "plane_" + std::to_string(i));
            viewer.setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
                "plane_" + std::to_string(i));
        }

        // Remaining "part" cloud — green
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            green(remaining, 80, 220, 80);
        viewer.addPointCloud<pcl::PointXYZ>(remaining, green, "part");
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "part");

        viewer.addCoordinateSystem(0.05, "ax", 0);
        viewer.addText("Grey = floor   Green = isolated part", 5, 15, 12, 1,1,1, "leg");
        setupInitialView(viewer);

        std::cout << "Grey = extracted floor plane   |   Green = isolated part\n";
        std::cout << "Press 'q' to quit.\n";
        while (!viewer.wasStopped())
            viewer.spinOnce(100);
    } catch (const std::exception& e) {
        std::cout << "Visualisation skipped: " << e.what() << "\n";
    }

    return 0;
}

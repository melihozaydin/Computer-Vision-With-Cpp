// =============================================================================
// 02 - Pre-Processing Pipeline: PassThrough → VoxelGrid → StatisticalOutlierRemoval
// =============================================================================
//
// PURPOSE:
//   Raw scans are never clean.  Before any registration or deviation analysis
//   you need to: crop junk outside your workspace, normalise point density,
//   and kill noise spikes.  This three-stage pipeline does all three.
//
// PIPELINE:
//
//   Raw scan
//     │
//     ├─► PassThrough (axis-aligned crop)
//     │     Removes points outside the known part volume.
//     │     Kills fixture/table points below the part, stray reflections above.
//     │     Cost: O(N) — always run first, it makes every subsequent step faster.
//     │
//     ├─► VoxelGrid (density normalisation)
//     │     Divides space into equal-sized cubes ("voxels").
//     │     Replaces every cluster of points in a voxel with their centroid.
//     │     Why: sensors over-sample near-surfaces. Non-uniform density biases
//     │     ICP (dense regions pull the solution harder than sparse ones).
//     │     Rule of thumb: leaf_size ≈ 0.3 × your deviation tolerance.
//     │
//     └─► StatisticalOutlierRemoval (spike removal)
//           For each point computes the mean distance to its K nearest neighbours.
//           If mean distance > μ + multiplier·σ → classified as outlier, removed.
//           Kills isolated spikes that VoxelGrid cannot touch (a single point
//           in an otherwise empty voxel survives the grid filter intact).
//
// =============================================================================

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include "pcl_viewer_utils.h"

// Build a dense cylinder scan with fixture points and noise spikes
pcl::PointCloud<pcl::PointXYZ>::Ptr makeDirtyScan() {
    std::mt19937 rng(7);
    std::normal_distribution<float>  gauss(0.0f, 0.001f);   // 1 mm noise
    std::uniform_real_distribution<float> spike(-0.12f, 0.12f);

    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                     new pcl::PointCloud<pcl::PointXYZ>());

    // Over-sampled cylinder surface (many overlapping scan lines)
    for (int ti = 0; ti < 400; ++ti) {
        float theta = 2.0f * static_cast<float>(M_PI) * ti / 400;
        for (int zi = 0; zi <= 200; ++zi) {
            cloud->push_back({
                0.05f * std::cos(theta) + gauss(rng),
                0.05f * std::sin(theta) + gauss(rng),
                0.1f  * zi / 200.0f    + gauss(rng)
            });
        }
    }

    // Fixture / table — flat grid below the part (z = -0.020)
    for (int x = -12; x <= 12; ++x)
        for (int y = -12; y <= 12; ++y)
            cloud->push_back({x * 0.012f, y * 0.012f, -0.020f});

    // Multipath noise spikes — isolated random points far from the surface
    for (int i = 0; i < 400; ++i)
        cloud->push_back({spike(rng), spike(rng), spike(rng)});

    cloud->width    = static_cast<uint32_t>(cloud->size());
    cloud->height   = 1;
    cloud->is_dense = false;   // contains potential NaN/outliers
    return cloud;
}

int main() {
    auto raw = makeDirtyScan();
    std::cout << "=== Pre-Processing Pipeline ===\n";
    std::cout << std::left
              << std::setw(38) << "Stage 0  Raw scan"
              << raw->size() << " pts\n";

    // ── Stage 1: PassThrough — crop workspace ─────────────────────────────────
    // Keep only Z ∈ [−1 mm, 102 mm]: the cylinder, not the fixture below.
    pcl::PassThrough<pcl::PointXYZ> pass;
    auto s1 = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pass.setInputCloud(raw);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.001f, 0.102f);
    pass.filter(*s1);
    std::cout << std::setw(38) << "Stage 1a PassThrough Z"
              << s1->size() << " pts  (fixture removed)\n";

    // Also crop XY: keep radius ≤ 80 mm from axis (removes wide-angle spikes)
    auto s1b = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pass.setInputCloud(s1);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-0.08f, 0.08f);
    pass.filter(*s1b);
    auto s1c = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pass.setInputCloud(s1b);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.08f, 0.08f);
    pass.filter(*s1c);
    std::cout << std::setw(38) << "Stage 1b PassThrough XY"
              << s1c->size() << " pts  (wide-angle spikes removed)\n";

    // ── Stage 2: VoxelGrid — normalise density ────────────────────────────────
    // 2 mm voxels.  With a 0.5 mm tolerance this is coarser than ideal, but
    // exaggerated here so the point-count reduction is clearly visible.
    auto s2 = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(s1c);
    vg.setLeafSize(0.002f, 0.002f, 0.002f);
    vg.filter(*s2);
    std::cout << std::setw(38) << "Stage 2  VoxelGrid 2 mm"
              << s2->size() << " pts  ("
              << std::fixed << std::setprecision(1)
              << 100.0f * s2->size() / s1c->size() << "% retained)\n";

    // ── Stage 3: StatisticalOutlierRemoval — kill remaining spikes ────────────
    // meank = 50 : examine each point's 50 nearest neighbours.
    // stddev = 2 : points with mean_dist > global_mean + 2·σ → outlier.
    auto s3 = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(s2);
    sor.setMeanK(50);
    sor.setStddevMulThresh(2.0);
    sor.filter(*s3);
    std::cout << std::setw(38) << "Stage 3  StatisticalOutlierRemoval"
              << s3->size() << " pts  ("
              << (s2->size() - s3->size()) << " outliers removed)\n\n";

    float total = 100.0f * (1.0f - static_cast<float>(s3->size()) / raw->size());
    std::cout << "Total reduction vs raw: " << total << "% of points discarded.\n\n";

    // Optionally save the cleaned cloud for use by later examples
    std::filesystem::create_directories("data");
    pcl::io::savePCDFileASCII("data/cleaned_scan.pcd", *s3);
    std::cout << "Saved → data/cleaned_scan.pcd\n\n";

    // ── Visualise: raw (left) vs pipeline output (right) ─────────────────────
    if (!canLaunchViewer()) {
        printViewerSkipMessage("Visualisation");
        return 0;
    }

    try {
        pcl::visualization::PCLVisualizer viewer("02 - Pre-Processing Pipeline");
        int v1(0), v2(0);
        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        viewer.setBackgroundColor(0.08, 0.08, 0.08, v1);
        viewer.setBackgroundColor(0.08, 0.08, 0.08, v2);
        viewer.addText("Raw scan (fixture + spikes + noise)",      5, 15, 12, 1,1,1, "t1", v1);
        viewer.addText("After PassThrough + VoxelGrid + SOR",      5, 15, 12, 1,1,1, "t2", v2);

        // Raw — grey
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            cRaw(raw, 150, 150, 150);
        viewer.addPointCloud<pcl::PointXYZ>(raw, cRaw, "raw", v1);
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "raw");

        // Cleaned — cyan
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            cClean(s3, 60, 210, 210);
        viewer.addPointCloud<pcl::PointXYZ>(s3, cClean, "clean", v2);
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "clean");

        viewer.addCoordinateSystem(0.05, "ax", 0);
        setupInitialView(viewer);

        std::cout << "Left = raw scan   |   Right = cleaned cloud\n";
        std::cout << "Press 'q' to quit.\n";
        while (!viewer.wasStopped())
            viewer.spinOnce(100);
    } catch (const std::exception& e) {
        std::cout << "Visualisation skipped: " << e.what() << "\n";
    }

    return 0;
}

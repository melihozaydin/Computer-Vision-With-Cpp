// =============================================================================
// 00 - PCL Setup Verification & Point Type Survey
// =============================================================================
//
// PURPOSE:
//   Verify your PCL/Eigen installation and understand the foundational
//   data structures that every PCL algorithm is built on.
//
// KEY CONCEPTS:
//
//   Point Types — PCL uses compile-time polymorphism (templates), not virtual
//   dispatch.  You pick the point type that matches your data at compile time.
//   Common types used in inspection:
//
//     PointXYZ         — raw geometry only (3 floats + 4-byte SSE padding)
//     PointXYZRGB      — geometry + per-point colour (deviation heat-maps)
//     Normal           — surface normal vector + curvature scalar
//     PointNormal      — PointXYZ + Normal combined in one struct
//     PointXYZRGBNormal — everything: position + colour + normal
//
//   All structs are 16-byte aligned for SIMD (SSE/AVX) vectorisation.
//   That padding is intentional — never strip it.
//
//   Cloud organisation:
//     Unorganised: height = 1, width = N  — just a flat list of N points.
//                  Comes from LiDAR sweeps or offline processing.
//     Organised  : height = H, width = W — a 2-D grid (depth camera frame).
//                  Enables fast row/column neighbourhood queries.
//     is_dense   : false if the cloud may contain NaN / Inf coordinates.
//                  Many filters require is_dense = false handling.
//
// =============================================================================

#include <pcl/pcl_config.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include "pcl_viewer_utils.h"

int main() {
    // ── Version banner ────────────────────────────────────────────────────────
    std::cout << "================================================\n";
    std::cout << " PCL   version : " << PCL_VERSION_PRETTY << "\n";
    std::cout << " Eigen version : "
              << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << "\n";
    std::cout << "================================================\n\n";

    // ── Point type sizes — confirms SSE alignment is active ──────────────────
    // Each struct is padded to a multiple of 16 bytes so SIMD loads work on
    // contiguous cloud memory without alignment faults.
    std::cout << "Point type sizes (bytes — includes SSE alignment padding):\n";
    std::cout << std::left
              << "  " << std::setw(22) << "PointXYZ"
              << sizeof(pcl::PointXYZ)          << "\n"
              << "  " << std::setw(22) << "PointXYZRGB"
              << sizeof(pcl::PointXYZRGB)        << "\n"
              << "  " << std::setw(22) << "Normal"
              << sizeof(pcl::Normal)             << "\n"
              << "  " << std::setw(22) << "PointNormal"
              << sizeof(pcl::PointNormal)        << "\n"
              << "  " << std::setw(22) << "PointXYZRGBNormal"
              << sizeof(pcl::PointXYZRGBNormal)  << "\n\n";

    // ── Build a coloured coordinate-axis cloud ────────────────────────────────
    // Using PointXYZRGB — the type used throughout this series for deviation
    // heat-maps (green = in-tolerance, red = oversized, blue = undersized).
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    auto addAxisLine = [&](int axis, uint8_t r, uint8_t g, uint8_t b) {
        for (int i = 0; i <= 30; ++i) {
            pcl::PointXYZRGB p{};
            float t = i * 0.02f;   // 2 cm steps → 60 cm total
            p.x = (axis == 0) ? t : 0.0f;
            p.y = (axis == 1) ? t : 0.0f;
            p.z = (axis == 2) ? t : 0.0f;
            p.r = r; p.g = g; p.b = b;
            cloud->push_back(p);
        }
    };
    addAxisLine(0, 255,  60,  60);  // X → red
    addAxisLine(1,  60, 255,  60);  // Y → green
    addAxisLine(2,  60,  60, 255);  // Z → blue

    cloud->width  = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;          // unorganised
    cloud->is_dense = true;     // no NaN/Inf in this cloud

    std::cout << "Cloud metadata (unorganised example):\n";
    std::cout << "  points   : " << cloud->size()  << "\n";
    std::cout << "  width    : " << cloud->width   << "  (= N for unorganised)\n";
    std::cout << "  height   : " << cloud->height  << "  (= 1 for unorganised)\n";
    std::cout << "  is_dense : " << std::boolalpha << cloud->is_dense << "\n\n";

    // ── Visualise ─────────────────────────────────────────────────────────────
    if (!canLaunchViewer()) {
        printViewerSkipMessage("Viewer");
        return 0;
    }

    try {
        pcl::visualization::PCLVisualizer viewer("00 - PCL Setup");
        viewer.addPointCloud<pcl::PointXYZRGB>(cloud, "axes_cloud");
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "axes_cloud");
        viewer.addCoordinateSystem(0.2, "world", 0);
        viewer.addText("Red = X   Green = Y   Blue = Z", 10, 20, 14, 1.0f, 1.0f, 1.0f, "legend");
        setupInitialView(viewer);

        std::cout << "Viewer open.  Rotate: left-mouse  |  Zoom: scroll  |  Quit: q\n";
        while (!viewer.wasStopped())
            viewer.spinOnce(100);
    } catch (const std::exception& e) {
        std::cout << "Visualisation skipped (headless): " << e.what() << "\n";
    }

    return 0;
}

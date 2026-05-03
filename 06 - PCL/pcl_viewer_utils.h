#pragma once

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/**
 * Setup initial camera view to display point cloud(s) properly.
 * Resets camera to auto-fit all geometry, sets reasonable lighting and background.
 */
inline void setupInitialView(pcl::visualization::PCLVisualizer& viewer) {
    viewer.resetCamera();
    viewer.setBackgroundColor(0.1f, 0.1f, 0.1f);
}

/**
 * Setup initial camera view with optional custom focal point offset.
 * Useful for multi-cloud scenes where default fit may not be ideal.
 */
inline void setupInitialViewWithOffset(
    pcl::visualization::PCLVisualizer& viewer,
    float fx = 0.0f, float fy = 0.0f, float fz = 0.0f) {
    viewer.resetCamera();
    viewer.setBackgroundColor(0.1f, 0.1f, 0.1f);
    if (fx != 0.0f || fy != 0.0f || fz != 0.0f) {
        viewer.setCameraClipDistances(0.00001, 50000.01);
    }
}

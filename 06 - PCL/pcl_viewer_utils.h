#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/**
 * Returns true when an interactive display server is available for VTK/PCL.
 * This avoids hard process aborts in headless WSL / CI environments where
 * constructing a PCLVisualizer can fail before exceptions are thrown.
 */
inline bool canLaunchViewer() {
    const char* force_headless = std::getenv("PCL_FORCE_HEADLESS");
    if (force_headless != nullptr && std::string(force_headless) == "1")
        return false;

    const char* display = std::getenv("DISPLAY");
    if (display != nullptr && display[0] != '\0')
        return true;

    const char* wayland = std::getenv("WAYLAND_DISPLAY");
    if (wayland != nullptr && wayland[0] != '\0')
        return true;

    const char* session = std::getenv("XDG_SESSION_TYPE");
    return session != nullptr && std::string(session) == "wayland";
}

inline void printViewerSkipMessage(const std::string& title) {
    std::cout << title
              << " skipped (no interactive DISPLAY/WAYLAND_DISPLAY detected).\n";
}

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

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include "pcl_viewer_utils.h"

int main() {
    // Create source cloud (original)
    auto source = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    
    // Create target cloud (transformed version of source)
    auto target = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

    // Generate matching point sets
    for (int i = 0; i < 100; ++i) {
        float x = 0.01f * i;
        float y = 0.02f * (i % 10);
        
        source->push_back({x, y, 0.0f});
        // Target is slightly rotated and translated
        target->push_back({x + 0.05f, y - 0.02f, 0.0f});
    }

    std::cout << "Source points: " << source->size() << std::endl;
    std::cout << "Target points: " << target->size() << std::endl;

    // Run ICP to align source to target
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-6);
    
    auto aligned = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    icp.align(*aligned);

    std::cout << "ICP converged: " << (icp.hasConverged() ? "yes" : "no") << std::endl;
    std::cout << "Fitness score: " << icp.getFitnessScore() << std::endl;
    std::cout << "Max iterations (configured): " << icp.getMaximumIterations() << std::endl;

    // Visualize: source (red) → aligned (green), target (blue)
    try {
        pcl::visualization::PCLVisualizer viewer("ICP Registration");
        
        // Source cloud in red (before alignment)
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(source, 255, 0, 0);
        viewer.addPointCloud<pcl::PointXYZ>(source, red, "source");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");

        // Aligned cloud in green (after ICP)
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(aligned, 0, 255, 0);
        viewer.addPointCloud<pcl::PointXYZ>(aligned, green, "aligned");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned");

        // Target cloud in blue (reference)
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(target, 0, 0, 255);
        viewer.addPointCloud<pcl::PointXYZ>(target, blue, "target");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");

        viewer.addCoordinateSystem(0.1, "axes", 0);
        setupInitialView(viewer);
        
        std::cout << "Red = source, Green = aligned (after ICP), Blue = target. Press 'q' to quit." << std::endl;
        
        while (!viewer.wasStopped()) {
            viewer.spinOnce(100);
        }
    } catch (const std::exception& e) {
        std::cout << "Visualization skipped (headless environment): " << e.what() << std::endl;
    }

    return 0;
}

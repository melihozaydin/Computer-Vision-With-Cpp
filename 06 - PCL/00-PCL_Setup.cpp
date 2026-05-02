#include <pcl/pcl_config.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>

int main() {
    std::cout << "PCL version: " << PCL_VERSION_PRETTY << std::endl;

    // Create a tiny synthetic cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->push_back({0.0f, 0.0f, 0.0f});
    cloud->push_back({1.0f, 0.0f, 0.0f});
    cloud->push_back({0.0f, 1.0f, 0.0f});
    cloud->push_back({0.0f, 0.0f, 1.0f});

    std::cout << "Synthetic cloud size: " << cloud->size() << std::endl;

    // Attempt visualization
    try {
        pcl::visualization::PCLVisualizer viewer("Setup Test");
        viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
        viewer.addCoordinateSystem(0.5, "axes", 0);
        viewer.initCameraParameters();
        
        std::cout << "Visualizer opened. Rotate with mouse, press 'q' to quit." << std::endl;
        
        // Spin viewer for 5 seconds or until user closes
        while (!viewer.wasStopped()) {
            viewer.spinOnce(100);
        }
    } catch (const std::exception& e) {
        std::cout << "Visualization skipped (headless environment): " << e.what() << std::endl;
    }

    return 0;
}

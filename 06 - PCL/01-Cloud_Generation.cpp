#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include "pcl_viewer_utils.h"

int main() {
    // Generate a synthetic wavy line cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->width = 200;
    cloud->height = 1;
    cloud->is_dense = true;
    cloud->points.resize(cloud->width * cloud->height);

    for (std::size_t i = 0; i < cloud->points.size(); ++i) {
        float x = static_cast<float>(i) * 0.01f;
        cloud->points[i].x = x;
        cloud->points[i].y = 0.5f * x;
        cloud->points[i].z = 0.1f * std::sin(x * 6.28f);
    }

    // Save to PCD file
    pcl::io::savePCDFileASCII("synthetic_line_cloud.pcd", *cloud);
    std::cout << "Saved synthetic_line_cloud.pcd with " << cloud->size() << " points" << std::endl;

    // Visualize the cloud
    try {
        pcl::visualization::PCLVisualizer viewer("Cloud Generation");
        viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
        viewer.addCoordinateSystem(0.2, "axes", 0);
        setupInitialView(viewer);
        
        std::cout << "Showing generated wavy line cloud. Press 'q' to quit." << std::endl;
        
        while (!viewer.wasStopped()) {
            viewer.spinOnce(100);
        }
    } catch (const std::exception& e) {
        std::cout << "Visualization skipped (headless environment): " << e.what() << std::endl;
    }

    return 0;
}

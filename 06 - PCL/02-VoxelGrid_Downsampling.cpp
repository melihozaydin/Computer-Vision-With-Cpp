#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>

int main() {
    // Generate a dense 3D grid cloud
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    for (int x = 0; x < 50; ++x)
        for (int y = 0; y < 50; ++y)
            for (int z = 0; z < 4; ++z)
                cloud->push_back({x * 0.01f, y * 0.01f, z * 0.01f});

    std::cout << "Original points: " << cloud->size() << std::endl;

    // Downsample using voxel grid (averages points in each cubic cell)
    auto filtered = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);
    vg.filter(*filtered);

    std::cout << "Downsampled points: " << filtered->size() << std::endl;
    std::cout << "Compression ratio: " << (100.0 * filtered->size() / cloud->size()) << "%" << std::endl;

    // Visualize before and after
    try {
        pcl::visualization::PCLVisualizer viewer("VoxelGrid Downsampling");
        
        // Original cloud in red
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud, 255, 0, 0);
        viewer.addPointCloud<pcl::PointXYZ>(cloud, red, "original");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original");

        // Filtered cloud in blue
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(filtered, 0, 0, 255);
        viewer.addPointCloud<pcl::PointXYZ>(filtered, blue, "filtered");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "filtered");

        viewer.addCoordinateSystem(0.1, 0);
        viewer.initCameraParameters();
        
        std::cout << "Red = original (dense), Blue = downsampled. Press 'q' to quit." << std::endl;
        
        while (!viewer.wasStopped()) {
            viewer.spinOnce(100);
        }
    } catch (const std::exception& e) {
        std::cout << "Visualization skipped (headless environment): " << e.what() << std::endl;
    }

    return 0;
}

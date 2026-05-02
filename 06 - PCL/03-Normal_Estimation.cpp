#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>

int main() {
    // Generate a flat surface (XY plane with slight Z variation)
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    for (int x = 0; x < 25; ++x)
        for (int y = 0; y < 25; ++y)
            cloud->push_back({x * 0.02f, y * 0.02f, 0.01f * (x + y)});

    std::cout << "Cloud points: " << cloud->size() << std::endl;

    // Estimate normals using KD-tree neighborhood
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    auto tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setKSearch(15);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    ne.compute(*normals);

    std::cout << "Computed normals: " << normals->size() << std::endl;
    if (!normals->empty()) {
        std::cout << "First normal: (" << normals->points[0].normal_x << ", "
                  << normals->points[0].normal_y << ", " << normals->points[0].normal_z << ")" << std::endl;
    }

    // Visualize cloud and normals
    try {
        pcl::visualization::PCLVisualizer viewer("Normal Estimation");
        
        // Show cloud in green
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloud, 0, 255, 0);
        viewer.addPointCloud<pcl::PointXYZ>(cloud, green, "cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

        // Draw normal vectors as cyan lines
        viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 5, 0.02, "normals");

        viewer.addCoordinateSystem(0.1, "axes", 0);
        viewer.initCameraParameters();
        
        std::cout << "Green points = cloud, Cyan lines = normals (every 5th). Press 'q' to quit." << std::endl;
        
        while (!viewer.wasStopped()) {
            viewer.spinOnce(100);
        }
    } catch (const std::exception& e) {
        std::cout << "Visualization skipped (headless environment): " << e.what() << std::endl;
    }

    return 0;
}

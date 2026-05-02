#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <set>
#include <iostream>

int main() {
    // Create synthetic data: a plane (XY at Z=0) plus some outlier points
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    
    // Add plane points
    for (int x = 0; x < 30; ++x)
        for (int y = 0; y < 30; ++y)
            cloud->push_back({x * 0.02f, y * 0.02f, 0.0f});
    
    // Add outlier points (vertical line)
    for (int i = 0; i < 20; ++i)
        cloud->push_back({0.3f, 0.3f, 0.2f + i * 0.01f});

    std::cout << "Total points: " << cloud->size() << std::endl;

    // Use RANSAC to find the dominant plane
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    std::cout << "Plane inliers: " << inliers->indices.size() << std::endl;
    std::cout << "Plane coefficients (ax + by + cz + d = 0): ";
    for (float c : coefficients->values) std::cout << c << " ";
    std::cout << std::endl;

    // Separate inliers and outliers for visualization
    auto plane_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    auto outlier_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    
    std::set<int> inlier_set(inliers->indices.begin(), inliers->indices.end());
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        if (inlier_set.count(i))
            plane_cloud->push_back(cloud->points[i]);
        else
            outlier_cloud->push_back(cloud->points[i]);
    }

    // Visualize
    try {
        pcl::visualization::PCLVisualizer viewer("Plane Segmentation");
        
        // Plane inliers in green
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(plane_cloud, 0, 255, 0);
        viewer.addPointCloud<pcl::PointXYZ>(plane_cloud, green, "plane");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "plane");

        // Outliers in red
        if (!outlier_cloud->empty()) {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(outlier_cloud, 255, 0, 0);
            viewer.addPointCloud<pcl::PointXYZ>(outlier_cloud, red, "outliers");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "outliers");
        }

        viewer.addCoordinateSystem(0.1, "axes", 0);
        viewer.initCameraParameters();
        
        std::cout << "Green = plane (inliers), Red = outliers. Press 'q' to quit." << std::endl;
        
        while (!viewer.wasStopped()) {
            viewer.spinOnce(100);
        }
    } catch (const std::exception& e) {
        std::cout << "Visualization skipped (headless environment): " << e.what() << std::endl;
    }

    return 0;
}

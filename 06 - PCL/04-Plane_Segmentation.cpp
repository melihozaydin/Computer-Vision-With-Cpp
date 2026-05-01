#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <iostream>

int main() {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    for (int x = 0; x < 30; ++x)
        for (int y = 0; y < 30; ++y)
            cloud->push_back({x * 0.02f, y * 0.02f, 0.0f});
    for (int i = 0; i < 20; ++i)
        cloud->push_back({0.3f, 0.3f, 0.2f + i * 0.01f});

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices inliers;
    pcl::ModelCoefficients coefficients;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(inliers, coefficients);

    std::cout << "Plane inliers: " << inliers.indices.size() << std::endl;
    std::cout << "Plane coefficients: ";
    for (float c : coefficients.values) std::cout << c << ' ';
    std::cout << std::endl;
    return 0;
}

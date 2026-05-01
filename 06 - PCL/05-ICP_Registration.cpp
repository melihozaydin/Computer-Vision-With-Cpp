#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <iostream>

int main() {
    auto source = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    auto target = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

    for (int i = 0; i < 100; ++i) {
        float x = 0.01f * i;
        float y = 0.02f * (i % 10);
        source->push_back({x, y, 0.0f});
        target->push_back({x + 0.05f, y - 0.02f, 0.0f});
    }

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    pcl::PointCloud<pcl::PointXYZ> aligned;
    icp.align(aligned);

    std::cout << "ICP converged: " << icp.hasConverged() << std::endl;
    std::cout << "Fitness score: " << icp.getFitnessScore() << std::endl;
    std::cout << "Estimated transform:\n" << icp.getFinalTransformation() << std::endl;
    return 0;
}

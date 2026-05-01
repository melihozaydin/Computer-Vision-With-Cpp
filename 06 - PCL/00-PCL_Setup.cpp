#include <pcl/pcl_config.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>

int main() {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.push_back({0.0f, 0.0f, 0.0f});
    cloud.push_back({1.0f, 0.0f, 0.0f});
    cloud.push_back({0.0f, 1.0f, 0.0f});

    std::cout << "PCL version: " << PCL_VERSION_PRETTY << std::endl;
    std::cout << "Synthetic cloud size: " << cloud.size() << std::endl;
    return 0;
}

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>

int main() {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    for (int x = 0; x < 50; ++x)
        for (int y = 0; y < 50; ++y)
            for (int z = 0; z < 4; ++z)
                cloud->push_back({x * 0.01f, y * 0.01f, z * 0.01f});

    auto filtered = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);
    vg.filter(*filtered);

    std::cout << "Original points: " << cloud->size() << std::endl;
    std::cout << "Downsampled points: " << filtered->size() << std::endl;
    return 0;
}

#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>

int main() {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    for (int x = 0; x < 25; ++x)
        for (int y = 0; y < 25; ++y)
            cloud->push_back({x * 0.02f, y * 0.02f, 0.01f * (x + y)});

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    auto tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setKSearch(15);

    pcl::PointCloud<pcl::Normal> normals;
    ne.compute(normals);

    std::cout << "Computed normals: " << normals.size() << std::endl;
    if (!normals.empty()) {
        std::cout << "First normal: (" << normals[0].normal_x << ", "
                  << normals[0].normal_y << ", " << normals[0].normal_z << ")" << std::endl;
    }
    return 0;
}

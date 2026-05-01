#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>

int main() {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width = 200;
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.points.resize(cloud.width * cloud.height);

    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        float x = static_cast<float>(i) * 0.01f;
        cloud.points[i].x = x;
        cloud.points[i].y = 0.5f * x;
        cloud.points[i].z = 0.1f * std::sin(x * 6.28f);
    }

    pcl::io::savePCDFileASCII("synthetic_line_cloud.pcd", cloud);
    std::cout << "Saved synthetic_line_cloud.pcd with " << cloud.size() << " points" << std::endl;
    return 0;
}

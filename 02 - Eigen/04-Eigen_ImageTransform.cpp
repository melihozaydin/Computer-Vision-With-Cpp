// 04-Eigen_ImageTransform.cpp
// Using Eigen for basic image-like transformations (rotation, scaling)
// Compile with: g++ 04-Eigen_ImageTransform.cpp -I /path/to/eigen -o 04-Eigen_ImageTransform

#include <iostream>
#include <Eigen/Dense>
#include <cmath>

int main() {
    // Simulate a 2D grayscale image as a matrix
    Eigen::Matrix3f image;
    image << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;
    std::cout << "Original image (matrix):\n" << image << std::endl;

    // Scaling (multiply by scalar)
    Eigen::Matrix3f scaled = 2.0f * image;
    std::cout << "\nScaled image (x2):\n" << scaled << std::endl;

    // Rotation by 90 degrees (counterclockwise)
    Eigen::Matrix3f rotated = image.transpose().colwise().reverse();
    std::cout << "\nRotated image (90 deg CCW):\n" << rotated << std::endl;

    // Flipping (vertical)
    Eigen::Matrix3f flipped = image.colwise().reverse();
    std::cout << "\nFlipped image (vertical):\n" << flipped << std::endl;

    return 0;
}

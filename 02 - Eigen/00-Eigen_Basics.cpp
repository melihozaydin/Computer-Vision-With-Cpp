// 00-Eigen_Basics.cpp
// Introduction to Eigen: Matrix and Vector Basics
// Compile with: g++ 00-Eigen_Basics.cpp -I /path/to/eigen -o 00-Eigen_Basics

#include <iostream>
#include <Eigen/Dense>

int main() {
    // 1. Creating matrices and vectors
    Eigen::Matrix3f mat3x3; // 3x3 float matrix
    mat3x3 << 1, 2, 3,
               4, 5, 6,
               7, 8, 9;
    std::cout << "Matrix 3x3:\n" << mat3x3 << std::endl;

    Eigen::Vector3f vec3; // 3D float vector
    vec3 << 1, 2, 3;
    std::cout << "\nVector 3:\n" << vec3 << std::endl;

    // 2. Identity, Zero, and Random matrices
    Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();
    std::cout << "\nIdentity Matrix:\n" << identity << std::endl;

    Eigen::Matrix3f zeros = Eigen::Matrix3f::Zero();
    std::cout << "\nZero Matrix:\n" << zeros << std::endl;

    Eigen::Matrix3f random = Eigen::Matrix3f::Random();
    std::cout << "\nRandom Matrix:\n" << random << std::endl;

    // 3. Matrix transpose
    std::cout << "\nTranspose of mat3x3:\n" << mat3x3.transpose() << std::endl;

    // 4. Matrix and vector sizes
    std::cout << "\nmat3x3 rows: " << mat3x3.rows() << ", cols: " << mat3x3.cols() << std::endl;
    std::cout << "vec3 size: " << vec3.size() << std::endl;

    return 0;
}

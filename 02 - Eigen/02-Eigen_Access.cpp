// 02-Eigen_Access.cpp
// Accessing and modifying elements, block operations
// Compile with: g++ 02-Eigen_Access.cpp -I /path/to/eigen -o 02-Eigen_Access

#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix3i mat;
    mat << 1, 2, 3,
           4, 5, 6,
           7, 8, 9;
    std::cout << "Original matrix:\n" << mat << std::endl;

    // Access and modify an element
    mat(0, 1) = 20;
    std::cout << "\nAfter modifying (0,1):\n" << mat << std::endl;

    // Block access (top-left 2x2)
    Eigen::Matrix2i block = mat.block<2,2>(0,0);
    std::cout << "\nTop-left 2x2 block:\n" << block << std::endl;

    // Row and column access
    std::cout << "\nFirst row: " << mat.row(0) << std::endl;
    std::cout << "First column: " << mat.col(0) << std::endl;

    // Set entire row/column
    mat.row(1) = Eigen::Vector3i::Constant(99);
    std::cout << "\nAfter setting row 1 to 99:\n" << mat << std::endl;

    return 0;
}

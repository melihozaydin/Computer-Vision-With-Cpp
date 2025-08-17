// 01-Eigen_Arithmetic.cpp
// Matrix arithmetic: addition, subtraction, multiplication
// Compile with: g++ 01-Eigen_Arithmetic.cpp -I /path/to/eigen -o 01-Eigen_Arithmetic

#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d A;
    A << 1, 2,
         3, 4;
    Eigen::Matrix2d B;
    B << 5, 6,
         7, 8;

    std::cout << "A:\n" << A << std::endl;
    std::cout << "B:\n" << B << std::endl;

    // Addition
    std::cout << "\nA + B:\n" << A + B << std::endl;
    // Subtraction
    std::cout << "\nA - B:\n" << A - B << std::endl;
    // Matrix multiplication
    std::cout << "\nA * B:\n" << A * B << std::endl;
    // Scalar multiplication
    std::cout << "\n2 * A:\n" << 2 * A << std::endl;

    return 0;
}

// 03-Eigen_Advanced.cpp
// Eigenvalues, eigenvectors, SVD, and decompositions
// Compile with: g++ 03-Eigen_Advanced.cpp -I /path/to/eigen -o 03-Eigen_Advanced

#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d A;
    A << 2, -1,
         -1, 2;
    std::cout << "Matrix A:\n" << A << std::endl;

    // Eigenvalues and eigenvectors
    Eigen::EigenSolver<Eigen::Matrix2d> es(A);
    std::cout << "\nEigenvalues:\n" << es.eigenvalues() << std::endl;
    std::cout << "Eigenvectors:\n" << es.eigenvectors() << std::endl;

    // Singular Value Decomposition (SVD)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    std::cout << "\nSingular values:\n" << svd.singularValues() << std::endl;
    std::cout << "U matrix:\n" << svd.matrixU() << std::endl;
    std::cout << "V matrix:\n" << svd.matrixV() << std::endl;

    // LU decomposition
    Eigen::FullPivLU<Eigen::Matrix2d> lu(A);
    std::cout << "\nDeterminant: " << lu.determinant() << std::endl;
    std::cout << "Rank: " << lu.rank() << std::endl;

    return 0;
}

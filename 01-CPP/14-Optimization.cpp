#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>

/*
This script covers:

Avoiding Unnecessary Memory Allocations: Demonstrates how to use reserve to avoid multiple reallocations in a std::vector.
Move Semantics: Shows how to use move constructors and move assignment operators to avoid unnecessary copying of large objects.
Loop Optimization: Demonstrates how to use iterators instead of indexing for better performance.
Using Appropriate Data Structures: Shows how to use std::vector for dynamic arrays and std::unordered_map for fast lookups.
*/

// Function to demonstrate avoiding unnecessary memory allocations
void avoidUnnecessaryAllocations() {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int> vec;
    vec.reserve(100); // Reserve memory to avoid multiple reallocations

    for (int i = 0; i < 100; ++i) {
        vec.push_back(i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Avoid Unnecessary Allocations - Duration: " << duration.count() << " seconds\n";
}

// Function to demonstrate move semantics
class LargeObject {
public:
    LargeObject() {
        data = new int[1000];
        std::cout << "LargeObject created\n";
    }

    ~LargeObject() {
        delete[] data;
        std::cout << "LargeObject destroyed\n";
    }

    // Move constructor
    LargeObject(LargeObject&& other) noexcept : data(other.data) {
        other.data = nullptr;
        std::cout << "LargeObject moved\n";
    }

    // Move assignment operator
    LargeObject& operator=(LargeObject&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            other.data = nullptr;
            std::cout << "LargeObject move-assigned\n";
        }
        return *this;
    }

private:
    int* data;
};

void demonstrateMoveSemantics() {
    auto start = std::chrono::high_resolution_clock::now();

    LargeObject obj1;
    LargeObject obj2 = std::move(obj1); // Move constructor
    LargeObject obj3;
    obj3 = std::move(obj2); // Move assignment operator

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Demonstrate Move Semantics - Duration: " << duration.count() << " seconds\n";
}

// Function to demonstrate loop optimization
void loopOptimization() {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int> vec(1000);
    std::iota(vec.begin(), vec.end(), 0); // Fill vector with 0 to 999

    // Use iterators instead of indexing
    int sum = 0;
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        sum += *it;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Loop Optimization - Duration: " << duration.count() << " seconds\n";
    std::cout << "Sum of vector elements: " << sum << "\n";
}

// Function to demonstrate using appropriate data structures
void useAppropriateDataStructures() {
    auto start = std::chrono::high_resolution_clock::now();

    // Use std::vector instead of raw arrays for dynamic arrays
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Use std::unordered_map for fast lookups
    std::unordered_map<int, std::string> map = {
        {1, "one"},
        {2, "two"},
        {3, "three"}
    };

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Use Appropriate Data Structures - Duration: " << duration.count() << " seconds\n";
    std::cout << "Value for key 2: " << map[2] << "\n";
}

// Function to generate a random matrix
std::vector<std::vector<int>> generateRandomMatrix(int rows, int cols) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

// Function to multiply two matrices
//Matrix Multiplication - Duration: 0.903692 seconds
std::vector<std::vector<int>> multiplyMatrices_old(const std::vector<std::vector<int>>& mat1, const std::vector<std::vector<int>>& mat2) {
    int rows = mat1.size();
    int cols = mat2[0].size();
    int commonDim = mat2.size();
    std::vector<std::vector<int>> result(rows, std::vector<int>(cols, 0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < commonDim; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return result;
}


#include <omp.h>
//Matrix Multiplication New - Duration: 0.638036 seconds
std::vector<std::vector<int>> multiplyMatrices_new(const std::vector<std::vector<int>>& mat1, const std::vector<std::vector<int>>& mat2) {
    if (mat2.empty() || mat2[0].empty()) {
        throw std::invalid_argument("mat2 must not be empty");
    }
    int rows = mat1.size();
    int cols = mat2[0].size();
    int commonDim = mat2.size();
    std::vector<std::vector<int>> result(rows, std::vector<int>(cols, 0));
    
    // Define the unrolling factor
    const int unrollFactor = 4;

    // this loop is optimized for cache locality
    // Also the inner loop is unrolled by the unrollFactor which is a technique to reduce the overhead of loop control tradeoff with code size.
    // Pragma "omp parallel for" OpenMP is used to parallelize the outer loop
    // which tells the compiler to distribute the iterations of the outer loop among the available threads.
    // This can significantly improve performance for large matrices.
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int sum = 0;
            for (int k = 0; k < commonDim; k += unrollFactor) {
                for (int u = 0; u < unrollFactor && k + u < commonDim; ++u) {
                    sum += mat1[i][k + u] * mat2[k + u][j];
                }
            }
            result[i][j] = sum;
        }
    }

    return result;
}

// Function to demonstrate matrix multiplication
void matrixMultiplicationExample() {
    auto mat1 = generateRandomMatrix(10000, 10);
    auto mat2 = generateRandomMatrix(11, 1000);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = multiplyMatrices_old(mat1, mat2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Matrix Multiplication Old - Duration: " << duration.count() << " seconds\n";

    // Memory usage and copy count for old method
    size_t memUsageOld = sizeof(int) * mat1.size() * mat1[0].size() + sizeof(int) * mat2.size() * mat2[0].size() + sizeof(int) * result.size() * result[0].size();
    size_t memCopiesOld = mat1.size() * mat2[0].size() * mat2.size();
    std::cout << "Old Method - Memory Usage: " << memUsageOld << " bytes, Memory Copies: " << memCopiesOld << "\n";

    auto start1 = std::chrono::high_resolution_clock::now();
    auto result1 = multiplyMatrices_new(mat1, mat2);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end1 - start1;
    std::cout << "Matrix Multiplication New - Duration: " << duration1.count() << " seconds\n";

    // Memory usage and copy count for new method
    size_t memUsageNew = sizeof(int) * mat1.size() * mat1[0].size() + sizeof(int) * mat2.size() * mat2[0].size() + sizeof(int) * result1.size() * result1[0].size();
    size_t memCopiesNew = mat1.size() * mat2[0].size() * (mat2.size() / 4);
    std::cout << "New Method - Memory Usage: " << memUsageNew << " bytes, Memory Copies: " << memCopiesNew << "\n";
}

int main() {
    std::cout << "Avoid Unnecessary Allocations:\n";
    avoidUnnecessaryAllocations();

    std::cout << "\nDemonstrate Move Semantics:\n";
    demonstrateMoveSemantics();

    std::cout << "\nLoop Optimization:\n";
    loopOptimization();

    std::cout << "\nUse Appropriate Data Structures:\n";
    useAppropriateDataStructures();

    std::cout << "\nMatrix Multiplication Example:\n";
    matrixMultiplicationExample();

    return 0;
}
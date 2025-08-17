/**
 * 01-CUDA_Basic_Operations.cpp
 * 
 * Basic GPU operations and memory management.
 * 
 * Concepts covered:
 * - GPU memory allocation\n * - Host-device transfers\n * - Basic arithmetic operations\n * - Memory streams\n * - Synchronization
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CUDA Basic Operations ===" << std::endl;
    try {
        // Allocate host memory
        int rows = 512, cols = 512;
        cv::Mat h_A(rows, cols, CV_32F, cv::Scalar(2.0f));
        cv::Mat h_B(rows, cols, CV_32F, cv::Scalar(3.0f));
        cv::Mat h_C;

        // Allocate device memory (GpuMat)
        cv::cuda::GpuMat d_A, d_B, d_C;

        // Upload data to device
        d_A.upload(h_A);
        d_B.upload(h_B);

        // Basic arithmetic: C = A + B
        cv::cuda::add(d_A, d_B, d_C);

        // Download result
        d_C.download(h_C);
        std::cout << "Sample result (A+B)[0,0]: " << h_C.at<float>(0,0) << std::endl;

        // Use CUDA stream for async operations
        cv::cuda::Stream stream;
        d_A.setTo(cv::Scalar(5.0f), stream);
        d_B.setTo(cv::Scalar(7.0f), stream);
        cv::cuda::add(d_A, d_B, d_C, cv::noArray(), -1, stream);
        stream.waitForCompletion();
        d_C.download(h_C);
        std::cout << "Sample result (A+B, stream)[0,0]: " << h_C.at<float>(0,0) << std::endl;

        // Synchronization
        stream.waitForCompletion();
        std::cout << "CUDA stream operations complete." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA Error: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "\n✓ CUDA Basic Operations demonstration complete!" << std::endl;
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

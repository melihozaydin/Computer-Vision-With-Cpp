/**
 * 09-CUDA_Memory_Management.cpp
 * 
 * Efficient GPU memory usage patterns.
 * 
 * Concepts covered:
 * - Memory allocation strategies\n * - Pinned memory\n * - Memory pools\n * - Stream management\n * - Profiling memory usage
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CUDA Memory Management ===" << std::endl;
    try {
        // Pinned memory allocation
        int rows = 256, cols = 256;
        cv::Mat h_pinned;
        cv::cuda::HostMem pinned(rows, cols, CV_32F, cv::cuda::HostMem::PAGE_LOCKED);
        h_pinned = pinned.createMatHeader();
        h_pinned.setTo(2.0f);

        // Upload to GPU
        cv::cuda::GpuMat d_mat;
        d_mat.upload(h_pinned);

        // Memory pool usage
        cv::cuda::setBufferPoolUsage(true);
        cv::cuda::GpuMat d_pool1(rows, cols, CV_32F);
        d_pool1.setTo(1.0f);
        cv::cuda::GpuMat d_pool2(rows, cols, CV_32F);
        d_pool2.setTo(3.0f);

        // Stream management
        cv::cuda::Stream stream;
        d_pool1.setTo(5.0f, stream);
        stream.waitForCompletion();

        std::cout << "Pinned memory sample: " << h_pinned.at<float>(0,0) << std::endl;
        std::cout << "Buffer pool and stream operations complete." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA Error: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "\n✓ CUDA Memory Management demonstration complete!" << std::endl;
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

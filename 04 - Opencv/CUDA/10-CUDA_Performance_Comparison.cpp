/**
 * 10-CUDA_Performance_Comparison.cpp
 * 
 * Comprehensive CPU vs GPU performance analysis.
 * 
 * Concepts covered:
 * - Benchmarking framework\n * - Various operation comparisons\n * - Memory transfer overhead\n * - Scalability analysis\n * - Optimization recommendations
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CUDA Performance Comparison ===" << std::endl;
    try {
        int rows = 1024, cols = 1024;
        cv::Mat h_A(rows, cols, CV_32F, cv::Scalar(2.0f));
        cv::Mat h_B(rows, cols, CV_32F, cv::Scalar(3.0f));
        cv::Mat h_C, h_C_cpu;

        // GPU multiply
        cv::cuda::GpuMat d_A, d_B, d_C;
        d_A.upload(h_A);
        d_B.upload(h_B);
        auto start = cv::getTickCount();
        cv::cuda::multiply(d_A, d_B, d_C);
        double gpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        d_C.download(h_C);

        // CPU multiply
        start = cv::getTickCount();
        cv::multiply(h_A, h_B, h_C_cpu);
        double cpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

        std::cout << "Matrix multiply (" << rows << "x" << cols << "):\n  GPU: " << gpu_time << " ms\n  CPU: " << cpu_time << " ms\n";

        // Gaussian blur comparison
        cv::Mat img(1024, 1024, CV_8UC1);
        cv::randu(img, 0, 255);
        cv::cuda::GpuMat d_img, d_blur;
        d_img.upload(img);
        auto gauss = cv::cuda::createGaussianFilter(d_img.type(), -1, cv::Size(15,15), 5.0);
        start = cv::getTickCount();
        gauss->apply(d_img, d_blur);
        double gpu_blur = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        cv::Mat cpu_blur;
        start = cv::getTickCount();
        cv::GaussianBlur(img, cpu_blur, cv::Size(15,15), 5.0);
        double cpu_blur_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        std::cout << "Gaussian blur (15x15):\n  GPU: " << gpu_blur << " ms\n  CPU: " << cpu_blur_time << " ms\n";
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA Error: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "\n✓ CUDA Performance Comparison demonstration complete!" << std::endl;
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

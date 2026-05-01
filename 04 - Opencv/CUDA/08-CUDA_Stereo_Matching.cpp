/**
 * 08-CUDA_Stereo_Matching.cpp
 * 
 * Stereo vision algorithms on GPU.
 * 
 * Concepts covered:
 * - GPU stereo block matching\n * - Disparity computation\n * - Real-time depth estimation\n * - Memory optimization\n * - Quality improvement
 */

#include <opencv2/opencv.hpp>
#if __has_include(<opencv2/cudastereo.hpp>)
#include <opencv2/cudastereo.hpp>
#define HAS_OPENCV_CUDA_STEREO 1
#else
#define HAS_OPENCV_CUDA_STEREO 0
#endif
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CUDA Stereo Matching ===" << std::endl;
    try {
        // Create synthetic stereo pair (shifted rectangles)
        int rows = 128, cols = 256;
        cv::Mat left(rows, cols, CV_8UC1, cv::Scalar(0));
        cv::Mat right(rows, cols, CV_8UC1, cv::Scalar(0));
        cv::rectangle(left, {60,40}, {120,100}, cv::Scalar(255), -1);
        cv::rectangle(right, {50,40}, {110,100}, cv::Scalar(255), -1); // shift by 10 px

    #if HAS_OPENCV_CUDA_STEREO
        // Upload to GPU
        cv::cuda::GpuMat d_left, d_right, d_disp;
        d_left.upload(left);
        d_right.upload(right);

        // StereoBM (GPU)
        auto stereo = cv::cuda::createStereoBM(64, 9);
        auto start = cv::getTickCount();
        stereo->compute(d_left, d_right, d_disp);
        double gpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

        // Download and print disparity stats
        cv::Mat disp;
        d_disp.download(disp);
        double minVal, maxVal;
        cv::minMaxLoc(disp, &minVal, &maxVal);
        std::cout << "StereoBM (GPU): " << gpu_time << " ms, disparity range: [" << minVal << ", " << maxVal << "]" << std::endl;
    #else
        std::cout << "CUDA stereo module not available in this OpenCV build." << std::endl;
        std::cout << "Falling back to CPU StereoBM for demonstration." << std::endl;
        auto stereo = cv::StereoBM::create(64, 9);
        cv::Mat disp;
        auto start = cv::getTickCount();
        stereo->compute(left, right, disp);
        double cpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        double minVal, maxVal;
        cv::minMaxLoc(disp, &minVal, &maxVal);
        std::cout << "StereoBM (CPU): " << cpu_time << " ms, disparity range: [" << minVal << ", " << maxVal << "]" << std::endl;
    #endif
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA Error: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "\n✓ CUDA Stereo Matching demonstration complete!" << std::endl;
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

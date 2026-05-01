/**
 * 07-CUDA_Background_Subtraction.cpp
 * 
 * Background modeling using GPU acceleration.
 * 
 * Concepts covered:
 * - GPU MOG2 background subtraction\n * - Real-time processing\n * - Memory optimization\n * - Parameter tuning\n * - Performance analysis
 */

#include <opencv2/opencv.hpp>
#if __has_include(<opencv2/cudabgsegm.hpp>)
#include <opencv2/cudabgsegm.hpp>
#define HAS_OPENCV_CUDA_BGSEGM 1
#else
#define HAS_OPENCV_CUDA_BGSEGM 0
#endif
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CUDA Background Subtraction ===" << std::endl;
    try {
        // Create synthetic video frames (moving square)
        int rows = 240, cols = 320;
#if HAS_OPENCV_CUDA_BGSEGM
        cv::cuda::GpuMat d_frame, d_mask;
        auto mog2 = cv::cuda::createBackgroundSubtractorMOG2();
        std::cout << "Using CUDA MOG2 background subtractor." << std::endl;
#else
        auto mog2 = cv::createBackgroundSubtractorMOG2();
        std::cout << "CUDA background subtraction module not available in this OpenCV build." << std::endl;
        std::cout << "Falling back to CPU MOG2 for demonstration." << std::endl;
#endif
        for (int i = 0; i < 10; ++i) {
            cv::Mat frame(rows, cols, CV_8UC3, cv::Scalar(0,0,0));
            cv::rectangle(frame, {20+i*10, 60}, {80+i*10, 120}, cv::Scalar(255,255,255), -1);

#if HAS_OPENCV_CUDA_BGSEGM
            d_frame.upload(frame);
            auto start = cv::getTickCount();
            mog2->apply(d_frame, d_mask);
            double gpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
            cv::Mat mask;
            d_mask.download(mask);
            double fg_pixels = cv::countNonZero(mask);
            std::cout << "Frame " << i << ": FG pixels = " << fg_pixels << ", GPU time = " << gpu_time << " ms" << std::endl;
#else
            cv::Mat mask;
            auto start = cv::getTickCount();
            mog2->apply(frame, mask);
            double cpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
            double fg_pixels = cv::countNonZero(mask);
            std::cout << "Frame " << i << ": FG pixels = " << fg_pixels << ", CPU time = " << cpu_time << " ms" << std::endl;
#endif
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA Error: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "\n✓ CUDA Background Subtraction demonstration complete!" << std::endl;
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

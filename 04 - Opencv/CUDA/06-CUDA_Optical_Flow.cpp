/**
 * 06-CUDA_Optical_Flow.cpp
 * 
 * Optical flow computation on GPU.
 * 
 * Concepts covered:
 * - GPU Lucas-Kanade flow\n * - Farneback dense flow\n * - Real-time motion tracking\n * - Flow field processing\n * - Memory management
 */

#include <opencv2/opencv.hpp>
#if __has_include(<opencv2/cudaoptflow.hpp>)
#include <opencv2/cudaoptflow.hpp>
#define HAS_OPENCV_CUDA_OPTFLOW 1
#else
#define HAS_OPENCV_CUDA_OPTFLOW 0
#endif
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CUDA Optical Flow ===" << std::endl;
    try {
        // Create two synthetic images (shifted squares)
        cv::Mat img1(256, 256, CV_8UC1, cv::Scalar(0));
        cv::rectangle(img1, {60,60}, {120,120}, cv::Scalar(255), -1);
        cv::Mat img2 = img1.clone();
        cv::Mat roi = img2(cv::Rect(65,65,60,60));
        img1(cv::Rect(60,60,60,60)).copyTo(roi);

    #if HAS_OPENCV_CUDA_OPTFLOW
        // Upload to GPU
        cv::cuda::GpuMat d_img1, d_img2;
        d_img1.upload(img1);
        d_img2.upload(img2);

        // Farneback dense optical flow (GPU)
        auto farneback = cv::cuda::FarnebackOpticalFlow::create();
        cv::cuda::GpuMat d_flow;
        auto start = cv::getTickCount();
        farneback->calc(d_img1, d_img2, d_flow);
        double gpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

        // Download and print flow stats
        cv::Mat flow;
        d_flow.download(flow);
        cv::Scalar mean_flow = cv::mean(flow);
        std::cout << "Farneback (GPU): " << gpu_time << " ms, mean flow: (" << mean_flow[0] << ", " << mean_flow[1] << ")" << std::endl;
    #else
        std::cout << "CUDA optical flow module not available in this OpenCV build." << std::endl;
        std::cout << "Falling back to CPU Farneback for demonstration." << std::endl;
        cv::Mat flow;
        auto start = cv::getTickCount();
        cv::calcOpticalFlowFarneback(img1, img2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        double cpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        cv::Scalar mean_flow = cv::mean(flow);
        std::cout << "Farneback (CPU): " << cpu_time << " ms, mean flow: (" << mean_flow[0] << ", " << mean_flow[1] << ")" << std::endl;
    #endif
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA Error: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "\n✓ CUDA Optical Flow demonstration complete!" << std::endl;
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

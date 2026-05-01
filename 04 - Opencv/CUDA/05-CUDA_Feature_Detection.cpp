/**
 * 05-CUDA_Feature_Detection.cpp
 * 
 * Feature detection using GPU acceleration.
 * 
 * Concepts covered:
 * - GPU ORB features\n * - FAST corner detection\n * - Feature descriptors\n * - Keypoint filtering\n * - Parallel feature extraction
 */

#include <opencv2/opencv.hpp>
#if __has_include(<opencv2/cudafeatures2d.hpp>)
#include <opencv2/cudafeatures2d.hpp>
#define HAS_OPENCV_CUDA_FEATURES2D 1
#else
#define HAS_OPENCV_CUDA_FEATURES2D 0
#endif
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CUDA Feature Detection ===" << std::endl;
    try {
        // Create synthetic image
        cv::Mat img(512, 512, CV_8UC1);
        cv::randu(img, 0, 255);

    #if HAS_OPENCV_CUDA_FEATURES2D
        cv::cuda::GpuMat d_img;
        d_img.upload(img);

        // ORB (GPU)
        auto orb = cv::cuda::ORB::create(500);
        std::vector<cv::KeyPoint> keypoints_orb;
        cv::cuda::GpuMat keypoints_orb_gpu, descriptors_orb;
        auto start = cv::getTickCount();
        orb->detectAndComputeAsync(d_img, cv::noArray(), keypoints_orb_gpu, descriptors_orb);
        orb->convert(keypoints_orb_gpu, keypoints_orb);
        double orb_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        std::cout << "ORB (GPU): " << keypoints_orb.size() << " keypoints, " << orb_time << " ms" << std::endl;

        // FAST (GPU)
        auto fast = cv::cuda::FastFeatureDetector::create(20);
        std::vector<cv::KeyPoint> keypoints_fast;
        cv::cuda::GpuMat keypoints_fast_gpu;
        start = cv::getTickCount();
        fast->detectAsync(d_img, keypoints_fast_gpu);
        fast->convert(keypoints_fast_gpu, keypoints_fast);
        double fast_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        std::cout << "FAST (GPU): " << keypoints_fast.size() << " keypoints, " << fast_time << " ms" << std::endl;
    #else
        std::cout << "CUDA features2d module not available in this OpenCV build." << std::endl;
        std::cout << "Falling back to CPU ORB/FAST for demonstration." << std::endl;

        auto start = cv::getTickCount();
        auto orb = cv::ORB::create(500);
        std::vector<cv::KeyPoint> keypoints_orb;
        cv::Mat descriptors_orb;
        orb->detectAndCompute(img, cv::noArray(), keypoints_orb, descriptors_orb);
        double orb_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        std::cout << "ORB (CPU): " << keypoints_orb.size() << " keypoints, " << orb_time << " ms" << std::endl;

        start = cv::getTickCount();
        auto fast = cv::FastFeatureDetector::create(20);
        std::vector<cv::KeyPoint> keypoints_fast;
        fast->detect(img, keypoints_fast);
        double fast_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        std::cout << "FAST (CPU): " << keypoints_fast.size() << " keypoints, " << fast_time << " ms" << std::endl;
    #endif
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA Error: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "\n✓ CUDA Feature Detection demonstration complete!" << std::endl;
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

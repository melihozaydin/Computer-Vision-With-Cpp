/**
 * 05-CUDA_Feature_Detection.cpp
 * 
 * Feature detection using GPU acceleration.
 * 
 * Concepts covered:
 * - GPU ORB features\n * - FAST corner detection\n * - Feature descriptors\n * - Keypoint filtering\n * - Parallel feature extraction
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CUDA Feature Detection ===" << std::endl;
    try {
        // Create synthetic image
        cv::Mat img(512, 512, CV_8UC1);
        cv::randu(img, 0, 255);
        cv::cuda::GpuMat d_img;
        d_img.upload(img);

        // ORB (GPU)
        auto orb = cv::cuda::ORB::create(500);
        std::vector<cv::KeyPoint> keypoints_orb;
        cv::cuda::GpuMat descriptors_orb;
        auto start = cv::getTickCount();
        orb->detectAndComputeAsync(d_img, cv::noArray(), keypoints_orb, descriptors_orb);
        orb->convert(keypoints_orb, keypoints_orb);
        double orb_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        std::cout << "ORB (GPU): " << keypoints_orb.size() << " keypoints, " << orb_time << " ms" << std::endl;

        // FAST (GPU)
        auto fast = cv::cuda::FastFeatureDetector::create(20);
        std::vector<cv::KeyPoint> keypoints_fast;
        start = cv::getTickCount();
        fast->detectAsync(d_img, keypoints_fast);
        fast->convert(keypoints_fast, keypoints_fast);
        double fast_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        std::cout << "FAST (GPU): " << keypoints_fast.size() << " keypoints, " << fast_time << " ms" << std::endl;
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

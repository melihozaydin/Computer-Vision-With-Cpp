/**
 * 02-CUDA_Filtering.cpp
 * 
 * GPU-accelerated image filtering operations.
 * 
 * Concepts covered:
 * - Gaussian filtering on GPU\n * - Convolution operations\n * - Separable filters\n * - Custom kernel filters\n * - Performance comparison
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CUDA Filtering ===" << std::endl;
    try {
        // Create synthetic image
        cv::Mat img(512, 512, CV_8UC1);
        cv::randu(img, 0, 255);

        // Upload to GPU
        cv::cuda::GpuMat d_img, d_blur, d_custom;
        d_img.upload(img);

        // Gaussian blur (GPU)
        auto start = cv::getTickCount();
        cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(d_img.type(), -1, cv::Size(11,11), 2.0);
        gauss->apply(d_img, d_blur);
        double gpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

        // Gaussian blur (CPU)
        cv::Mat cpu_blur;
        start = cv::getTickCount();
        cv::GaussianBlur(img, cpu_blur, cv::Size(11,11), 2.0);
        double cpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

        std::cout << "Gaussian blur (11x11, sigma=2):\n  GPU: " << gpu_time << " ms\n  CPU: " << cpu_time << " ms\n";

        // Custom kernel (sharpen)
        cv::Mat kernel = (cv::Mat_<float>(3,3) << 0,-1,0, -1,5,-1, 0,-1,0);
        cv::Ptr<cv::cuda::Filter> custom = cv::cuda::createLinearFilter(d_img.type(), -1, kernel);
        custom->apply(d_img, d_custom);

        // Download and show sample values
        cv::Mat blur, custom_out;
        d_blur.download(blur);
        d_custom.download(custom_out);
        std::cout << "Sample blurred pixel: " << (int)blur.at<uchar>(0,0) << std::endl;
        std::cout << "Sample sharpened pixel: " << (int)custom_out.at<uchar>(0,0) << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA Error: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "\n✓ CUDA Filtering demonstration complete!" << std::endl;
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

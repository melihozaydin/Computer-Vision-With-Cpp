/**
 * 02-CUDA_Filtering.cpp
 * 
 * GPU-accelerated image filtering operations.
 * 
 * Concepts covered:
 * - Gaussian filtering on GPU\n * - Convolution operations\n * - Separable filters\n * - Custom kernel filters\n * - Performance comparison
 */

#include <opencv2/opencv.hpp>
#if __has_include(<opencv2/cudafilters.hpp>)
#include <opencv2/cudafilters.hpp>
#define HAS_OPENCV_CUDA_FILTERS 1
#else
#define HAS_OPENCV_CUDA_FILTERS 0
#endif
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

        auto start = cv::getTickCount();
        double gpu_time = 0.0;

    #if HAS_OPENCV_CUDA_FILTERS
        // Gaussian blur (GPU)
        cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(d_img.type(), -1, cv::Size(11,11), 2.0);
        gauss->apply(d_img, d_blur);
        gpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
    #else
        std::cout << "CUDA filters module not available in this OpenCV build." << std::endl;
        std::cout << "Falling back to CPU filtering for demonstration." << std::endl;
        cv::Mat tmp;
        cv::GaussianBlur(img, tmp, cv::Size(11,11), 2.0);
        gpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
    #endif

        // Gaussian blur (CPU)
        cv::Mat cpu_blur;
        start = cv::getTickCount();
        cv::GaussianBlur(img, cpu_blur, cv::Size(11,11), 2.0);
        double cpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

        std::cout << "Gaussian blur (11x11, sigma=2):\n  GPU: " << gpu_time << " ms\n  CPU: " << cpu_time << " ms\n";

        // Custom kernel (sharpen)
        cv::Mat kernel = (cv::Mat_<float>(3,3) << 0,-1,0, -1,5,-1, 0,-1,0);
        cv::Mat blur, custom_out;

    #if HAS_OPENCV_CUDA_FILTERS
        cv::Ptr<cv::cuda::Filter> custom = cv::cuda::createLinearFilter(d_img.type(), -1, kernel);
        custom->apply(d_img, d_custom);
        // Download and show sample values
        d_blur.download(blur);
        d_custom.download(custom_out);
    #else
        cv::GaussianBlur(img, blur, cv::Size(11,11), 2.0);
        cv::filter2D(img, custom_out, -1, kernel);
    #endif

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

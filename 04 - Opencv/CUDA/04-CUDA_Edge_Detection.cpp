/**
 * 04-CUDA_Edge_Detection.cpp
 * 
 * Edge detection algorithms on GPU.
 * 
 * Concepts covered:
 * - GPU Canny edge detection\n * - Sobel filters on GPU\n * - Gradient computation\n * - Memory optimization\n * - Real-time edge detection
 */

#include <opencv2/opencv.hpp>
#if __has_include(<opencv2/cudafilters.hpp>)
#include <opencv2/cudafilters.hpp>
#define HAS_OPENCV_CUDA_FILTERS 1
#else
#define HAS_OPENCV_CUDA_FILTERS 0
#endif
#if __has_include(<opencv2/cudaimgproc.hpp>)
#include <opencv2/cudaimgproc.hpp>
#define HAS_OPENCV_CUDA_IMGPROC 1
#else
#define HAS_OPENCV_CUDA_IMGPROC 0
#endif
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CUDA Edge Detection ===" << std::endl;
    try {
        // Create synthetic image
        cv::Mat img(256, 256, CV_8UC1);
        cv::randu(img, 0, 255);

        // Upload to GPU
        cv::cuda::GpuMat d_img, d_sobel, d_canny;
        d_img.upload(img);

        cv::Mat sobel_out, canny_out;

    #if HAS_OPENCV_CUDA_FILTERS && HAS_OPENCV_CUDA_IMGPROC
        // Sobel (GPU)
        cv::Ptr<cv::cuda::Filter> sobel = cv::cuda::createSobelFilter(d_img.type(), -1, 1, 0, 3);
        sobel->apply(d_img, d_sobel);

        // Canny (GPU)
        cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(50.0, 150.0);
        canny->detect(d_img, d_canny);

        // Download outputs
        d_sobel.download(sobel_out);
        d_canny.download(canny_out);
    #else
        std::cout << "CUDA edge-detection modules not available in this OpenCV build." << std::endl;
        std::cout << "Falling back to CPU Sobel/Canny for demonstration." << std::endl;
        cv::Sobel(img, sobel_out, CV_8U, 1, 0, 3);
        cv::Canny(img, canny_out, 50, 150);
    #endif

        std::cout << "Sobel sample [0,0]: " << (int)sobel_out.at<uchar>(0,0) << std::endl;
        std::cout << "Canny sample [0,0]: " << (int)canny_out.at<uchar>(0,0) << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA Error: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "\n✓ CUDA Edge Detection demonstration complete!" << std::endl;
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

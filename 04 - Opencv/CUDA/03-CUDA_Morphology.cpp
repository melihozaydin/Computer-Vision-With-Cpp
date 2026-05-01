/**
 * 03-CUDA_Morphology.cpp
 * 
 * Morphological operations using GPU acceleration.
 * 
 * Concepts covered:
 * - GPU erosion and dilation\n * - Opening and closing\n * - Morphological gradient\n * - Custom structuring elements\n * - Large kernel operations
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
    std::cout << "=== CUDA Morphology ===" << std::endl;
    try {
        // Create synthetic binary image
        cv::Mat img(256, 256, CV_8UC1, cv::Scalar(0));
        cv::rectangle(img, {50,50}, {200,200}, cv::Scalar(255), -1);
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));

        // Upload to GPU
        cv::cuda::GpuMat d_img, d_erode, d_dilate, d_open, d_close;
        d_img.upload(img);

        cv::Mat out;

    #if HAS_OPENCV_CUDA_FILTERS
        // Erosion
        cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, d_img.type(), element);
        erode->apply(d_img, d_erode);

        // Dilation
        cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, d_img.type(), element);
        dilate->apply(d_img, d_dilate);

        // Opening
        cv::Ptr<cv::cuda::Filter> open = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, d_img.type(), element);
        open->apply(d_img, d_open);

        // Closing
        cv::Ptr<cv::cuda::Filter> close = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, d_img.type(), element);
        close->apply(d_img, d_close);

        // Download and print sample values
        d_erode.download(out);
        std::cout << "Eroded pixel [60,60]: " << (int)out.at<uchar>(60,60) << std::endl;
        d_dilate.download(out);
        std::cout << "Dilated pixel [60,60]: " << (int)out.at<uchar>(60,60) << std::endl;
        d_open.download(out);
        std::cout << "Opened pixel [60,60]: " << (int)out.at<uchar>(60,60) << std::endl;
        d_close.download(out);
        std::cout << "Closed pixel [60,60]: " << (int)out.at<uchar>(60,60) << std::endl;
    #else
        std::cout << "CUDA morphology filters not available in this OpenCV build." << std::endl;
        std::cout << "Falling back to CPU morphology for demonstration." << std::endl;
        cv::erode(img, out, element);
        std::cout << "Eroded pixel [60,60]: " << (int)out.at<uchar>(60,60) << std::endl;
        cv::dilate(img, out, element);
        std::cout << "Dilated pixel [60,60]: " << (int)out.at<uchar>(60,60) << std::endl;
        cv::morphologyEx(img, out, cv::MORPH_OPEN, element);
        std::cout << "Opened pixel [60,60]: " << (int)out.at<uchar>(60,60) << std::endl;
        cv::morphologyEx(img, out, cv::MORPH_CLOSE, element);
        std::cout << "Closed pixel [60,60]: " << (int)out.at<uchar>(60,60) << std::endl;
    #endif
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA Error: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "\n✓ CUDA Morphology demonstration complete!" << std::endl;
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

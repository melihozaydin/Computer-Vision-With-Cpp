/**
 * 04-Image_Thresholding.cpp
 * 
 * Comprehensive thresholding techniques for image segmentation.
 * Converting grayscale images to binary for further processing.
 * 
 * Concepts covered:
 * - Simple binary thresholding
 * - Adaptive thresholding
 * - Otsu's automatic threshold selection
 * - Multiple threshold types
 * - Color image thresholding
 * - Multi-level thresholding
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat createTestImage() {
    // Create an image with varying intensities and textures
    cv::Mat image(400, 600, CV_8UC1);
    
    // Background gradient
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            image.at<uchar>(y, x) = (x + y) * 255 / (image.cols + image.rows);
        }
    }
    
    // Add some shapes with different intensities
    cv::circle(image, cv::Point(150, 100), 60, cv::Scalar(200), -1);
    cv::circle(image, cv::Point(450, 100), 60, cv::Scalar(80), -1);
    cv::rectangle(image, cv::Point(50, 200), cv::Point(200, 350), cv::Scalar(160), -1);
    cv::rectangle(image, cv::Point(400, 200), cv::Point(550, 350), cv::Scalar(40), -1);
    
    // Add some noise
    cv::Mat noise(image.size(), CV_8UC1);
    cv::randu(noise, 0, 30);
    cv::add(image, noise, image);
    
    return image;
}

void demonstrateSimpleThresholding(const cv::Mat& src) {
    std::cout << "\n=== Simple Binary Thresholding ===" << std::endl;
    
    cv::Mat thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero, thresh_tozero_inv;
    
    double threshold_value = 127;
    double max_value = 255;
    
    // Different threshold types
    cv::threshold(src, thresh_binary, threshold_value, max_value, cv::THRESH_BINARY);
    cv::threshold(src, thresh_binary_inv, threshold_value, max_value, cv::THRESH_BINARY_INV);
    cv::threshold(src, thresh_trunc, threshold_value, max_value, cv::THRESH_TRUNC);
    cv::threshold(src, thresh_tozero, threshold_value, max_value, cv::THRESH_TOZERO);
    cv::threshold(src, thresh_tozero_inv, threshold_value, max_value, cv::THRESH_TOZERO_INV);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("THRESH_BINARY", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("THRESH_BINARY_INV", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("THRESH_TRUNC", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("THRESH_TOZERO", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("THRESH_TOZERO_INV", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("THRESH_BINARY", thresh_binary);
    cv::imshow("THRESH_BINARY_INV", thresh_binary_inv);
    cv::imshow("THRESH_TRUNC", thresh_trunc);
    cv::imshow("THRESH_TOZERO", thresh_tozero);
    cv::imshow("THRESH_TOZERO_INV", thresh_tozero_inv);
    
    std::cout << "Threshold value: " << threshold_value << std::endl;
    std::cout << "BINARY: pixel > thresh ? max_val : 0" << std::endl;
    std::cout << "BINARY_INV: pixel > thresh ? 0 : max_val" << std::endl;
    std::cout << "TRUNC: pixel > thresh ? thresh : pixel" << std::endl;
    std::cout << "TOZERO: pixel > thresh ? pixel : 0" << std::endl;
    std::cout << "TOZERO_INV: pixel > thresh ? 0 : pixel" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateAdaptiveThresholding(const cv::Mat& src) {
    std::cout << "\n=== Adaptive Thresholding ===" << std::endl;
    
    // Simple global thresholding for comparison
    cv::Mat global_thresh;
    cv::threshold(src, global_thresh, 127, 255, cv::THRESH_BINARY);
    
    // Adaptive thresholding - Mean
    cv::Mat adaptive_mean;
    cv::adaptiveThreshold(src, adaptive_mean, 255, cv::ADAPTIVE_THRESH_MEAN_C, 
                         cv::THRESH_BINARY, 11, 2);
    
    // Adaptive thresholding - Gaussian
    cv::Mat adaptive_gaussian;
    cv::adaptiveThreshold(src, adaptive_gaussian, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY, 11, 2);
    
    // Different block sizes for adaptive thresholding
    cv::Mat adaptive_small, adaptive_large;
    cv::adaptiveThreshold(src, adaptive_small, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY, 5, 2);   // Small neighborhood
    cv::adaptiveThreshold(src, adaptive_large, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY, 25, 2);  // Large neighborhood
    
    // Different C values (constant subtracted from mean)
    cv::Mat adaptive_c_low, adaptive_c_high;
    cv::adaptiveThreshold(src, adaptive_c_low, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY, 11, 0);   // C = 0
    cv::adaptiveThreshold(src, adaptive_c_high, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY, 11, 10);  // C = 10
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Global Threshold", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Adaptive Mean", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Adaptive Gaussian", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Global Threshold", global_thresh);
    cv::imshow("Adaptive Mean", adaptive_mean);
    cv::imshow("Adaptive Gaussian", adaptive_gaussian);
    
    std::cout << "Adaptive thresholding adjusts threshold locally based on neighborhood." << std::endl;
    std::cout << "Press any key to see different parameters..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Show parameter effects
    cv::namedWindow("Block Size 5", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Block Size 25", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("C = 0", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("C = 10", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Block Size 5", adaptive_small);
    cv::imshow("Block Size 25", adaptive_large);
    cv::imshow("C = 0", adaptive_c_low);
    cv::imshow("C = 10", adaptive_c_high);
    
    std::cout << "Block size affects local neighborhood size." << std::endl;
    std::cout << "C parameter is subtracted from calculated threshold." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateOtsuThresholding(const cv::Mat& src) {
    std::cout << "\n=== Otsu's Automatic Thresholding ===" << std::endl;
    
    // Simple Otsu thresholding
    cv::Mat otsu_thresh;
    double otsu_threshold = cv::threshold(src, otsu_thresh, 0, 255, 
                                         cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // Otsu with Gaussian blur preprocessing
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(5, 5), 0);
    cv::Mat otsu_blurred;
    double otsu_threshold_blurred = cv::threshold(blurred, otsu_blurred, 0, 255, 
                                                 cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // Compare with manual threshold at Otsu value
    cv::Mat manual_thresh;
    cv::threshold(src, manual_thresh, otsu_threshold, 255, cv::THRESH_BINARY);
    
    // Triangle thresholding (another automatic method)
    cv::Mat triangle_thresh;
    double triangle_threshold = cv::threshold(src, triangle_thresh, 0, 255, 
                                            cv::THRESH_BINARY + cv::THRESH_TRIANGLE);
    
    // Calculate and display histogram
    std::vector<int> histogram(256, 0);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            histogram[src.at<uchar>(y, x)]++;
        }
    }
    
    // Create histogram image
    cv::Mat hist_image(400, 512, CV_8UC3, cv::Scalar(0, 0, 0));
    int max_count = *std::max_element(histogram.begin(), histogram.end());
    
    for (int i = 0; i < 256; i++) {
        int height = (histogram[i] * 350) / max_count;
        cv::line(hist_image, cv::Point(i * 2, 400), cv::Point(i * 2, 400 - height), 
                cv::Scalar(255, 255, 255), 2);
    }
    
    // Mark thresholds on histogram
    cv::line(hist_image, cv::Point(otsu_threshold * 2, 0), cv::Point(otsu_threshold * 2, 400), 
            cv::Scalar(0, 255, 0), 2);  // Green for Otsu
    cv::line(hist_image, cv::Point(triangle_threshold * 2, 0), cv::Point(triangle_threshold * 2, 400), 
            cv::Scalar(0, 0, 255), 2);  // Red for Triangle
    
    cv::putText(hist_image, "Otsu: " + std::to_string((int)otsu_threshold), 
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(hist_image, "Triangle: " + std::to_string((int)triangle_threshold), 
               cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Otsu Threshold", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Otsu with Blur", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Triangle Threshold", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Histogram with Thresholds", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Otsu Threshold", otsu_thresh);
    cv::imshow("Otsu with Blur", otsu_blurred);
    cv::imshow("Triangle Threshold", triangle_thresh);
    cv::imshow("Histogram with Thresholds", hist_image);
    
    std::cout << "Otsu threshold value: " << otsu_threshold << std::endl;
    std::cout << "Otsu with blur threshold: " << otsu_threshold_blurred << std::endl;
    std::cout << "Triangle threshold value: " << triangle_threshold << std::endl;
    std::cout << "Otsu minimizes intra-class variance (automatic optimal threshold)." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateColorThresholding() {
    std::cout << "\n=== Color Image Thresholding ===" << std::endl;
    
    // Create a color test image
    cv::Mat color_image(300, 400, CV_8UC3);
    
    // Different colored regions
    cv::rectangle(color_image, cv::Point(0, 0), cv::Point(200, 150), cv::Scalar(100, 50, 200), -1);      // Red-ish
    cv::rectangle(color_image, cv::Point(200, 0), cv::Point(400, 150), cv::Scalar(50, 200, 100), -1);    // Green-ish
    cv::rectangle(color_image, cv::Point(0, 150), cv::Point(200, 300), cv::Scalar(200, 100, 50), -1);    // Blue-ish
    cv::rectangle(color_image, cv::Point(200, 150), cv::Point(400, 300), cv::Scalar(150, 150, 150), -1); // Gray
    
    // Add some noise
    cv::Mat noise(color_image.size(), CV_8UC3);
    cv::randu(noise, 0, 50);
    cv::add(color_image, noise, color_image);
    
    // Convert to different color spaces for thresholding
    cv::Mat gray, hsv, lab;
    cv::cvtColor(color_image, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(color_image, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(color_image, lab, cv::COLOR_BGR2Lab);
    
    // Threshold grayscale
    cv::Mat gray_thresh;
    cv::threshold(gray, gray_thresh, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // HSV color range thresholding (example: green colors)
    cv::Mat hsv_mask;
    cv::inRange(hsv, cv::Scalar(40, 50, 50), cv::Scalar(80, 255, 255), hsv_mask);
    
    // Per-channel thresholding in BGR
    std::vector<cv::Mat> bgr_channels;
    cv::split(color_image, bgr_channels);
    
    cv::Mat blue_thresh, green_thresh, red_thresh;
    cv::threshold(bgr_channels[0], blue_thresh, 100, 255, cv::THRESH_BINARY);   // Blue channel
    cv::threshold(bgr_channels[1], green_thresh, 100, 255, cv::THRESH_BINARY);  // Green channel
    cv::threshold(bgr_channels[2], red_thresh, 100, 255, cv::THRESH_BINARY);    // Red channel
    
    // Combine channel thresholds
    cv::Mat combined_thresh;
    cv::bitwise_and(blue_thresh, green_thresh, combined_thresh);
    cv::bitwise_and(combined_thresh, red_thresh, combined_thresh);
    
    // Apply mask to original image
    cv::Mat masked_result;
    color_image.copyTo(masked_result, hsv_mask);
    
    // Display results
    cv::namedWindow("Color Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Grayscale Otsu", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("HSV Green Mask", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Blue Channel Thresh", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Green Channel Thresh", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Red Channel Thresh", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Combined Channels", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Green Color Selection", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Color Original", color_image);
    cv::imshow("Grayscale Otsu", gray_thresh);
    cv::imshow("HSV Green Mask", hsv_mask);
    cv::imshow("Blue Channel Thresh", blue_thresh);
    cv::imshow("Green Channel Thresh", green_thresh);
    cv::imshow("Red Channel Thresh", red_thresh);
    cv::imshow("Combined Channels", combined_thresh);
    cv::imshow("Green Color Selection", masked_result);
    
    std::cout << "Color thresholding can isolate specific colors or combine channels." << std::endl;
    std::cout << "HSV is often better for color-based segmentation." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateMultiLevelThresholding(const cv::Mat& src) {
    std::cout << "\n=== Multi-Level Thresholding ===" << std::endl;
    
    cv::Mat multi_level = cv::Mat::zeros(src.size(), CV_8UC1);
    
    // Create multiple threshold levels
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            uchar pixel = src.at<uchar>(y, x);
            if (pixel < 64) {
                multi_level.at<uchar>(y, x) = 0;      // Black
            } else if (pixel < 128) {
                multi_level.at<uchar>(y, x) = 85;     // Dark gray
            } else if (pixel < 192) {
                multi_level.at<uchar>(y, x) = 170;    // Light gray
            } else {
                multi_level.at<uchar>(y, x) = 255;    // White
            }
        }
    }
    
    // Alternative using multiple binary thresholds
    cv::Mat level1, level2, level3, combined_levels;
    cv::threshold(src, level1, 64, 85, cv::THRESH_BINARY);
    cv::threshold(src, level2, 128, 85, cv::THRESH_BINARY);
    cv::threshold(src, level3, 192, 85, cv::THRESH_BINARY);
    
    combined_levels = level1 + level2 + level3;
    
    // Quantization approach
    cv::Mat quantized = src.clone();
    quantized = (quantized / 64) * 64;  // Quantize to 4 levels
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("4-Level Threshold", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Combined Binary", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Quantized", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("4-Level Threshold", multi_level);
    cv::imshow("Combined Binary", combined_levels);
    cv::imshow("Quantized", quantized);
    
    std::cout << "Multi-level thresholding creates multiple intensity regions." << std::endl;
    std::cout << "Useful for segmenting images into multiple classes." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== OpenCV Image Thresholding Demo ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createTestImage();
    
    // Try to load a real image for additional demonstration
    cv::Mat real_image = cv::imread("../data/test.jpg", cv::IMREAD_GRAYSCALE);
    if (!real_image.empty()) {
        std::cout << "Using real image for some demonstrations." << std::endl;
        test_image = real_image;
    } else {
        std::cout << "Using synthetic test image." << std::endl;
    }
    
    // Demonstrate all thresholding techniques
    demonstrateSimpleThresholding(test_image);
    demonstrateAdaptiveThresholding(test_image);
    demonstrateOtsuThresholding(test_image);
    demonstrateColorThresholding();
    demonstrateMultiLevelThresholding(test_image);
    
    std::cout << "\n✓ Image thresholding demonstration complete!" << std::endl;
    std::cout << "Thresholding is fundamental for image segmentation and binary operations." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Simple thresholding works best with uniform lighting
 * 2. Adaptive thresholding handles varying illumination
 * 3. Otsu's method automatically finds optimal threshold
 * 4. Different threshold types serve different purposes
 * 5. Color thresholding requires careful color space selection
 * 6. HSV is often better than BGR for color-based segmentation
 * 7. Multi-level thresholding creates multiple regions
 * 8. Preprocessing (blur, etc.) can improve thresholding results
 */

/**
 * 18-Image_Pyramids.cpp
 * 
 * Multi-scale image analysis using pyramids.
 * 
 * Concepts covered:
 * - Gaussian pyramids
 * - Laplacian pyramids
 * - Image blending
 * - Multi-scale processing
 * - Scale-space analysis
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat createPyramidTestImage() {
    cv::Mat image = cv::Mat::zeros(512, 512, CV_8UC3);
    
    // Create concentric circles with different frequencies
    cv::Point2f center(image.cols/2.0, image.rows/2.0);
    
    for (int r = 20; r < 250; r += 20) {
        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
        cv::circle(image, center, r, color, 3);
    }
    
    // Add high frequency details
    for (int i = 0; i < image.rows; i += 10) {
        cv::line(image, cv::Point(0, i), cv::Point(image.cols, i), cv::Scalar(128, 128, 128), 1);
    }
    
    for (int j = 0; j < image.cols; j += 10) {
        cv::line(image, cv::Point(j, 0), cv::Point(j, image.rows), cv::Scalar(128, 128, 128), 1);
    }
    
    // Add some text for high frequency content
    cv::putText(image, "PYRAMID TEST", cv::Point(150, 450), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
    
    return image;
}

void demonstrateGaussianPyramid(const cv::Mat& src) {
    std::cout << "\n=== Gaussian Pyramid ===" << std::endl;
    
    // Build Gaussian pyramid
    std::vector<cv::Mat> gaussian_pyramid;
    gaussian_pyramid.push_back(src);
    
    cv::Mat current = src;
    for (int i = 0; i < 4; i++) {
        cv::Mat next;
        cv::pyrDown(current, next);
        gaussian_pyramid.push_back(next);
        current = next;
        
        std::cout << "Level " << i+1 << " size: " << next.cols << "x" << next.rows << std::endl;
    }
    
    // Display pyramid levels
    cv::Mat display = cv::Mat::zeros(src.rows, src.cols * 2, CV_8UC3);
    
    // Original image
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    
    // Pyramid levels arranged vertically
    int y_offset = 0;
    for (size_t i = 1; i < gaussian_pyramid.size(); i++) {
        cv::Mat level = gaussian_pyramid[i];
        if (y_offset + level.rows <= src.rows) {
            level.copyTo(display(cv::Rect(src.cols, y_offset, level.cols, level.rows)));
            y_offset += level.rows + 10;
        }
    }
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Gaussian Pyramid", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Gaussian Pyramid", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gaussian Pyramid", display);
    
    // Demonstrate reconstruction
    cv::Mat reconstructed = gaussian_pyramid.back();
    for (int i = gaussian_pyramid.size() - 2; i >= 0; i--) {
        cv::Mat upsampled;
        cv::pyrUp(reconstructed, upsampled, gaussian_pyramid[i].size());
        reconstructed = upsampled;
    }
    
    // Calculate reconstruction error
    cv::Mat diff;
    cv::absdiff(src, reconstructed, diff);
    cv::Scalar mean_error = cv::mean(diff);
    
    std::cout << "Gaussian pyramid properties:" << std::endl;
    std::cout << "  Each level is 1/4 the size of the previous" << std::endl;
    std::cout << "  Low-pass filtering removes high frequency content" << std::endl;
    std::cout << "  Reconstruction error (mean): " << mean_error[0] + mean_error[1] + mean_error[2] << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateLaplacianPyramid(const cv::Mat& src) {
    std::cout << "\n=== Laplacian Pyramid ===" << std::endl;
    
    // Build Gaussian pyramid first
    std::vector<cv::Mat> gaussian_pyramid;
    gaussian_pyramid.push_back(src);
    
    cv::Mat current = src;
    for (int i = 0; i < 4; i++) {
        cv::Mat next;
        cv::pyrDown(current, next);
        gaussian_pyramid.push_back(next);
        current = next;
    }
    
    // Build Laplacian pyramid
    std::vector<cv::Mat> laplacian_pyramid;
    
    for (size_t i = 0; i < gaussian_pyramid.size() - 1; i++) {
        cv::Mat upsampled;
        cv::pyrUp(gaussian_pyramid[i + 1], upsampled, gaussian_pyramid[i].size());
        
        cv::Mat laplacian;
        cv::subtract(gaussian_pyramid[i], upsampled, laplacian);
        laplacian_pyramid.push_back(laplacian);
    }
    
    // The top level is just the smallest Gaussian level
    laplacian_pyramid.push_back(gaussian_pyramid.back());
    
    // Display Laplacian pyramid levels
    cv::Mat display = cv::Mat::zeros(src.rows, src.cols * 2, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    
    // Display Laplacian levels (enhanced for visibility)
    int y_offset = 0;
    for (size_t i = 0; i < laplacian_pyramid.size() && y_offset < src.rows; i++) {
        cv::Mat level = laplacian_pyramid[i];
        
        // Enhance contrast for visualization
        cv::Mat enhanced;
        level.convertTo(enhanced, CV_8UC3, 2.0, 128);  // Scale by 2 and add offset
        
        if (level.rows <= src.rows - y_offset) {
            enhanced.copyTo(display(cv::Rect(src.cols, y_offset, level.cols, level.rows)));
            y_offset += level.rows + 10;
        }
    }
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Laplacian Pyramid", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Laplacian Pyramid", cv::WINDOW_AUTOSIZE);
    cv::imshow("Laplacian Pyramid", display);
    
    // Reconstruct from Laplacian pyramid
    cv::Mat reconstructed = laplacian_pyramid.back().clone();
    
    for (int i = laplacian_pyramid.size() - 2; i >= 0; i--) {
        cv::Mat upsampled;
        cv::pyrUp(reconstructed, upsampled, laplacian_pyramid[i].size());
        cv::add(upsampled, laplacian_pyramid[i], reconstructed);
    }
    
    // Calculate reconstruction error
    cv::Mat diff;
    cv::absdiff(src, reconstructed, diff);
    cv::Scalar mean_error = cv::mean(diff);
    
    std::cout << "Laplacian pyramid properties:" << std::endl;
    std::cout << "  Stores band-pass filtered versions at each scale" << std::endl;
    std::cout << "  Allows perfect reconstruction" << std::endl;
    std::cout << "  Reconstruction error (mean): " << mean_error[0] + mean_error[1] + mean_error[2] << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstratePyramidBlending(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n=== Pyramid Blending ===" << std::endl;
    
    // Ensure images are the same size
    cv::Mat src1, src2;
    cv::resize(img1, src1, cv::Size(512, 512));
    cv::resize(img2, src2, cv::Size(512, 512));
    
    // Create a mask for blending (vertical split)
    cv::Mat mask = cv::Mat::zeros(src1.size(), CV_8UC1);
    mask(cv::Rect(0, 0, mask.cols/2, mask.rows)) = 255;
    
    // Apply Gaussian blur to the mask for smooth blending
    cv::GaussianBlur(mask, mask, cv::Size(51, 51), 20);
    
    // Build Gaussian pyramids for both images
    auto buildGaussianPyramid = [](const cv::Mat& src, int levels) -> std::vector<cv::Mat> {
        std::vector<cv::Mat> pyramid;
        pyramid.push_back(src);
        
        cv::Mat current = src;
        for (int i = 0; i < levels; i++) {
            cv::Mat next;
            cv::pyrDown(current, next);
            pyramid.push_back(next);
            current = next;
        }
        return pyramid;
    };
    
    int levels = 4;
    std::vector<cv::Mat> pyramid1 = buildGaussianPyramid(src1, levels);
    std::vector<cv::Mat> pyramid2 = buildGaussianPyramid(src2, levels);
    std::vector<cv::Mat> mask_pyramid = buildGaussianPyramid(mask, levels);
    
    // Blend at each pyramid level
    std::vector<cv::Mat> blended_pyramid;
    for (size_t i = 0; i < pyramid1.size(); i++) {
        cv::Mat blended;
        cv::Mat mask_3ch;
        
        // Convert mask to 3-channel and normalize
        std::vector<cv::Mat> mask_channels = {mask_pyramid[i], mask_pyramid[i], mask_pyramid[i]};
        cv::merge(mask_channels, mask_3ch);
        mask_3ch.convertTo(mask_3ch, CV_32F, 1.0/255.0);
        
        // Convert images to float
        cv::Mat img1_f, img2_f;
        pyramid1[i].convertTo(img1_f, CV_32F);
        pyramid2[i].convertTo(img2_f, CV_32F);
        
        // Blend: result = img1 * mask + img2 * (1 - mask)
        cv::multiply(img1_f, mask_3ch, img1_f);
        cv::multiply(img2_f, cv::Scalar::all(1.0) - mask_3ch, img2_f);
        cv::add(img1_f, img2_f, blended);
        
        blended.convertTo(blended, CV_8U);
        blended_pyramid.push_back(blended);
    }
    
    // Reconstruct the blended image
    cv::Mat result = blended_pyramid.back();
    for (int i = blended_pyramid.size() - 2; i >= 0; i--) {
        cv::Mat upsampled;
        cv::pyrUp(result, upsampled, blended_pyramid[i].size());
        result = upsampled;
    }
    
    // Simple blending for comparison
    cv::Mat simple_blend;
    mask.convertTo(mask, CV_32F, 1.0/255.0);
    std::vector<cv::Mat> mask_channels = {mask, mask, mask};
    cv::Mat mask_3ch;
    cv::merge(mask_channels, mask_3ch);
    
    cv::Mat src1_f, src2_f;
    src1.convertTo(src1_f, CV_32F);
    src2.convertTo(src2_f, CV_32F);
    
    cv::multiply(src1_f, mask_3ch, src1_f);
    cv::multiply(src2_f, cv::Scalar::all(1.0) - mask_3ch, src2_f);
    cv::add(src1_f, src2_f, simple_blend);
    simple_blend.convertTo(simple_blend, CV_8U);
    
    // Display results
    cv::Mat display = cv::Mat::zeros(src1.rows, src1.cols * 4, CV_8UC3);
    
    src1.copyTo(display(cv::Rect(0, 0, src1.cols, src1.rows)));
    src2.copyTo(display(cv::Rect(src1.cols, 0, src1.cols, src1.rows)));
    simple_blend.copyTo(display(cv::Rect(src1.cols * 2, 0, src1.cols, src1.rows)));
    result.copyTo(display(cv::Rect(src1.cols * 3, 0, src1.cols, src1.rows)));
    
    // Add labels
    cv::putText(display, "Image 1", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Image 2", cv::Point(src1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Simple Blend", cv::Point(src1.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Pyramid Blend", cv::Point(src1.cols * 3 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Pyramid Blending", cv::WINDOW_AUTOSIZE);
    cv::imshow("Pyramid Blending", display);
    
    std::cout << "Pyramid blending advantages:" << std::endl;
    std::cout << "  - Smooth transitions without visible seams" << std::endl;
    std::cout << "  - Preserves both high and low frequency content" << std::endl;
    std::cout << "  - Better than simple alpha blending for natural images" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateMultiScaleProcessing(const cv::Mat& src) {
    std::cout << "\n=== Multi-Scale Processing ===" << std::endl;
    
    // Build Gaussian pyramid
    std::vector<cv::Mat> pyramid;
    pyramid.push_back(src);
    
    cv::Mat current = src;
    for (int i = 0; i < 3; i++) {
        cv::Mat next;
        cv::pyrDown(current, next);
        pyramid.push_back(next);
        current = next;
    }
    
    // Apply different processing at each scale
    std::vector<cv::Mat> processed_pyramid;
    
    for (size_t i = 0; i < pyramid.size(); i++) {
        cv::Mat gray, processed;
        cv::cvtColor(pyramid[i], gray, cv::COLOR_BGR2GRAY);
        
        if (i == 0) {
            // Finest scale: edge detection
            cv::Canny(gray, processed, 50, 150);
        } else if (i == 1) {
            // Medium scale: corner detection
            cv::cornerHarris(gray, processed, 2, 3, 0.04);
            cv::normalize(processed, processed, 0, 255, cv::NORM_MINMAX);
            processed.convertTo(processed, CV_8U);
        } else {
            // Coarse scale: blob detection (simplified)
            cv::GaussianBlur(gray, processed, cv::Size(15, 15), 5);
            cv::threshold(processed, processed, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        }
        
        // Convert to 3-channel for display
        cv::Mat processed_3ch;
        if (processed.channels() == 1) {
            cv::cvtColor(processed, processed_3ch, cv::COLOR_GRAY2BGR);
        } else {
            processed_3ch = processed;
        }
        
        processed_pyramid.push_back(processed_3ch);
    }
    
    // Combine results by upsampling and adding
    cv::Mat combined = cv::Mat::zeros(src.size(), CV_8UC3);
    
    for (int i = processed_pyramid.size() - 1; i >= 0; i--) {
        cv::Mat upsampled;
        if (processed_pyramid[i].size() != src.size()) {
            cv::resize(processed_pyramid[i], upsampled, src.size());
        } else {
            upsampled = processed_pyramid[i];
        }
        
        // Add with different weights for each scale
        cv::addWeighted(combined, 1.0, upsampled, 0.3, 0, combined);
    }
    
    // Display results
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    
    // Resize processed levels for display
    cv::Mat proc1_resized, proc2_resized, proc3_resized;
    cv::resize(processed_pyramid[0], proc1_resized, src.size());
    cv::resize(processed_pyramid[1], proc2_resized, src.size());
    cv::resize(processed_pyramid[2], proc3_resized, src.size());
    
    proc1_resized.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    proc2_resized.copyTo(display(cv::Rect(0, src.rows, src.cols, src.rows)));
    combined.copyTo(display(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Edges (Fine)", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Corners (Medium)", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Combined", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Multi-Scale Processing", cv::WINDOW_AUTOSIZE);
    cv::imshow("Multi-Scale Processing", display);
    
    std::cout << "Multi-scale processing benefits:" << std::endl;
    std::cout << "  - Different features visible at different scales" << std::endl;
    std::cout << "  - Computational efficiency at coarse scales" << std::endl;
    std::cout << "  - Robust feature detection across scales" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Image Pyramids ===" << std::endl;
    
    // Create test images
    cv::Mat test_image = createPyramidTestImage();
    
    // Try to load real images for blending demo
    cv::Mat img1 = cv::imread("../data/test.jpg");
    cv::Mat img2 = test_image;
    
    if (!img1.empty()) {
        std::cout << "Using loaded image for demonstrations." << std::endl;
        
        // Demonstrate pyramid operations
        demonstrateGaussianPyramid(img1);
        demonstrateLaplacianPyramid(img1);
        demonstratePyramidBlending(img1, img2);
        demonstrateMultiScaleProcessing(img1);
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Create second test image for blending
        cv::Mat img2 = cv::Mat::zeros(512, 512, CV_8UC3);
        cv::randu(img2, cv::Scalar::all(0), cv::Scalar::all(255));
        cv::GaussianBlur(img2, img2, cv::Size(15, 15), 5);
        
        // Demonstrate all pyramid operations
        demonstrateGaussianPyramid(test_image);
        demonstrateLaplacianPyramid(test_image);
        demonstratePyramidBlending(test_image, img2);
        demonstrateMultiScaleProcessing(test_image);
    }
    
    std::cout << "\n✓ Image Pyramids demonstration complete!" << std::endl;
    std::cout << "Pyramids enable efficient multi-scale image analysis and processing." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Gaussian pyramids provide multi-resolution representation by successive downsampling
 * 2. Laplacian pyramids store band-pass information and enable perfect reconstruction
 * 3. pyrDown() and pyrUp() functions implement pyramid operations with anti-aliasing
 * 4. Pyramid blending creates seamless image composites better than simple alpha blending
 * 5. Multi-scale processing reveals different features at different resolutions
 * 6. Pyramids reduce computational cost for coarse-to-fine processing strategies
 * 7. Scale-space analysis helps in robust feature detection and matching
 * 8. Proper reconstruction requires careful handling of image sizes and data types
 */

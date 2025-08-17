/**
 * 07-Edge_Detection.cpp
 * 
 * Comprehensive edge detection algorithms in OpenCV.
 * Essential for feature extraction, object detection, and image analysis.
 * 
 * Concepts covered:
 * - Sobel edge detection (X and Y gradients)
 * - Laplacian edge detection
 * - Canny edge detection (multi-stage algorithm)
 * - Scharr operator
 * - Gradient magnitude and direction
 * - Edge thinning and non-maximum suppression
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat createEdgeTestImage() {
    cv::Mat image(400, 600, CV_8UC1, cv::Scalar(128));
    
    // Add various edge types
    cv::rectangle(image, cv::Point(50, 50), cv::Point(200, 200), cv::Scalar(255), -1);  // Square
    cv::circle(image, cv::Point(350, 125), 75, cv::Scalar(0), -1);                      // Circle
    cv::rectangle(image, cv::Point(450, 50), cv::Point(550, 200), cv::Scalar(200), 3);  // Rectangle outline
    
    // Add diagonal line
    cv::line(image, cv::Point(100, 250), cv::Point(300, 350), cv::Scalar(255), 5);
    
    // Add some texture
    for (int y = 300; y < 380; y += 10) {
        cv::line(image, cv::Point(350, y), cv::Point(550, y), cv::Scalar(255), 2);
    }
    
    // Add some noise
    cv::Mat noise(image.size(), CV_8UC1);
    cv::randu(noise, 0, 30);
    cv::add(image, noise, image);
    
    return image;
}

void demonstrateSobelEdges(const cv::Mat& src) {
    std::cout << "\n=== Sobel Edge Detection ===" << std::endl;
    
    cv::Mat grad_x, grad_y, grad_combined;
    cv::Mat abs_grad_x, abs_grad_y;
    
    // Calculate Sobel gradients
    cv::Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    
    // Convert to absolute values and 8-bit
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    // Combine X and Y gradients
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_combined);
    
    // Calculate gradient magnitude and direction
    cv::Mat magnitude, direction;
    cv::Mat grad_x_32f, grad_y_32f;
    grad_x.convertTo(grad_x_32f, CV_32F);
    grad_y.convertTo(grad_y_32f, CV_32F);
    cv::magnitude(grad_x_32f, grad_y_32f, magnitude);
    cv::phase(grad_x_32f, grad_y_32f, direction, true);  // in degrees
    
    // Normalize magnitude for display
    cv::Mat magnitude_norm;
    cv::normalize(magnitude, magnitude_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    // Different kernel sizes
    cv::Mat sobel_3, sobel_5, sobel_7;
    cv::Mat grad_x_3, grad_y_3, grad_x_5, grad_y_5, grad_x_7, grad_y_7;
    
    cv::Sobel(src, grad_x_3, CV_16S, 1, 0, 3);
    cv::Sobel(src, grad_y_3, CV_16S, 0, 1, 3);
    cv::addWeighted(cv::abs(grad_x_3), 0.5, cv::abs(grad_y_3), 0.5, 0, sobel_3);
    
    cv::Sobel(src, grad_x_5, CV_16S, 1, 0, 5);
    cv::Sobel(src, grad_y_5, CV_16S, 0, 1, 5);
    cv::addWeighted(cv::abs(grad_x_5), 0.5, cv::abs(grad_y_5), 0.5, 0, sobel_5);
    
    cv::Sobel(src, grad_x_7, CV_16S, 1, 0, 7);
    cv::Sobel(src, grad_y_7, CV_16S, 0, 1, 7);
    cv::addWeighted(cv::abs(grad_x_7), 0.5, cv::abs(grad_y_7), 0.5, 0, sobel_7);
    
    sobel_3.convertTo(sobel_3, CV_8U);
    sobel_5.convertTo(sobel_5, CV_8U);
    sobel_7.convertTo(sobel_7, CV_8U);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Sobel X", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Sobel Y", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Sobel Combined", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Gradient Magnitude", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Sobel X", abs_grad_x);
    cv::imshow("Sobel Y", abs_grad_y);
    cv::imshow("Sobel Combined", grad_combined);
    cv::imshow("Gradient Magnitude", magnitude_norm);
    
    std::cout << "Sobel X detects vertical edges, Sobel Y detects horizontal edges." << std::endl;
    std::cout << "Press any key to see different kernel sizes..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Show kernel size comparison
    cv::namedWindow("Kernel 3x3", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Kernel 5x5", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Kernel 7x7", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Kernel 3x3", sobel_3);
    cv::imshow("Kernel 5x5", sobel_5);
    cv::imshow("Kernel 7x7", sobel_7);
    
    std::cout << "Larger kernels provide smoother gradients but less detail." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateLaplacianEdges(const cv::Mat& src) {
    std::cout << "\n=== Laplacian Edge Detection ===" << std::endl;
    
    // Apply Gaussian blur first to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    
    // Apply Laplacian operator
    cv::Mat laplacian, abs_laplacian;
    cv::Laplacian(blurred, laplacian, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(laplacian, abs_laplacian);
    
    // Different kernel sizes
    cv::Mat laplacian_1, laplacian_3, laplacian_5;
    cv::Laplacian(blurred, laplacian_1, CV_8U, 1);
    cv::Laplacian(blurred, laplacian_3, CV_8U, 3);
    cv::Laplacian(blurred, laplacian_5, CV_8U, 5);
    
    // Laplacian of Gaussian (LoG)
    cv::Mat log_result;
    cv::GaussianBlur(src, blurred, cv::Size(5, 5), 1.4);
    cv::Laplacian(blurred, log_result, CV_8U, 3);
    
    // Zero-crossing detection (simplified)
    cv::Mat zero_crossings = cv::Mat::zeros(laplacian.size(), CV_8U);
    for (int y = 1; y < laplacian.rows - 1; y++) {
        for (int x = 1; x < laplacian.cols - 1; x++) {
            short center = laplacian.at<short>(y, x);
            short neighbors[8] = {
                laplacian.at<short>(y-1, x-1), laplacian.at<short>(y-1, x), laplacian.at<short>(y-1, x+1),
                laplacian.at<short>(y, x-1),                                   laplacian.at<short>(y, x+1),
                laplacian.at<short>(y+1, x-1), laplacian.at<short>(y+1, x),   laplacian.at<short>(y+1, x+1)
            };
            
            for (int i = 0; i < 8; i++) {
                if ((center > 0 && neighbors[i] < 0) || (center < 0 && neighbors[i] > 0)) {
                    if (abs(center - neighbors[i]) > 30) {  // Threshold for significance
                        zero_crossings.at<uchar>(y, x) = 255;
                        break;
                    }
                }
            }
        }
    }
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Laplacian", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Kernel 1x1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Kernel 3x3", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Kernel 5x5", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("LoG (Laplacian of Gaussian)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Zero Crossings", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Laplacian", abs_laplacian);
    cv::imshow("Kernel 1x1", laplacian_1);
    cv::imshow("Kernel 3x3", laplacian_3);
    cv::imshow("Kernel 5x5", laplacian_5);
    cv::imshow("LoG (Laplacian of Gaussian)", log_result);
    cv::imshow("Zero Crossings", zero_crossings);
    
    std::cout << "Laplacian detects edges as zero-crossings of second derivative." << std::endl;
    std::cout << "LoG combines Gaussian smoothing with Laplacian edge detection." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateCannyEdges(const cv::Mat& src) {
    std::cout << "\n=== Canny Edge Detection ===" << std::endl;
    
    cv::Mat canny_default, canny_tight, canny_loose, canny_blurred;
    
    // Default Canny parameters
    cv::Canny(src, canny_default, 100, 200, 3, false);
    
    // Tight thresholds (less edges)
    cv::Canny(src, canny_tight, 150, 250, 3, false);
    
    // Loose thresholds (more edges)
    cv::Canny(src, canny_loose, 50, 150, 3, false);
    
    // Canny with preprocessing
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, canny_blurred, 100, 200, 3, false);
    
    // Different kernel sizes
    cv::Mat canny_3, canny_5, canny_7;
    cv::Canny(src, canny_3, 100, 200, 3, false);
    cv::Canny(src, canny_5, 100, 200, 5, false);
    cv::Canny(src, canny_7, 100, 200, 7, false);
    
    // Automatic threshold calculation using Otsu
    cv::Mat binary;
    double otsu_thresh = cv::threshold(src, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::Mat canny_auto;
    cv::Canny(src, canny_auto, otsu_thresh * 0.5, otsu_thresh, 3, false);
    
    // L2 gradient (more accurate)
    cv::Mat canny_l2;
    cv::Canny(src, canny_l2, 100, 200, 3, true);  // L2 gradient = true
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Canny Default (100,200)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Canny Tight (150,250)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Canny Loose (50,150)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Canny with Blur", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Canny Default (100,200)", canny_default);
    cv::imshow("Canny Tight (150,250)", canny_tight);
    cv::imshow("Canny Loose (50,150)", canny_loose);
    cv::imshow("Canny with Blur", canny_blurred);
    
    std::cout << "Canny uses double thresholding: low and high thresholds." << std::endl;
    std::cout << "High threshold finds strong edges, low threshold extends them." << std::endl;
    std::cout << "Press any key to see other variations..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    cv::namedWindow("Kernel 3x3", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Kernel 5x5", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Kernel 7x7", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Auto Threshold", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("L2 Gradient", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Kernel 3x3", canny_3);
    cv::imshow("Kernel 5x5", canny_5);
    cv::imshow("Kernel 7x7", canny_7);
    cv::imshow("Auto Threshold", canny_auto);
    cv::imshow("L2 Gradient", canny_l2);
    
    std::cout << "Automatic threshold based on Otsu: " << otsu_thresh << std::endl;
    std::cout << "L2 gradient provides more accurate gradient magnitude." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateScharrEdges(const cv::Mat& src) {
    std::cout << "\n=== Scharr Edge Detection ===" << std::endl;
    
    cv::Mat scharr_x, scharr_y, scharr_combined;
    cv::Mat abs_scharr_x, abs_scharr_y;
    
    // Scharr operator (more accurate for small kernels)
    cv::Scharr(src, scharr_x, CV_16S, 1, 0, 1, 0, cv::BORDER_DEFAULT);
    cv::Scharr(src, scharr_y, CV_16S, 0, 1, 1, 0, cv::BORDER_DEFAULT);
    
    cv::convertScaleAbs(scharr_x, abs_scharr_x);
    cv::convertScaleAbs(scharr_y, abs_scharr_y);
    cv::addWeighted(abs_scharr_x, 0.5, abs_scharr_y, 0.5, 0, scharr_combined);
    
    // Compare with Sobel 3x3
    cv::Mat sobel_x, sobel_y, sobel_combined;
    cv::Mat abs_sobel_x, abs_sobel_y;
    
    cv::Sobel(src, sobel_x, CV_16S, 1, 0, 3);
    cv::Sobel(src, sobel_y, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(sobel_x, abs_sobel_x);
    cv::convertScaleAbs(sobel_y, abs_sobel_y);
    cv::addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0, sobel_combined);
    
    // Difference between Scharr and Sobel
    cv::Mat difference;
    cv::absdiff(scharr_combined, sobel_combined, difference);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Scharr X", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Scharr Y", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Scharr Combined", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Sobel 3x3 Combined", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Scharr vs Sobel Diff", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Scharr X", abs_scharr_x);
    cv::imshow("Scharr Y", abs_scharr_y);
    cv::imshow("Scharr Combined", scharr_combined);
    cv::imshow("Sobel 3x3 Combined", sobel_combined);
    cv::imshow("Scharr vs Sobel Diff", difference);
    
    std::cout << "Scharr operator provides better rotation invariance than Sobel 3x3." << std::endl;
    std::cout << "Useful when precise gradient calculation is needed." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateEdgeComparison(const cv::Mat& src) {
    std::cout << "\n=== Edge Detection Methods Comparison ===" << std::endl;
    
    // Apply all methods with similar parameters
    cv::Mat sobel, laplacian, canny, scharr;
    
    // Sobel
    cv::Mat grad_x, grad_y;
    cv::Sobel(src, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(src, grad_y, CV_16S, 0, 1, 3);
    cv::addWeighted(cv::abs(grad_x), 0.5, cv::abs(grad_y), 0.5, 0, sobel);
    sobel.convertTo(sobel, CV_8U);
    
    // Laplacian
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(3, 3), 0);
    cv::Laplacian(blurred, laplacian, CV_8U, 3);
    
    // Canny
    cv::Canny(src, canny, 100, 200, 3);
    
    // Scharr
    cv::Scharr(src, grad_x, CV_16S, 1, 0);
    cv::Scharr(src, grad_y, CV_16S, 0, 1);
    cv::addWeighted(cv::abs(grad_x), 0.5, cv::abs(grad_y), 0.5, 0, scharr);
    scharr.convertTo(scharr, CV_8U);
    
    // Create comparison grid
    cv::Mat top_row, bottom_row, comparison;
    cv::hconcat(std::vector<cv::Mat>{src, sobel}, top_row);
    cv::hconcat(std::vector<cv::Mat>{laplacian, canny}, bottom_row);
    cv::vconcat(std::vector<cv::Mat>{top_row, bottom_row}, comparison);
    
    // Add labels
    cv::putText(comparison, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2);
    cv::putText(comparison, "Sobel", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2);
    cv::putText(comparison, "Laplacian", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2);
    cv::putText(comparison, "Canny", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2);
    
    cv::namedWindow("Edge Detection Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("Edge Detection Comparison", comparison);
    
    cv::namedWindow("Scharr", cv::WINDOW_AUTOSIZE);
    cv::imshow("Scharr", scharr);
    
    std::cout << "Method characteristics:" << std::endl;
    std::cout << "- Sobel: Good general purpose, directional gradients" << std::endl;
    std::cout << "- Laplacian: Omnidirectional, sensitive to noise" << std::endl;
    std::cout << "- Canny: Multi-stage, clean thin edges" << std::endl;
    std::cout << "- Scharr: Better rotation invariance than Sobel" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== OpenCV Edge Detection Demo ===" << std::endl;
    
    // Create or load test image
    cv::Mat test_image = createEdgeTestImage();
    
    // Try to load a real image
    cv::Mat real_image = cv::imread("../data/test.jpg", cv::IMREAD_GRAYSCALE);
    if (!real_image.empty()) {
        test_image = real_image;
        std::cout << "Using loaded grayscale image for demonstrations." << std::endl;
    } else {
        std::cout << "Using synthetic test image." << std::endl;
    }
    
    // Demonstrate all edge detection methods
    demonstrateSobelEdges(test_image);
    demonstrateLaplacianEdges(test_image);
    demonstrateCannyEdges(test_image);
    demonstrateScharrEdges(test_image);
    demonstrateEdgeComparison(test_image);
    
    std::cout << "\n✓ Edge detection demonstration complete!" << std::endl;
    std::cout << "Choose edge detection method based on noise level, accuracy needs, and application." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Sobel: First derivative, directional gradients, good general purpose
 * 2. Laplacian: Second derivative, omnidirectional, sensitive to noise
 * 3. Canny: Multi-stage algorithm, best overall performance, thin edges
 * 4. Scharr: Better rotation invariance than Sobel for 3x3 kernels
 * 5. Preprocessing with Gaussian blur reduces noise sensitivity
 * 6. Threshold selection critical for quality (automatic methods help)
 * 7. Each method has different characteristics and optimal use cases
 * 8. Canny is most commonly used for general edge detection tasks
 */

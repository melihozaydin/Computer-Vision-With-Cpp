/**
 * 09-Morphological_Operations.cpp
 * 
 * Mathematical morphology for shape analysis and filtering.
 * 
 * Concepts covered:
 * - Erosion and dilation
 * - Opening and closing
 * - Morphological gradient
 * - Top-hat and black-hat
 * - Custom structuring elements
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat createMorphologyTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC1);
    
    // Add various shapes for morphological operations
    cv::rectangle(image, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255), -1);
    cv::circle(image, cv::Point(250, 100), 40, cv::Scalar(255), -1);
    
    // Add some thin structures
    cv::line(image, cv::Point(350, 50), cv::Point(350, 150), cv::Scalar(255), 3);
    cv::line(image, cv::Point(400, 50), cv::Point(500, 150), cv::Scalar(255), 2);
    
    // Add some noisy dots
    for (int i = 0; i < 20; i++) {
        cv::Point pt(50 + i * 25, 200 + (i % 3) * 10);
        cv::circle(image, pt, 2, cv::Scalar(255), -1);
    }
    
    // Add text (which has fine details)
    cv::putText(image, "MORPHOLOGY", cv::Point(100, 300), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2);
    
    // Add some connected components with bridges
    cv::rectangle(image, cv::Point(100, 320), cv::Point(140, 360), cv::Scalar(255), -1);
    cv::rectangle(image, cv::Point(160, 320), cv::Point(200, 360), cv::Scalar(255), -1);
    cv::line(image, cv::Point(140, 340), cv::Point(160, 340), cv::Scalar(255), 3);  // Bridge
    
    return image;
}

void demonstrateBasicOperations(const cv::Mat& src) {
    std::cout << "\n=== Basic Morphological Operations ===" << std::endl;
    
    // Define structuring element
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    
    cv::Mat eroded, dilated, opened, closed;
    
    // Basic operations
    cv::erode(src, eroded, kernel, cv::Point(-1, -1), 1);
    cv::dilate(src, dilated, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(src, opened, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(src, closed, cv::MORPH_CLOSE, kernel);
    
    // Multiple iterations
    cv::Mat eroded_2, dilated_2;
    cv::erode(src, eroded_2, kernel, cv::Point(-1, -1), 2);
    cv::dilate(src, dilated_2, kernel, cv::Point(-1, -1), 2);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Eroded (1 iter)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Dilated (1 iter)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Opened", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Closed", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Eroded (1 iter)", eroded);
    cv::imshow("Dilated (1 iter)", dilated);
    cv::imshow("Opened", opened);
    cv::imshow("Closed", closed);
    
    std::cout << "Erosion: shrinks white regions, removes small noise" << std::endl;
    std::cout << "Dilation: expands white regions, fills small holes" << std::endl;
    std::cout << "Opening: erosion followed by dilation, removes noise" << std::endl;
    std::cout << "Closing: dilation followed by erosion, fills gaps" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Show multiple iterations
    cv::namedWindow("Eroded (2 iter)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Dilated (2 iter)", cv::WINDOW_AUTOSIZE);
    cv::imshow("Eroded (2 iter)", eroded_2);
    cv::imshow("Dilated (2 iter)", dilated_2);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateAdvancedOperations(const cv::Mat& src) {
    std::cout << "\n=== Advanced Morphological Operations ===" << std::endl;
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    
    cv::Mat gradient, tophat, blackhat;
    
    // Morphological gradient (outline)
    cv::morphologyEx(src, gradient, cv::MORPH_GRADIENT, kernel);
    
    // Top-hat (original - opening): highlights small bright structures
    cv::morphologyEx(src, tophat, cv::MORPH_TOPHAT, kernel);
    
    // Black-hat (closing - original): highlights small dark structures
    cv::morphologyEx(src, blackhat, cv::MORPH_BLACKHAT, kernel);
    
    // Different kernel sizes for gradient
    cv::Mat kernel_small = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat kernel_large = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    
    cv::Mat gradient_small, gradient_large;
    cv::morphologyEx(src, gradient_small, cv::MORPH_GRADIENT, kernel_small);
    cv::morphologyEx(src, gradient_large, cv::MORPH_GRADIENT, kernel_large);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Morphological Gradient", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Top-hat", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Black-hat", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Morphological Gradient", gradient);
    cv::imshow("Top-hat", tophat);
    cv::imshow("Black-hat", blackhat);
    
    std::cout << "Gradient: edge detection using morphology" << std::endl;
    std::cout << "Top-hat: extracts small bright features" << std::endl;
    std::cout << "Black-hat: extracts small dark features" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Show different gradient sizes
    cv::namedWindow("Gradient Small (3x3)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Gradient Large (9x9)", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gradient Small (3x3)", gradient_small);
    cv::imshow("Gradient Large (9x9)", gradient_large);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateStructuringElements(const cv::Mat& src) {
    std::cout << "\n=== Different Structuring Elements ===" << std::endl;
    
    // Different shapes
    cv::Mat kernel_rect = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat kernel_cross = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(7, 7));
    cv::Mat kernel_ellipse = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    
    // Custom kernel
    cv::Mat kernel_custom = (cv::Mat_<uchar>(5, 5) <<
        0, 1, 1, 1, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 1, 1, 1, 0);
    
    // Line structuring elements for specific directions
    cv::Mat kernel_horizontal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 1));
    cv::Mat kernel_vertical = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 9));
    
    // Apply morphological opening with different kernels
    cv::Mat result_rect, result_cross, result_ellipse, result_custom;
    cv::Mat result_horizontal, result_vertical;
    
    cv::morphologyEx(src, result_rect, cv::MORPH_OPEN, kernel_rect);
    cv::morphologyEx(src, result_cross, cv::MORPH_OPEN, kernel_cross);
    cv::morphologyEx(src, result_ellipse, cv::MORPH_OPEN, kernel_ellipse);
    cv::morphologyEx(src, result_custom, cv::MORPH_OPEN, kernel_custom);
    cv::morphologyEx(src, result_horizontal, cv::MORPH_OPEN, kernel_horizontal);
    cv::morphologyEx(src, result_vertical, cv::MORPH_OPEN, kernel_vertical);
    
    // Display kernels (scaled up for visibility)
    cv::Mat kernel_rect_vis, kernel_cross_vis, kernel_ellipse_vis, kernel_custom_vis;
    cv::resize(kernel_rect * 255, kernel_rect_vis, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
    cv::resize(kernel_cross * 255, kernel_cross_vis, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
    cv::resize(kernel_ellipse * 255, kernel_ellipse_vis, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
    cv::resize(kernel_custom * 255, kernel_custom_vis, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
    
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Rectangle Kernel", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Cross Kernel", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ellipse Kernel", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Custom Kernel", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Rectangle Kernel", result_rect);
    cv::imshow("Cross Kernel", result_cross);
    cv::imshow("Ellipse Kernel", result_ellipse);
    cv::imshow("Custom Kernel", result_custom);
    
    std::cout << "Different kernel shapes produce different effects:" << std::endl;
    std::cout << "- Rectangle: preserves rectangular features" << std::endl;
    std::cout << "- Cross: preserves thin linear features better" << std::endl;
    std::cout << "- Ellipse: more isotropic, good for round features" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Show directional kernels
    cv::namedWindow("Horizontal Opening", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Vertical Opening", cv::WINDOW_AUTOSIZE);
    cv::imshow("Horizontal Opening", result_horizontal);
    cv::imshow("Vertical Opening", result_vertical);
    
    std::cout << "Directional kernels extract features in specific orientations." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateNoiseRemoval(const cv::Mat& src) {
    std::cout << "\n=== Noise Removal with Morphology ===" << std::endl;
    
    // Add noise to the image
    cv::Mat noisy = src.clone();
    
    // Add salt and pepper noise
    cv::Mat noise(src.size(), CV_8UC1);
    cv::randu(noise, 0, 255);
    cv::Mat salt_mask = noise > 240;
    cv::Mat pepper_mask = noise < 15;
    noisy.setTo(255, salt_mask);
    noisy.setTo(0, pepper_mask);
    
    // Add small random dots
    for (int i = 0; i < 50; i++) {
        cv::Point pt(rand() % noisy.cols, rand() % noisy.rows);
        cv::circle(noisy, pt, 1, cv::Scalar(255), -1);
    }
    
    // Different approaches to noise removal
    cv::Mat kernel_small = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat kernel_medium = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    
    cv::Mat opened, closed, open_close, close_open;
    
    // Opening removes white noise (small bright spots)
    cv::morphologyEx(noisy, opened, cv::MORPH_OPEN, kernel_small);
    
    // Closing removes black noise (small dark spots)
    cv::morphologyEx(noisy, closed, cv::MORPH_CLOSE, kernel_small);
    
    // Open-close: opening followed by closing
    cv::morphologyEx(opened, open_close, cv::MORPH_CLOSE, kernel_small);
    
    // Close-open: closing followed by opening
    cv::morphologyEx(closed, close_open, cv::MORPH_OPEN, kernel_small);
    
    // Alternating sequential filter (more iterations)
    cv::Mat asf = noisy.clone();
    for (int i = 0; i < 3; i++) {
        cv::morphologyEx(asf, asf, cv::MORPH_OPEN, kernel_small);
        cv::morphologyEx(asf, asf, cv::MORPH_CLOSE, kernel_small);
    }
    
    cv::namedWindow("Original Clean", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Noisy", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("After Opening", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("After Closing", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Open-Close", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Close-Open", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("ASF (3 iterations)", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original Clean", src);
    cv::imshow("Noisy", noisy);
    cv::imshow("After Opening", opened);
    cv::imshow("After Closing", closed);
    cv::imshow("Open-Close", open_close);
    cv::imshow("Close-Open", close_open);
    cv::imshow("ASF (3 iterations)", asf);
    
    std::cout << "Morphological operations can effectively remove noise:" << std::endl;
    std::cout << "- Opening removes small bright noise" << std::endl;
    std::cout << "- Closing removes small dark noise" << std::endl;
    std::cout << "- Combination handles both types" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateConnectedComponentAnalysis(const cv::Mat&) {
    std::cout << "\n=== Morphology for Connected Component Analysis ===" << std::endl;
    
    // Create an image with separated components
    cv::Mat binary = cv::Mat::zeros(300, 400, CV_8UC1);
    
    // Add several components
    cv::rectangle(binary, cv::Point(50, 50), cv::Point(100, 100), cv::Scalar(255), -1);
    cv::rectangle(binary, cv::Point(120, 60), cv::Point(140, 120), cv::Scalar(255), -1);  // Thin bridge
    cv::rectangle(binary, cv::Point(160, 50), cv::Point(210, 100), cv::Scalar(255), -1);
    
    cv::circle(binary, cv::Point(300, 80), 30, cv::Scalar(255), -1);
    
    // Add some components that are barely connected
    cv::rectangle(binary, cv::Point(50, 150), cv::Point(100, 200), cv::Scalar(255), -1);
    cv::rectangle(binary, cv::Point(110, 150), cv::Point(160, 200), cv::Scalar(255), -1);
    cv::line(binary, cv::Point(100, 175), cv::Point(110, 175), cv::Scalar(255), 1);  // Thin connection
    
    // Separate components by breaking connections
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat separated;
    cv::morphologyEx(binary, separated, cv::MORPH_OPEN, kernel_open);
    
    // Connect nearby components
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::Mat connected;
    cv::morphologyEx(binary, connected, cv::MORPH_CLOSE, kernel_close);
    
    // Find connected components before and after
    cv::Mat labels_before, labels_after_sep, labels_after_conn;
    cv::Mat stats_before, stats_after_sep, stats_after_conn;
    cv::Mat centroids_before, centroids_after_sep, centroids_after_conn;
    
    int num_before = cv::connectedComponentsWithStats(binary, labels_before, stats_before, centroids_before);
    int num_after_sep = cv::connectedComponentsWithStats(separated, labels_after_sep, stats_after_sep, centroids_after_sep);
    int num_after_conn = cv::connectedComponentsWithStats(connected, labels_after_conn, stats_after_conn, centroids_after_conn);
    
    // Create colored visualizations
    cv::Mat colored_before, colored_sep, colored_conn;
    cv::applyColorMap(labels_before * 50, colored_before, cv::COLORMAP_JET);
    cv::applyColorMap(labels_after_sep * 50, colored_sep, cv::COLORMAP_JET);
    cv::applyColorMap(labels_after_conn * 50, colored_conn, cv::COLORMAP_JET);
    
    cv::namedWindow("Original Binary", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("After Opening (Separated)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("After Closing (Connected)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Components Before", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Components After Sep", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Components After Conn", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original Binary", binary);
    cv::imshow("After Opening (Separated)", separated);
    cv::imshow("After Closing (Connected)", connected);
    cv::imshow("Components Before", colored_before);
    cv::imshow("Components After Sep", colored_sep);
    cv::imshow("Components After Conn", colored_conn);
    
    std::cout << "Connected components analysis:" << std::endl;
    std::cout << "Original: " << (num_before - 1) << " components" << std::endl;  // -1 for background
    std::cout << "After separation: " << (num_after_sep - 1) << " components" << std::endl;
    std::cout << "After connection: " << (num_after_conn - 1) << " components" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Morphological Operations ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createMorphologyTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg", cv::IMREAD_GRAYSCALE);
    if (!real_image.empty()) {
        cv::Mat binary_real;
        cv::threshold(real_image, binary_real, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        std::cout << "Using loaded image for some demonstrations." << std::endl;
        test_image = binary_real;
    } else {
        std::cout << "Using synthetic test image." << std::endl;
    }
    
    // Demonstrate all morphological operations
    demonstrateBasicOperations(test_image);
    demonstrateAdvancedOperations(test_image);
    demonstrateStructuringElements(test_image);
    demonstrateNoiseRemoval(test_image);
    demonstrateConnectedComponentAnalysis(test_image);
    
    std::cout << "\n✓ Morphological Operations demonstration complete!" << std::endl;
    std::cout << "Morphological operations are fundamental for shape analysis and image preprocessing." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Erosion shrinks white regions, removes small objects
 * 2. Dilation expands white regions, fills small holes
 * 3. Opening (erosion + dilation) removes noise, separates objects
 * 4. Closing (dilation + erosion) fills gaps, connects nearby objects
 * 5. Morphological gradient detects edges/boundaries
 * 6. Top-hat extracts small bright features
 * 7. Structuring element shape affects operation behavior
 * 8. Sequential operations can achieve complex shape analysis
 */

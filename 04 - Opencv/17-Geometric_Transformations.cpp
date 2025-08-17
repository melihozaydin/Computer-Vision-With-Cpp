/**
 * 17-Geometric_Transformations.cpp
 * 
 * Comprehensive geometric transformations in OpenCV.
 * 
 * Concepts covered:
 * - Translation, rotation, scaling
 * - Affine transformations
 * - Perspective transformations
 * - Homography
 * - Warping and rectification
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat createGeometricTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // Create a chessboard pattern for easy visualization of transformations
    int square_size = 50;
    for (int i = 0; i < image.rows; i += square_size) {
        for (int j = 0; j < image.cols; j += square_size) {
            if ((i/square_size + j/square_size) % 2 == 0) {
                cv::rectangle(image, cv::Point(j, i), cv::Point(j + square_size, i + square_size), 
                             cv::Scalar(255, 255, 255), -1);
            }
        }
    }
    
    // Add some distinctive features
    cv::circle(image, cv::Point(150, 150), 30, cv::Scalar(0, 0, 255), -1);        // Red circle
    cv::rectangle(image, cv::Point(400, 100), cv::Point(500, 200), cv::Scalar(0, 255, 0), -1);  // Green rectangle
    cv::putText(image, "OpenCV", cv::Point(200, 350), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 3);
    
    return image;
}

void demonstrateBasicTransformations(const cv::Mat& src) {
    std::cout << "\n=== Basic Transformations ===" << std::endl;
    
    // 1. Translation
    cv::Mat translation_matrix = (cv::Mat_<double>(2, 3) << 1, 0, 50, 0, 1, 100);
    cv::Mat translated;
    cv::warpAffine(src, translated, translation_matrix, src.size());
    
    // 2. Rotation
    cv::Point2f center(src.cols/2.0, src.rows/2.0);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, 30, 1.0);  // 30 degrees, scale 1.0
    cv::Mat rotated;
    cv::warpAffine(src, rotated, rotation_matrix, src.size());
    
    // 3. Scaling
    cv::Mat scaling_matrix = (cv::Mat_<double>(2, 3) << 1.5, 0, -src.cols*0.25, 0, 1.5, -src.rows*0.25);
    cv::Mat scaled;
    cv::warpAffine(src, scaled, scaling_matrix, src.size());
    
    // 4. Combined transformation (rotation + scaling)
    cv::Mat combined_matrix = cv::getRotationMatrix2D(center, 45, 0.7);  // 45 degrees, scale 0.7
    cv::Mat combined;
    cv::warpAffine(src, combined, combined_matrix, src.size());
    
    // Create a 2x2 grid for display
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    translated.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    rotated.copyTo(display(cv::Rect(0, src.rows, src.cols, src.rows)));
    scaled.copyTo(display(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Translated", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Rotated", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Scaled", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Basic Transformations", cv::WINDOW_AUTOSIZE);
    cv::imshow("Basic Transformations", display);
    
    std::cout << "Basic transformations demonstrated:" << std::endl;
    std::cout << "  Translation: Shifting image by (50, 100) pixels" << std::endl;
    std::cout << "  Rotation: 30 degrees around center" << std::endl;
    std::cout << "  Scaling: 1.5x enlargement" << std::endl;
    std::cout << "  Combined: 45 degrees rotation with 0.7x scaling" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateAffineTransformations(const cv::Mat& src) {
    std::cout << "\n=== Affine Transformations ===" << std::endl;
    
    // Define source points (triangle)
    std::vector<cv::Point2f> src_points = {
        cv::Point2f(0, 0),
        cv::Point2f(src.cols - 1, 0),
        cv::Point2f(0, src.rows - 1)
    };
    
    // Define destination points for different transformations
    std::vector<cv::Point2f> dst_points1 = {
        cv::Point2f(src.cols * 0.0, src.rows * 0.33),      // Shear transformation
        cv::Point2f(src.cols * 0.85, src.rows * 0.25),
        cv::Point2f(src.cols * 0.15, src.rows * 0.7)
    };
    
    std::vector<cv::Point2f> dst_points2 = {
        cv::Point2f(src.cols * 0.2, src.rows * 0.1),       // Perspective-like effect
        cv::Point2f(src.cols * 0.8, src.rows * 0.0),
        cv::Point2f(src.cols * 0.1, src.rows * 0.9)
    };
    
    // Calculate affine transformation matrices
    cv::Mat affine_matrix1 = cv::getAffineTransform(src_points, dst_points1);
    cv::Mat affine_matrix2 = cv::getAffineTransform(src_points, dst_points2);
    
    // Apply transformations
    cv::Mat affine_result1, affine_result2;
    cv::warpAffine(src, affine_result1, affine_matrix1, src.size());
    cv::warpAffine(src, affine_result2, affine_matrix2, src.size());
    
    // Manual shear transformation
    cv::Mat shear_matrix = (cv::Mat_<double>(2, 3) << 1, 0.5, 0, 0.2, 1, 0);
    cv::Mat shear_result;
    cv::warpAffine(src, shear_result, shear_matrix, src.size());
    
    // Display results
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    affine_result1.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    affine_result2.copyTo(display(cv::Rect(0, src.rows, src.cols, src.rows)));
    shear_result.copyTo(display(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Draw source and destination points on first result for visualization
    cv::Mat marked_result1 = affine_result1.clone();
    for (size_t i = 0; i < src_points.size(); i++) {
        cv::circle(marked_result1, dst_points1[i], 5, cv::Scalar(0, 255, 255), -1);
        cv::putText(marked_result1, std::to_string(i), cv::Point(dst_points1[i].x + 10, dst_points1[i].y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
    }
    marked_result1.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Affine 1", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Affine 2", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Shear", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Affine Transformations", cv::WINDOW_AUTOSIZE);
    cv::imshow("Affine Transformations", display);
    
    std::cout << "Affine transformation properties:" << std::endl;
    std::cout << "  - Preserves parallel lines" << std::endl;
    std::cout << "  - Preserves ratios of distances along lines" << std::endl;
    std::cout << "  - Requires 3 point correspondences" << std::endl;
    std::cout << "  - Combines rotation, scaling, shearing, and translation" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstratePerspectiveTransformations(const cv::Mat& src) {
    std::cout << "\n=== Perspective Transformations ===" << std::endl;
    
    // Define source points (rectangle)
    std::vector<cv::Point2f> src_points = {
        cv::Point2f(0, 0),
        cv::Point2f(src.cols - 1, 0),
        cv::Point2f(src.cols - 1, src.rows - 1),
        cv::Point2f(0, src.rows - 1)
    };
    
    // Define destination points for perspective effects
    std::vector<cv::Point2f> dst_points1 = {
        cv::Point2f(src.cols * 0.1, src.rows * 0.2),       // Top-left moved in
        cv::Point2f(src.cols * 0.9, src.rows * 0.2),       // Top-right moved in
        cv::Point2f(src.cols * 1.0, src.rows * 0.8),       // Bottom-right unchanged
        cv::Point2f(src.cols * 0.0, src.rows * 0.8)        // Bottom-left unchanged
    };
    
    // Perspective transformation (trapezoid effect)
    std::vector<cv::Point2f> dst_points2 = {
        cv::Point2f(src.cols * 0.2, 0),
        cv::Point2f(src.cols * 0.8, 0),
        cv::Point2f(src.cols * 1.0, src.rows - 1),
        cv::Point2f(src.cols * 0.0, src.rows - 1)
    };
    
    // Calculate perspective transformation matrices
    cv::Mat perspective_matrix1 = cv::getPerspectiveTransform(src_points, dst_points1);
    cv::Mat perspective_matrix2 = cv::getPerspectiveTransform(src_points, dst_points2);
    
    // Apply perspective transformations
    cv::Mat perspective_result1, perspective_result2;
    cv::warpPerspective(src, perspective_result1, perspective_matrix1, src.size());
    cv::warpPerspective(src, perspective_result2, perspective_matrix2, src.size());
    
    // Inverse perspective transformation (rectification)
    cv::Mat inverse_matrix = cv::getPerspectiveTransform(dst_points1, src_points);
    cv::Mat rectified;
    cv::warpPerspective(perspective_result1, rectified, inverse_matrix, src.size());
    
    // Display results
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    perspective_result1.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    perspective_result2.copyTo(display(cv::Rect(0, src.rows, src.cols, src.rows)));
    rectified.copyTo(display(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Draw corner points for visualization
    cv::Mat marked_result = perspective_result1.clone();
    for (size_t i = 0; i < dst_points1.size(); i++) {
        cv::circle(marked_result, dst_points1[i], 5, cv::Scalar(0, 255, 255), -1);
        cv::putText(marked_result, std::to_string(i), cv::Point(dst_points1[i].x + 10, dst_points1[i].y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
    }
    marked_result.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Perspective 1", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Perspective 2", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Rectified", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Perspective Transformations", cv::WINDOW_AUTOSIZE);
    cv::imshow("Perspective Transformations", display);
    
    std::cout << "Perspective transformation properties:" << std::endl;
    std::cout << "  - Can change perspective view" << std::endl;
    std::cout << "  - Does not preserve parallel lines" << std::endl;
    std::cout << "  - Requires 4 point correspondences" << std::endl;
    std::cout << "  - Useful for document scanning and rectification" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Geometric Transformations ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createGeometricTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg");
    if (!real_image.empty()) {
        std::cout << "Using loaded image for demonstrations." << std::endl;
        
        // Use real image for demonstrations
        demonstrateBasicTransformations(real_image);
        demonstrateAffineTransformations(real_image);
        demonstratePerspectiveTransformations(real_image);
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all transformations
        demonstrateBasicTransformations(test_image);
        demonstrateAffineTransformations(test_image);
        demonstratePerspectiveTransformations(test_image);
    }
    
    std::cout << "\n✓ Geometric Transformations demonstration complete!" << std::endl;
    std::cout << "Geometric transformations are fundamental for image registration, augmentation, and correction." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Basic transformations (translation, rotation, scaling) use 2x3 matrices
 * 2. Affine transformations preserve parallel lines and require 3 point correspondences
 * 3. Perspective transformations can change viewpoint and require 4 point correspondences
 * 4. Homography relates two views of the same planar surface
 * 5. warpAffine() for affine transformations, warpPerspective() for projective ones
 * 6. Custom warping using remap() allows arbitrary geometric distortions
 * 7. Polar transformations useful for rotation-invariant processing
 * 8. Proper interpolation methods (INTER_LINEAR, INTER_CUBIC) affect quality
 */

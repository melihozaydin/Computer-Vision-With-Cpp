/**
 * 02-Basic_Operations.cpp
 * 
 * Essential image manipulation operations in OpenCV.
 * Covers resizing, cropping, flipping, rotation, and drawing operations.
 * 
 * Concepts covered:
 * - Image resizing and scaling
 * - Cropping and ROI (Region of Interest)
 * - Image flipping and rotation
 * - Drawing shapes and text
 * - Image copying and cloning
 * - Pixel access and manipulation
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat createSampleImage() {
    // Create a colorful test image
    cv::Mat image(400, 600, CV_8UC3);
    
    // Create gradient background
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (x * 255) / image.cols,
                (y * 255) / image.rows,
                128
            );
        }
    }
    
    // Add some shapes for reference
    cv::circle(image, cv::Point(150, 100), 50, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(image, cv::Point(400, 50), cv::Point(550, 200), cv::Scalar(0, 255, 0), 3);
    cv::putText(image, "Original", cv::Point(50, 350), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);
    
    return image;
}

void demonstrateResizing(const cv::Mat& original) {
    std::cout << "\n=== Image Resizing Demo ===" << std::endl;
    
    cv::Mat resized_larger, resized_smaller, resized_aspect;
    
    // 1. Resize to specific dimensions
    cv::resize(original, resized_larger, cv::Size(800, 600));
    cv::resize(original, resized_smaller, cv::Size(300, 200));
    
    // 2. Resize by scale factor
    cv::resize(original, resized_aspect, cv::Size(), 1.5, 1.5, cv::INTER_CUBIC);
    
    // 3. Different interpolation methods
    cv::Mat resized_nearest, resized_linear, resized_cubic, resized_lanczos;
    cv::Size target_size(200, 150);
    
    cv::resize(original, resized_nearest, target_size, 0, 0, cv::INTER_NEAREST);
    cv::resize(original, resized_linear, target_size, 0, 0, cv::INTER_LINEAR);
    cv::resize(original, resized_cubic, target_size, 0, 0, cv::INTER_CUBIC);
    cv::resize(original, resized_lanczos, target_size, 0, 0, cv::INTER_LANCZOS4);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Larger", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Smaller", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Scaled 1.5x", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", original);
    cv::imshow("Larger", resized_larger);
    cv::imshow("Smaller", resized_smaller);
    cv::imshow("Scaled 1.5x", resized_aspect);
    
    std::cout << "Original size: " << original.cols << "x" << original.rows << std::endl;
    std::cout << "Larger size: " << resized_larger.cols << "x" << resized_larger.rows << std::endl;
    std::cout << "Smaller size: " << resized_smaller.cols << "x" << resized_smaller.rows << std::endl;
    std::cout << "Scaled size: " << resized_aspect.cols << "x" << resized_aspect.rows << std::endl;
    
    std::cout << "Press any key to see interpolation comparison..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Show interpolation comparison
    cv::Mat comparison;
    cv::hconcat(std::vector<cv::Mat>{resized_nearest, resized_linear, resized_cubic, resized_lanczos}, comparison);
    
    cv::namedWindow("Interpolation: Nearest | Linear | Cubic | Lanczos", cv::WINDOW_AUTOSIZE);
    cv::imshow("Interpolation: Nearest | Linear | Cubic | Lanczos", comparison);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateCropping(const cv::Mat& original) {
    std::cout << "\n=== Image Cropping Demo ===" << std::endl;
    
    // 1. Simple rectangular crop using ROI
    cv::Rect crop_rect(100, 50, 300, 200);  // x, y, width, height
    cv::Mat cropped = original(crop_rect);
    
    // 2. Multiple crops
    cv::Mat top_left = original(cv::Rect(0, 0, original.cols/2, original.rows/2));
    cv::Mat top_right = original(cv::Rect(original.cols/2, 0, original.cols/2, original.rows/2));
    cv::Mat bottom_left = original(cv::Rect(0, original.rows/2, original.cols/2, original.rows/2));
    cv::Mat bottom_right = original(cv::Rect(original.cols/2, original.rows/2, original.cols/2, original.rows/2));
    
    // 3. Center crop
    int crop_size = std::min(original.cols, original.rows) / 2;
    int start_x = (original.cols - crop_size) / 2;
    int start_y = (original.rows - crop_size) / 2;
    cv::Mat center_crop = original(cv::Rect(start_x, start_y, crop_size, crop_size));
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Cropped Region", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Center Crop", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", original);
    cv::imshow("Cropped Region", cropped);
    cv::imshow("Center Crop", center_crop);
    
    // Show quadrants
    cv::Mat quadrants_top, quadrants_bottom, quadrants_combined;
    cv::hconcat(std::vector<cv::Mat>{top_left, top_right}, quadrants_top);
    cv::hconcat(std::vector<cv::Mat>{bottom_left, bottom_right}, quadrants_bottom);
    cv::vconcat(std::vector<cv::Mat>{quadrants_top, quadrants_bottom}, quadrants_combined);
    
    cv::namedWindow("Quadrants", cv::WINDOW_AUTOSIZE);
    cv::imshow("Quadrants", quadrants_combined);
    
    std::cout << "Crop rectangle: " << crop_rect << std::endl;
    std::cout << "Center crop size: " << crop_size << "x" << crop_size << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateFlipping(const cv::Mat& original) {
    std::cout << "\n=== Image Flipping Demo ===" << std::endl;
    
    cv::Mat flipped_horizontal, flipped_vertical, flipped_both;
    
    // Flip horizontally (around y-axis)
    cv::flip(original, flipped_horizontal, 1);
    
    // Flip vertically (around x-axis)
    cv::flip(original, flipped_vertical, 0);
    
    // Flip both ways (180 degree rotation)
    cv::flip(original, flipped_both, -1);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Horizontal Flip", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Vertical Flip", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Both Flips", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", original);
    cv::imshow("Horizontal Flip", flipped_horizontal);
    cv::imshow("Vertical Flip", flipped_vertical);
    cv::imshow("Both Flips", flipped_both);
    
    std::cout << "Flipping codes: 1=horizontal, 0=vertical, -1=both" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateRotation(const cv::Mat& original) {
    std::cout << "\n=== Image Rotation Demo ===" << std::endl;
    
    // Get rotation center
    cv::Point2f center(original.cols / 2.0, original.rows / 2.0);
    
    // Create rotation matrices for different angles
    cv::Mat rotation_matrix_45 = cv::getRotationMatrix2D(center, 45, 1.0);
    cv::Mat rotation_matrix_90 = cv::getRotationMatrix2D(center, 90, 1.0);
    cv::Mat rotation_matrix_neg30 = cv::getRotationMatrix2D(center, -30, 0.8);  // with scaling
    
    // Apply rotations
    cv::Mat rotated_45, rotated_90, rotated_neg30;
    cv::warpAffine(original, rotated_45, rotation_matrix_45, original.size());
    cv::warpAffine(original, rotated_90, rotation_matrix_90, original.size());
    cv::warpAffine(original, rotated_neg30, rotation_matrix_neg30, original.size());
    
    // Rotation with different background colors
    cv::Mat rotated_white, rotated_black;
    cv::warpAffine(original, rotated_white, rotation_matrix_45, original.size(), 
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    cv::warpAffine(original, rotated_black, rotation_matrix_45, original.size(), 
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    
    // Quick 90-degree rotations (more efficient)
    cv::Mat rotated_90_fast, rotated_180_fast, rotated_270_fast;
    cv::rotate(original, rotated_90_fast, cv::ROTATE_90_CLOCKWISE);
    cv::rotate(original, rotated_180_fast, cv::ROTATE_180);
    cv::rotate(original, rotated_270_fast, cv::ROTATE_90_COUNTERCLOCKWISE);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("45° Rotation", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("90° Rotation", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("-30° with 0.8x Scale", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", original);
    cv::imshow("45° Rotation", rotated_45);
    cv::imshow("90° Rotation", rotated_90);
    cv::imshow("-30° with 0.8x Scale", rotated_neg30);
    
    std::cout << "Press any key to see background color options..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    cv::namedWindow("White Background", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Black Background", cv::WINDOW_AUTOSIZE);
    cv::imshow("White Background", rotated_white);
    cv::imshow("Black Background", rotated_black);
    
    std::cout << "Press any key to see fast 90° rotations..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    cv::Mat fast_rotations;
    cv::hconcat(std::vector<cv::Mat>{rotated_90_fast, rotated_180_fast, rotated_270_fast}, fast_rotations);
    cv::namedWindow("Fast Rotations: 90° | 180° | 270°", cv::WINDOW_AUTOSIZE);
    cv::imshow("Fast Rotations: 90° | 180° | 270°", fast_rotations);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateDrawing(const cv::Mat& original) {
    std::cout << "\n=== Drawing Operations Demo ===" << std::endl;
    
    cv::Mat canvas = original.clone();
    
    // Draw various shapes
    // Line
    cv::line(canvas, cv::Point(50, 50), cv::Point(200, 100), cv::Scalar(255, 0, 0), 3);
    
    // Rectangle
    cv::rectangle(canvas, cv::Point(220, 50), cv::Point(350, 150), cv::Scalar(0, 255, 0), 2);
    cv::rectangle(canvas, cv::Point(370, 50), cv::Point(500, 150), cv::Scalar(0, 0, 255), -1);  // filled
    
    // Circle
    cv::circle(canvas, cv::Point(100, 200), 30, cv::Scalar(255, 255, 0), 2);
    cv::circle(canvas, cv::Point(200, 200), 25, cv::Scalar(255, 0, 255), -1);  // filled
    
    // Ellipse
    cv::ellipse(canvas, cv::Point(350, 200), cv::Size(60, 30), 45, 0, 360, cv::Scalar(0, 255, 255), 2);
    
    // Polygon
    std::vector<cv::Point> triangle_points = {
        cv::Point(450, 180),
        cv::Point(480, 230),
        cv::Point(420, 230)
    };
    cv::fillPoly(canvas, triangle_points, cv::Scalar(128, 128, 255));
    cv::polylines(canvas, triangle_points, true, cv::Scalar(0, 0, 0), 2);
    
    // Text with different fonts and sizes
    cv::putText(canvas, "OpenCV Drawing", cv::Point(50, 300), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(canvas, "Small text", cv::Point(50, 330), 
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 200, 200), 1);
    cv::putText(canvas, "Bold text", cv::Point(50, 360), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 3);
    
    // Arrow
    cv::arrowedLine(canvas, cv::Point(400, 300), cv::Point(500, 330), cv::Scalar(255, 128, 0), 3);
    
    // Marker
    cv::drawMarker(canvas, cv::Point(520, 200), cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 20, 2);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("With Drawings", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", original);
    cv::imshow("With Drawings", canvas);
    
    std::cout << "Various shapes and text drawn on the image." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Save the drawing example
    cv::imwrite("drawing_example.jpg", canvas);
    std::cout << "Drawing example saved as 'drawing_example.jpg'" << std::endl;
}

void demonstratePixelAccess(const cv::Mat& original) {
    std::cout << "\n=== Pixel Access Demo ===" << std::endl;
    
    cv::Mat modified = original.clone();
    
    // Method 1: Direct pixel access using at()
    for (int y = 100; y < 150; y++) {
        for (int x = 100; x < 200; x++) {
            cv::Vec3b& pixel = modified.at<cv::Vec3b>(y, x);
            pixel[0] = 255;  // Blue channel
            pixel[1] = 0;    // Green channel
            pixel[2] = 0;    // Red channel
        }
    }
    
    // Method 2: Pointer access (faster for large operations)
    for (int y = 200; y < 250; y++) {
        cv::Vec3b* row_ptr = modified.ptr<cv::Vec3b>(y);
        for (int x = 100; x < 200; x++) {
            row_ptr[x] = cv::Vec3b(0, 255, 0);  // Green
        }
    }
    
    // Method 3: Using iterators
    cv::Mat roi = modified(cv::Rect(300, 100, 100, 100));
    for (auto it = roi.begin<cv::Vec3b>(); it != roi.end<cv::Vec3b>(); ++it) {
        (*it)[0] = 0;    // Blue
        (*it)[1] = 0;    // Green
        (*it)[2] = 255;  // Red
    }
    
    // Display pixel values at specific locations
    cv::Vec3b pixel_original = original.at<cv::Vec3b>(150, 150);
    cv::Vec3b pixel_modified = modified.at<cv::Vec3b>(150, 150);
    
    std::cout << "Original pixel at (150,150): B=" << (int)pixel_original[0] 
              << " G=" << (int)pixel_original[1] 
              << " R=" << (int)pixel_original[2] << std::endl;
    std::cout << "Modified pixel at (150,150): B=" << (int)pixel_modified[0] 
              << " G=" << (int)pixel_modified[1] 
              << " R=" << (int)pixel_modified[2] << std::endl;
    
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Pixel Modified", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", original);
    cv::imshow("Pixel Modified", modified);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== OpenCV Basic Operations Demo ===" << std::endl;
    
    // Create or load test image
    cv::Mat original = createSampleImage();
    
    // Try to load a real image if available
    cv::Mat loaded_image = cv::imread("../data/test.jpg");
    if (!loaded_image.empty()) {
        original = loaded_image;
        std::cout << "Using loaded image for demonstrations." << std::endl;
    } else {
        std::cout << "Using synthetic image for demonstrations." << std::endl;
    }
    
    // Demonstrate each operation
    demonstrateResizing(original);
    demonstrateCropping(original);
    demonstrateFlipping(original);
    demonstrateRotation(original);
    demonstrateDrawing(original);
    demonstratePixelAccess(original);
    
    std::cout << "\n✓ Basic operations demonstration complete!" << std::endl;
    std::cout << "All fundamental image manipulation operations covered." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. cv::resize() with different interpolation methods for quality vs speed
 * 2. ROI (Region of Interest) for efficient cropping without copying
 * 3. cv::flip() for simple reflections
 * 4. cv::getRotationMatrix2D() + cv::warpAffine() for arbitrary rotations
 * 5. cv::rotate() for efficient 90-degree rotations
 * 6. Drawing functions for visualization and annotation
 * 7. Different pixel access methods: at(), ptr(), iterators
 * 8. Always consider performance implications of your choice
 */

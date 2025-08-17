/**
 * 01-Image_IO.cpp
 * 
 * Comprehensive image input/output operations with OpenCV.
 * Covers reading, writing, displaying images and basic format conversions.
 * 
 * Concepts covered:
 * - Image loading (imread)
 * - Image saving (imwrite)
 * - Image display (imshow)
 * - Color space conversions
 * - Image properties and metadata
 * - Error handling for I/O operations
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

void displayImageProperties(const cv::Mat& image, const std::string& name) {
    std::cout << "\n=== " << name << " Properties ===" << std::endl;
    std::cout << "Dimensions: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Channels: " << image.channels() << std::endl;
    std::cout << "Depth: " << image.depth() << " (";
    
    switch(image.depth()) {
        case CV_8U: std::cout << "8-bit unsigned"; break;
        case CV_8S: std::cout << "8-bit signed"; break;
        case CV_16U: std::cout << "16-bit unsigned"; break;
        case CV_16S: std::cout << "16-bit signed"; break;
        case CV_32S: std::cout << "32-bit signed"; break;
        case CV_32F: std::cout << "32-bit float"; break;
        case CV_64F: std::cout << "64-bit float"; break;
        default: std::cout << "unknown"; break;
    }
    std::cout << ")" << std::endl;
    
    std::cout << "Type: " << image.type() << std::endl;
    std::cout << "Size in memory: " << image.total() * image.elemSize() << " bytes" << std::endl;
    std::cout << "Continuous: " << (image.isContinuous() ? "Yes" : "No") << std::endl;
}

void demonstrateColorConversions(const cv::Mat& original) {
    std::cout << "\n=== Color Space Conversions ===" << std::endl;
    
    if (original.empty()) {
        std::cout << "No image available for color conversions." << std::endl;
        return;
    }
    
    // BGR to Grayscale
    cv::Mat gray;
    cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    
    // BGR to HSV
    cv::Mat hsv;
    cv::cvtColor(original, hsv, cv::COLOR_BGR2HSV);
    
    // BGR to LAB
    cv::Mat lab;
    cv::cvtColor(original, lab, cv::COLOR_BGR2Lab);
    
    // BGR to RGB (for display purposes)
    cv::Mat rgb;
    cv::cvtColor(original, rgb, cv::COLOR_BGR2RGB);
    
    // Display all versions
    cv::namedWindow("Original (BGR)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Grayscale", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("HSV", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("LAB", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original (BGR)", original);
    cv::imshow("Grayscale", gray);
    cv::imshow("HSV", hsv);
    cv::imshow("LAB", lab);
    
    std::cout << "Color conversions displayed. Press any key to continue..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Save converted images
    cv::imwrite("output_gray.jpg", gray);
    cv::imwrite("output_hsv.jpg", hsv);
    cv::imwrite("output_lab.jpg", lab);
    
    std::cout << "Converted images saved to current directory." << std::endl;
}

void createAndSaveSampleImages() {
    std::cout << "\n=== Creating Sample Images ===" << std::endl;
    
    // Create a gradient image
    cv::Mat gradient(400, 600, CV_8UC3);
    for (int y = 0; y < gradient.rows; y++) {
        for (int x = 0; x < gradient.cols; x++) {
            gradient.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (x * 255) / gradient.cols,           // Blue gradient
                (y * 255) / gradient.rows,           // Green gradient
                ((x + y) * 255) / (gradient.cols + gradient.rows)  // Red gradient
            );
        }
    }
    
    // Create a pattern image
    cv::Mat pattern(300, 300, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Draw checkerboard pattern
    int square_size = 30;
    for (int y = 0; y < pattern.rows; y += square_size) {
        for (int x = 0; x < pattern.cols; x += square_size) {
            if (((x / square_size) + (y / square_size)) % 2 == 0) {
                cv::rectangle(pattern, 
                             cv::Point(x, y), 
                             cv::Point(x + square_size, y + square_size),
                             cv::Scalar(0, 0, 0), -1);
            }
        }
    }
    
    // Create noise image
    cv::Mat noise(200, 200, CV_8UC1);
    cv::randu(noise, 0, 255);
    
    // Display created images
    cv::namedWindow("Gradient", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Checkerboard Pattern", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Random Noise", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Gradient", gradient);
    cv::imshow("Checkerboard Pattern", pattern);
    cv::imshow("Random Noise", noise);
    
    std::cout << "Sample images created and displayed. Press any key to continue..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Save with different formats and quality settings
    cv::imwrite("sample_gradient.jpg", gradient);
    cv::imwrite("sample_pattern.png", pattern);
    cv::imwrite("sample_noise.bmp", noise);
    
    // Save JPEG with different quality levels
    std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, 95};
    cv::imwrite("sample_gradient_high_quality.jpg", gradient, jpeg_params);
    
    jpeg_params[1] = 30;
    cv::imwrite("sample_gradient_low_quality.jpg", gradient, jpeg_params);
    
    // Save PNG with compression
    std::vector<int> png_params = {cv::IMWRITE_PNG_COMPRESSION, 9};
    cv::imwrite("sample_pattern_compressed.png", pattern, png_params);
    
    std::cout << "Sample images saved with various formats and quality settings." << std::endl;
}

int main() {
    std::cout << "=== OpenCV Image I/O Demo ===" << std::endl;
    
    // 1. Try to load an existing image
    std::vector<std::string> test_paths = {
        "../data/test.jpg",
        "../data/lena.jpg", 
        "../data/sample.png",
        "test.jpg",
        "sample.png"
    };
    
    cv::Mat loaded_image;
    std::string successful_path;
    
    for (const auto& path : test_paths) {
        loaded_image = cv::imread(path, cv::IMREAD_COLOR);
        if (!loaded_image.empty()) {
            successful_path = path;
            std::cout << "Successfully loaded image: " << path << std::endl;
            break;
        }
    }
    
    if (loaded_image.empty()) {
        std::cout << "No test images found. Creating synthetic images instead." << std::endl;
        
        // Create a synthetic test image
        loaded_image = cv::Mat::zeros(400, 600, CV_8UC3);
        
        // Add some content
        cv::rectangle(loaded_image, cv::Point(50, 50), cv::Point(550, 350), cv::Scalar(100, 150, 200), -1);
        cv::circle(loaded_image, cv::Point(300, 200), 80, cv::Scalar(255, 255, 255), -1);
        cv::circle(loaded_image, cv::Point(300, 200), 60, cv::Scalar(0, 0, 0), -1);
        cv::putText(loaded_image, "OpenCV I/O Test", cv::Point(150, 120), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
        
        successful_path = "synthetic_image";
    }
    
    // Display image properties
    displayImageProperties(loaded_image, "Loaded Image");
    
    // Show the original image
    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original Image", loaded_image);
    
    std::cout << "\nPress any key to continue with color conversions..." << std::endl;
    cv::waitKey(0);
    cv::destroyWindow("Original Image");
    
    // 2. Demonstrate color conversions
    demonstrateColorConversions(loaded_image);
    
    // 3. Create and save sample images
    createAndSaveSampleImages();
    
    // 4. Demonstrate different loading modes
    std::cout << "\n=== Different Loading Modes ===" << std::endl;
    if (!successful_path.empty() && successful_path != "synthetic_image") {
        cv::Mat color_image = cv::imread(successful_path, cv::IMREAD_COLOR);
        cv::Mat gray_image = cv::imread(successful_path, cv::IMREAD_GRAYSCALE);
        cv::Mat unchanged_image = cv::imread(successful_path, cv::IMREAD_UNCHANGED);
        
        std::cout << "Color mode - Channels: " << color_image.channels() << std::endl;
        std::cout << "Grayscale mode - Channels: " << gray_image.channels() << std::endl;
        std::cout << "Unchanged mode - Channels: " << unchanged_image.channels() << std::endl;
        
        cv::namedWindow("Color Load", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Grayscale Load", cv::WINDOW_AUTOSIZE);
        
        cv::imshow("Color Load", color_image);
        cv::imshow("Grayscale Load", gray_image);
        
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    
    // 5. Save with metadata
    std::cout << "\n=== Saving with Metadata ===" << std::endl;
    
    // Create an image with metadata
    cv::Mat metadata_image = loaded_image.clone();
    cv::putText(metadata_image, "Saved: " + std::string(__DATE__), cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // Save with different parameters
    std::vector<int> tiff_params = {cv::IMWRITE_TIFF_COMPRESSION, 1};
    cv::imwrite("output_with_metadata.tiff", metadata_image, tiff_params);
    
    std::cout << "Image saved with metadata to 'output_with_metadata.tiff'" << std::endl;
    
    std::cout << "\n✓ Image I/O demonstration complete!" << std::endl;
    std::cout << "Check the current directory for output images." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. cv::imread() loads images in BGR format by default
 * 2. Different loading modes: COLOR, GRAYSCALE, UNCHANGED
 * 3. cv::imwrite() can save in various formats with quality parameters
 * 4. Color conversions are essential for different processing tasks
 * 5. Image properties like dimensions, channels, and data type are crucial
 * 6. Always check if image loading was successful (!image.empty())
 */

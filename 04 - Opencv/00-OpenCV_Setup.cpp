/**
 * 00-OpenCV_Setup.cpp
 * 
 * Basic OpenCV setup verification and simple image display.
 * This file tests your OpenCV installation and demonstrates basic functionality.
 * 
 * Concepts covered:
 * - OpenCV version checking
 * - Basic image loading and display
 * - Window management
 * - Error handling
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main() {
    std::cout << "=== OpenCV Setup Test ===" << std::endl;
    
    // Display OpenCV version
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "Major Version: " << CV_MAJOR_VERSION << std::endl;
    std::cout << "Minor Version: " << CV_MINOR_VERSION << std::endl;
    
    // Check available modules
    std::cout << "\nAvailable modules:" << std::endl;
    std::vector<std::string> modules;
    cv::getBuildInformation();
    
    // Create a simple test image (synthetic)
    cv::Mat test_image = cv::Mat::zeros(300, 400, CV_8UC3);
    
    // Draw some basic shapes to verify functionality
    cv::rectangle(test_image, cv::Point(50, 50), cv::Point(350, 250), cv::Scalar(0, 255, 0), 2);
    cv::circle(test_image, cv::Point(200, 150), 50, cv::Scalar(255, 0, 0), -1);
    cv::putText(test_image, "OpenCV Working!", cv::Point(80, 100), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // Display the test image
    cv::namedWindow("OpenCV Setup Test", cv::WINDOW_AUTOSIZE);
    cv::imshow("OpenCV Setup Test", test_image);
    
    std::cout << "\nTest image created and displayed successfully!" << std::endl;
    std::cout << "Press any key to continue..." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Test loading a real image if available
    std::string image_path = "../data/test.jpg";  // Adjust path as needed
    cv::Mat loaded_image = cv::imread(image_path);
    
    if (!loaded_image.empty()) {
        std::cout << "Successfully loaded image: " << image_path << std::endl;
        std::cout << "Image dimensions: " << loaded_image.cols << "x" << loaded_image.rows << std::endl;
        std::cout << "Channels: " << loaded_image.channels() << std::endl;
        
        cv::namedWindow("Loaded Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Loaded Image", loaded_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    } else {
        std::cout << "Note: Could not load test image from " << image_path << std::endl;
        std::cout << "This is normal if you haven't added test images yet." << std::endl;
    }
    
    // Display build information
    std::cout << "\n=== Build Information ===" << std::endl;
    std::string build_info = cv::getBuildInformation();
    
    // Extract and display key information
    size_t cuda_pos = build_info.find("CUDA:");
    if (cuda_pos != std::string::npos) {
        size_t end_pos = build_info.find("\n", cuda_pos);
        std::string cuda_info = build_info.substr(cuda_pos, end_pos - cuda_pos);
        std::cout << cuda_info << std::endl;
    }
    
    size_t python_pos = build_info.find("Python");
    if (python_pos != std::string::npos) {
        size_t end_pos = build_info.find("\n", python_pos);
        if (end_pos != std::string::npos) {
            std::string python_info = build_info.substr(python_pos, end_pos - python_pos);
            std::cout << python_info << std::endl;
        }
    }
    
    std::cout << "\n✓ OpenCV setup verification complete!" << std::endl;
    std::cout << "All basic functionality is working correctly." << std::endl;
    
    return 0;
}

/**
 * Expected Output:
 * - OpenCV version information
 * - Test image with rectangle, circle, and text
 * - Information about loaded image (if available)
 * - Build configuration details
 * 
 * If this runs successfully, your OpenCV installation is working correctly.
 */

/**
 * 03-Image_Arithmetic.cpp
 * 
 * Image arithmetic operations and blending techniques.
 * Essential for combining images, creating effects, and preprocessing.
 * 
 * Concepts covered:
 * - Basic arithmetic operations (add, subtract, multiply, divide)
 * - Image blending and alpha compositing
 * - Bitwise operations
 * - Absolute difference
 * - Weighted addition
 * - Logical operations with masks
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat createGradientImage(int width, int height, int direction = 0) {
    cv::Mat gradient(height, width, CV_8UC3);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (direction == 0) {  // Horizontal gradient
                uchar value = (x * 255) / width;
                gradient.at<cv::Vec3b>(y, x) = cv::Vec3b(value, value, value);
            } else if (direction == 1) {  // Vertical gradient
                uchar value = (y * 255) / height;
                gradient.at<cv::Vec3b>(y, x) = cv::Vec3b(value, value, value);
            } else {  // Diagonal gradient
                uchar value = ((x + y) * 255) / (width + height);
                gradient.at<cv::Vec3b>(y, x) = cv::Vec3b(value, value, value);
            }
        }
    }
    return gradient;
}

cv::Mat createColorPattern(int width, int height) {
    cv::Mat pattern(height, width, CV_8UC3);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            pattern.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (x * 255) / width,           // Blue channel
                (y * 255) / height,          // Green channel
                ((x + y) * 255) / (width + height)  // Red channel
            );
        }
    }
    return pattern;
}

void demonstrateBasicArithmetic() {
    std::cout << "\n=== Basic Arithmetic Operations ===" << std::endl;
    
    // Create test images
    cv::Mat img1 = createGradientImage(400, 300, 0);  // Horizontal gradient
    cv::Mat img2 = createGradientImage(400, 300, 1);  // Vertical gradient
    
    cv::Mat added, subtracted, multiplied, divided;
    cv::Mat added_saturated, subtracted_saturated;
    
    // Basic arithmetic operations
    cv::add(img1, img2, added);
    cv::subtract(img1, img2, subtracted);
    cv::multiply(img1, img2, multiplied, 1.0/255.0);  // Scale to prevent overflow
    cv::divide(img1, img2 + 1, divided);  // +1 to avoid division by zero
    
    // Saturated arithmetic (clips values at 0 and 255)
    added_saturated = img1 + img2;
    subtracted_saturated = img1 - img2;
    
    // Display results
    cv::namedWindow("Image 1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Image 2", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Added", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Subtracted", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Multiplied", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Divided", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Image 1", img1);
    cv::imshow("Image 2", img2);
    cv::imshow("Added", added);
    cv::imshow("Subtracted", subtracted);
    cv::imshow("Multiplied", multiplied);
    cv::imshow("Divided", divided);
    
    std::cout << "Basic arithmetic operations displayed." << std::endl;
    std::cout << "Note: multiply operation scaled by 1/255 to prevent overflow." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Show difference between normal and saturated arithmetic
    cv::namedWindow("cv::add() Result", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Operator + Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("cv::add() Result", added);
    cv::imshow("Operator + Result", added_saturated);
    
    std::cout << "Press any key to see the difference between cv::add() and operator+..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateBlending() {
    std::cout << "\n=== Image Blending Demo ===" << std::endl;
    
    // Create two different colored patterns
    cv::Mat pattern1 = createColorPattern(400, 300);
    cv::Mat pattern2(300, 400, CV_8UC3);
    cv::circle(pattern2, cv::Point(200, 150), 100, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(pattern2, cv::Point(50, 50), cv::Point(350, 250), cv::Scalar(255, 0, 0), 3);
    
    // Different blending ratios
    cv::Mat blend_25, blend_50, blend_75;
    cv::addWeighted(pattern1, 0.75, pattern2, 0.25, 0, blend_25);
    cv::addWeighted(pattern1, 0.5, pattern2, 0.5, 0, blend_50);
    cv::addWeighted(pattern1, 0.25, pattern2, 0.75, 0, blend_75);
    
    // Blending with scalar addition
    cv::Mat blend_bright, blend_dark;
    cv::addWeighted(pattern1, 0.7, pattern2, 0.3, 50, blend_bright);   // +50 brightness
    cv::addWeighted(pattern1, 0.7, pattern2, 0.3, -50, blend_dark);    // -50 brightness
    
    // Create animated blending sequence
    std::vector<cv::Mat> blend_sequence;
    for (int i = 0; i <= 10; i++) {
        cv::Mat blended;
        double alpha = i / 10.0;
        cv::addWeighted(pattern1, 1.0 - alpha, pattern2, alpha, 0, blended);
        blend_sequence.push_back(blended);
    }
    
    // Display static blends
    cv::namedWindow("Pattern 1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Pattern 2", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("25% Pattern2", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("50% Blend", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("75% Pattern2", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Pattern 1", pattern1);
    cv::imshow("Pattern 2", pattern2);
    cv::imshow("25% Pattern2", blend_25);
    cv::imshow("50% Blend", blend_50);
    cv::imshow("75% Pattern2", blend_75);
    
    std::cout << "Different blending ratios displayed." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Show brightness adjustment
    cv::namedWindow("Brighter Blend", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Darker Blend", cv::WINDOW_AUTOSIZE);
    cv::imshow("Brighter Blend", blend_bright);
    cv::imshow("Darker Blend", blend_dark);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Animated blending
    cv::namedWindow("Animated Blend", cv::WINDOW_AUTOSIZE);
    std::cout << "Showing animated blending sequence..." << std::endl;
    
    for (int cycle = 0; cycle < 3; cycle++) {
        for (const auto& frame : blend_sequence) {
            cv::imshow("Animated Blend", frame);
            cv::waitKey(100);
        }
        for (int i = blend_sequence.size() - 2; i > 0; i--) {
            cv::imshow("Animated Blend", blend_sequence[i]);
            cv::waitKey(100);
        }
    }
    cv::destroyWindow("Animated Blend");
}

void demonstrateBitwiseOperations() {
    std::cout << "\n=== Bitwise Operations Demo ===" << std::endl;
    
    // Create binary masks
    cv::Mat mask1 = cv::Mat::zeros(300, 400, CV_8UC1);
    cv::Mat mask2 = cv::Mat::zeros(300, 400, CV_8UC1);
    
    // Create shapes in masks
    cv::circle(mask1, cv::Point(150, 150), 80, cv::Scalar(255), -1);
    cv::rectangle(mask2, cv::Point(100, 100), cv::Point(300, 200), cv::Scalar(255), -1);
    
    // Bitwise operations
    cv::Mat mask_and, mask_or, mask_xor, mask_not1;
    cv::bitwise_and(mask1, mask2, mask_and);
    cv::bitwise_or(mask1, mask2, mask_or);
    cv::bitwise_xor(mask1, mask2, mask_xor);
    cv::bitwise_not(mask1, mask_not1);
    
    // Apply masks to colored image
    cv::Mat colored_image = createColorPattern(400, 300);
    cv::Mat masked_result;
    cv::bitwise_and(colored_image, colored_image, masked_result, mask_and);
    
    // Display results
    cv::namedWindow("Mask 1 (Circle)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Mask 2 (Rectangle)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("AND", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("OR", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("XOR", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("NOT Mask1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Masked Color Image", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Mask 1 (Circle)", mask1);
    cv::imshow("Mask 2 (Rectangle)", mask2);
    cv::imshow("AND", mask_and);
    cv::imshow("OR", mask_or);
    cv::imshow("XOR", mask_xor);
    cv::imshow("NOT Mask1", mask_not1);
    cv::imshow("Masked Color Image", masked_result);
    
    std::cout << "Bitwise operations on binary masks displayed." << std::endl;
    std::cout << "AND = intersection, OR = union, XOR = difference, NOT = inversion" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateAbsoluteDifference() {
    std::cout << "\n=== Absolute Difference Demo ===" << std::endl;
    
    // Create two similar images with slight differences
    cv::Mat img1 = createColorPattern(400, 300);
    cv::Mat img2 = img1.clone();
    
    // Add some differences to img2
    cv::circle(img2, cv::Point(150, 150), 50, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(img2, cv::Point(250, 100), cv::Point(350, 200), cv::Scalar(0, 0, 0), -1);
    
    // Calculate absolute difference
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    
    // Threshold the difference to create binary mask of changes
    cv::Mat diff_gray, diff_binary;
    cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(diff_gray, diff_binary, 30, 255, cv::THRESH_BINARY);
    
    // Create enhanced difference visualization
    cv::Mat diff_colored;
    cv::applyColorMap(diff_gray, diff_colored, cv::COLORMAP_JET);
    
    // Motion detection simulation
    cv::Mat motion_highlighted = img1.clone();
    motion_highlighted.setTo(cv::Scalar(0, 0, 255), diff_binary);  // Highlight changes in red
    cv::addWeighted(img1, 0.7, motion_highlighted, 0.3, 0, motion_highlighted);
    
    // Display results
    cv::namedWindow("Image 1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Image 2", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Absolute Difference", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Difference Binary", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Difference Colored", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Motion Highlighted", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Image 1", img1);
    cv::imshow("Image 2", img2);
    cv::imshow("Absolute Difference", diff);
    cv::imshow("Difference Binary", diff_binary);
    cv::imshow("Difference Colored", diff_colored);
    cv::imshow("Motion Highlighted", motion_highlighted);
    
    std::cout << "Absolute difference useful for change detection and motion analysis." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateAdvancedArithmetic() {
    std::cout << "\n=== Advanced Arithmetic Operations ===" << std::endl;
    
    cv::Mat img = createColorPattern(300, 300);
    
    // Scalar operations
    cv::Mat brighter, darker, scaled, offset;
    cv::add(img, cv::Scalar(50, 50, 50), brighter);      // Add constant
    cv::subtract(img, cv::Scalar(50, 50, 50), darker);   // Subtract constant
    cv::multiply(img, cv::Scalar(1.5, 1.2, 0.8), scaled); // Scale channels differently
    img.convertTo(offset, -1, 1.2, 30);                  // Linear transform: out = 1.2*in + 30
    
    // Min/Max operations
    cv::Mat min_result, max_result;
    cv::min(img, cv::Scalar(128, 128, 128), min_result);
    cv::max(img, cv::Scalar(128, 128, 128), max_result);
    
    // Power and square root
    cv::Mat normalized, powered, sqrt_result;
    img.convertTo(normalized, CV_32F, 1.0/255.0);  // Normalize to 0-1
    cv::pow(normalized, 2.0, powered);              // Square
    cv::sqrt(normalized, sqrt_result);              // Square root
    
    powered.convertTo(powered, CV_8U, 255.0);       // Convert back to 8-bit
    sqrt_result.convertTo(sqrt_result, CV_8U, 255.0);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Brighter (+50)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Darker (-50)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Scaled Channels", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Linear Transform", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Min with 128", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Max with 128", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Powered (x²)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Square Root", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", img);
    cv::imshow("Brighter (+50)", brighter);
    cv::imshow("Darker (-50)", darker);
    cv::imshow("Scaled Channels", scaled);
    cv::imshow("Linear Transform", offset);
    cv::imshow("Min with 128", min_result);
    cv::imshow("Max with 128", max_result);
    cv::imshow("Powered (x²)", powered);
    cv::imshow("Square Root", sqrt_result);
    
    std::cout << "Advanced arithmetic operations for image enhancement and processing." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== OpenCV Image Arithmetic Demo ===" << std::endl;
    
    demonstrateBasicArithmetic();
    demonstrateBlending();
    demonstrateBitwiseOperations();
    demonstrateAbsoluteDifference();
    demonstrateAdvancedArithmetic();
    
    std::cout << "\n✓ Image arithmetic demonstration complete!" << std::endl;
    std::cout << "These operations are fundamental for image processing pipelines." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. cv::add(), cv::subtract() handle overflow/underflow properly
 * 2. Operator +, - perform saturated arithmetic (clipping)
 * 3. cv::addWeighted() is essential for blending and transparency effects
 * 4. Bitwise operations work on binary masks for region selection
 * 5. cv::absdiff() is crucial for change/motion detection
 * 6. Scalar operations allow per-channel manipulation
 * 7. Always consider data type conversions for mathematical operations
 * 8. Linear transforms: convertTo() with alpha and beta parameters
 */

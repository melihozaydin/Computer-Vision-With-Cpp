/**
 * 15-Color_Spaces.cpp
 * 
 * Working with different color spaces for image analysis.
 * 
 * Concepts covered:
 * - BGR, RGB, HSV conversion
 * - LAB and YUV color spaces
 * - Color masking
 * - White balance
 * - Color constancy
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

cv::Mat createColorTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // Create color patches for testing different color spaces
    int patch_size = 60;
    int start_x = 50, start_y = 50;
    
    // Primary colors
    cv::rectangle(image, cv::Point(start_x, start_y), 
                 cv::Point(start_x + patch_size, start_y + patch_size), 
                 cv::Scalar(0, 0, 255), -1);  // Red
    cv::rectangle(image, cv::Point(start_x + patch_size + 10, start_y), 
                 cv::Point(start_x + 2*patch_size + 10, start_y + patch_size), 
                 cv::Scalar(0, 255, 0), -1);  // Green
    cv::rectangle(image, cv::Point(start_x + 2*(patch_size + 10), start_y), 
                 cv::Point(start_x + 3*patch_size + 20, start_y + patch_size), 
                 cv::Scalar(255, 0, 0), -1);  // Blue
    
    // Secondary colors
    start_y += patch_size + 20;
    cv::rectangle(image, cv::Point(start_x, start_y), 
                 cv::Point(start_x + patch_size, start_y + patch_size), 
                 cv::Scalar(0, 255, 255), -1);  // Yellow
    cv::rectangle(image, cv::Point(start_x + patch_size + 10, start_y), 
                 cv::Point(start_x + 2*patch_size + 10, start_y + patch_size), 
                 cv::Scalar(255, 0, 255), -1);  // Magenta
    cv::rectangle(image, cv::Point(start_x + 2*(patch_size + 10), start_y), 
                 cv::Point(start_x + 3*patch_size + 20, start_y + patch_size), 
                 cv::Scalar(255, 255, 0), -1);  // Cyan
    
    // Grayscale patches
    start_y += patch_size + 20;
    for (int i = 0; i < 6; i++) {
        int gray_val = i * 50;
        cv::rectangle(image, cv::Point(start_x + i * (patch_size/2 + 5), start_y), 
                     cv::Point(start_x + (i+1) * (patch_size/2 + 5) - 5, start_y + patch_size/2), 
                     cv::Scalar(gray_val, gray_val, gray_val), -1);
    }
    
    // Color gradient
    start_y += patch_size/2 + 20;
    for (int x = 0; x < 300; x++) {
        for (int y = 0; y < 30; y++) {
            int hue = (x * 180) / 300;  // HSV hue range 0-179
            cv::Vec3b hsv_color(hue, 255, 255);
            cv::Mat hsv_pixel(1, 1, CV_8UC3, hsv_color);
            cv::Mat bgr_pixel;
            cv::cvtColor(hsv_pixel, bgr_pixel, cv::COLOR_HSV2BGR);
            cv::Vec3b bgr_color = bgr_pixel.at<cv::Vec3b>(0, 0);
            image.at<cv::Vec3b>(start_y + y, start_x + x) = bgr_color;
        }
    }
    
    // Add some objects with distinct colors
    cv::circle(image, cv::Point(450, 150), 40, cv::Scalar(0, 128, 255), -1);  // Orange circle
    cv::rectangle(image, cv::Point(420, 250), cv::Point(480, 310), cv::Scalar(128, 0, 128), -1);  // Purple rectangle
    
    return image;
}

void demonstrateBasicColorConversions(const cv::Mat& src) {
    std::cout << "\n=== Basic Color Space Conversions ===" << std::endl;
    
    cv::Mat hsv, lab, yuv, gray;
    cv::Mat rgb;
    
    // Convert BGR to other color spaces
    cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    cv::cvtColor(src, yuv, cv::COLOR_BGR2YUV);
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Create visualization grid
    cv::Mat result = cv::Mat::zeros(src.rows * 2, src.cols * 3, CV_8UC3);
    
    // Top row: BGR, RGB, HSV
    src.copyTo(result(cv::Rect(0, 0, src.cols, src.rows)));
    rgb.copyTo(result(cv::Rect(src.cols, 0, src.cols, src.rows)));
    hsv.copyTo(result(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Bottom row: LAB, YUV, Gray (converted to 3-channel for display)
    lab.copyTo(result(cv::Rect(0, src.rows, src.cols, src.rows)));
    yuv.copyTo(result(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    cv::Mat gray_3ch;
    cv::cvtColor(gray, gray_3ch, cv::COLOR_GRAY2BGR);
    gray_3ch.copyTo(result(cv::Rect(src.cols * 2, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(result, "BGR (Original)", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "RGB", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "HSV", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "LAB", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "YUV", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "Grayscale", cv::Point(src.cols * 2 + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Color Space Conversions", cv::WINDOW_AUTOSIZE);
    cv::imshow("Color Space Conversions", result);
    
    std::cout << "Color space conversion complete:" << std::endl;
    std::cout << "  BGR: Blue-Green-Red (OpenCV default)" << std::endl;
    std::cout << "  RGB: Red-Green-Blue (standard)" << std::endl;
    std::cout << "  HSV: Hue-Saturation-Value (perceptual)" << std::endl;
    std::cout << "  LAB: Lightness-A-B (perceptually uniform)" << std::endl;
    std::cout << "  YUV: Luminance-Chrominance (video)" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateHSVChannelSeparation(const cv::Mat& src) {
    std::cout << "\n=== HSV Channel Separation ===" << std::endl;
    
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    // Split HSV channels
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    
    cv::Mat hue = hsv_channels[0];
    cv::Mat saturation = hsv_channels[1];
    cv::Mat value = hsv_channels[2];
    
    // Create colored visualizations
    cv::Mat hue_colored, sat_colored, val_colored;
    
    // Hue: convert back to BGR with full saturation and value
    cv::Mat hue_hsv;
    cv::merge(std::vector<cv::Mat>{hue, cv::Mat::ones(hue.size(), CV_8UC1) * 255, cv::Mat::ones(hue.size(), CV_8UC1) * 255}, hue_hsv);
    cv::cvtColor(hue_hsv, hue_colored, cv::COLOR_HSV2BGR);
    
    // Saturation: grayscale representation
    cv::cvtColor(saturation, sat_colored, cv::COLOR_GRAY2BGR);
    
    // Value: grayscale representation
    cv::cvtColor(value, val_colored, cv::COLOR_GRAY2BGR);
    
    // Create comparison grid
    cv::Mat comparison = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    src.copyTo(comparison(cv::Rect(0, 0, src.cols, src.rows)));
    hue_colored.copyTo(comparison(cv::Rect(src.cols, 0, src.cols, src.rows)));
    sat_colored.copyTo(comparison(cv::Rect(0, src.rows, src.cols, src.rows)));
    val_colored.copyTo(comparison(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(comparison, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Hue", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Saturation", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Value", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("HSV Channel Separation", cv::WINDOW_AUTOSIZE);
    cv::imshow("HSV Channel Separation", comparison);
    
    // Calculate and display statistics
    cv::Scalar hue_stats = cv::mean(hue);
    cv::Scalar sat_stats = cv::mean(saturation);
    cv::Scalar val_stats = cv::mean(value);
    
    std::cout << "HSV channel statistics:" << std::endl;
    std::cout << "  Average Hue: " << hue_stats[0] << " (0-179 scale)" << std::endl;
    std::cout << "  Average Saturation: " << sat_stats[0] << " (0-255 scale)" << std::endl;
    std::cout << "  Average Value: " << val_stats[0] << " (0-255 scale)" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateColorMasking(const cv::Mat& src) {
    std::cout << "\n=== Color Masking ===" << std::endl;
    
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    // Define color ranges for different objects
    // Red color range (handle wrap-around in hue)
    cv::Mat red_mask1, red_mask2, red_mask;
    cv::inRange(hsv, cv::Scalar(0, 50, 50), cv::Scalar(10, 255, 255), red_mask1);
    cv::inRange(hsv, cv::Scalar(170, 50, 50), cv::Scalar(179, 255, 255), red_mask2);
    red_mask = red_mask1 | red_mask2;
    
    // Blue color range
    cv::Mat blue_mask;
    cv::inRange(hsv, cv::Scalar(100, 50, 50), cv::Scalar(130, 255, 255), blue_mask);
    
    // Green color range
    cv::Mat green_mask;
    cv::inRange(hsv, cv::Scalar(40, 50, 50), cv::Scalar(80, 255, 255), green_mask);
    
    // Yellow color range
    cv::Mat yellow_mask;
    cv::inRange(hsv, cv::Scalar(20, 50, 50), cv::Scalar(30, 255, 255), yellow_mask);
    
    // Apply masks to original image
    cv::Mat red_result, blue_result, green_result, yellow_result;
    src.copyTo(red_result, red_mask);
    src.copyTo(blue_result, blue_mask);
    src.copyTo(green_result, green_mask);
    src.copyTo(yellow_result, yellow_mask);
    
    // Create visualization grid
    cv::Mat result = cv::Mat::zeros(src.rows * 2, src.cols * 3, CV_8UC3);
    
    // Top row: original and masks
    src.copyTo(result(cv::Rect(0, 0, src.cols, src.rows)));
    cv::Mat all_masks = red_mask | blue_mask | green_mask | yellow_mask;
    cv::Mat all_masks_colored;
    cv::cvtColor(all_masks, all_masks_colored, cv::COLOR_GRAY2BGR);
    all_masks_colored.copyTo(result(cv::Rect(src.cols, 0, src.cols, src.rows)));
    
    // Bottom row: individual color extractions
    cv::Mat combined_colors = red_result + blue_result + green_result + yellow_result;
    combined_colors.copyTo(result(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    red_result.copyTo(result(cv::Rect(0, src.rows, src.cols, src.rows)));
    blue_result.copyTo(result(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    green_result.copyTo(result(cv::Rect(src.cols * 2, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(result, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "All Masks", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "Combined Colors", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "Red Objects", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "Blue Objects", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "Green Objects", cv::Point(src.cols * 2 + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Color Masking", cv::WINDOW_AUTOSIZE);
    cv::imshow("Color Masking", result);
    
    // Count pixels for each color
    int red_pixels = cv::countNonZero(red_mask);
    int blue_pixels = cv::countNonZero(blue_mask);
    int green_pixels = cv::countNonZero(green_mask);
    int yellow_pixels = cv::countNonZero(yellow_mask);
    
    std::cout << "Color masking results:" << std::endl;
    std::cout << "  Red pixels: " << red_pixels << std::endl;
    std::cout << "  Blue pixels: " << blue_pixels << std::endl;
    std::cout << "  Green pixels: " << green_pixels << std::endl;
    std::cout << "  Yellow pixels: " << yellow_pixels << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateLABColorSpace(const cv::Mat& src) {
    std::cout << "\n=== LAB Color Space Analysis ===" << std::endl;
    
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    
    // Split LAB channels
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);
    
    cv::Mat L = lab_channels[0];  // Lightness
    cv::Mat A = lab_channels[1];  // Green-Red axis
    cv::Mat B = lab_channels[2];  // Blue-Yellow axis
    
    // Normalize A and B channels for better visualization
    cv::Mat A_norm, B_norm;
    cv::normalize(A, A_norm, 0, 255, cv::NORM_MINMAX);
    cv::normalize(B, B_norm, 0, 255, cv::NORM_MINMAX);
    
    // Create colored visualizations
    cv::Mat L_colored, A_colored, B_colored;
    cv::cvtColor(L, L_colored, cv::COLOR_GRAY2BGR);
    cv::cvtColor(A_norm, A_colored, cv::COLOR_GRAY2BGR);
    cv::cvtColor(B_norm, B_colored, cv::COLOR_GRAY2BGR);
    
    // Create comparison
    cv::Mat comparison = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    src.copyTo(comparison(cv::Rect(0, 0, src.cols, src.rows)));
    L_colored.copyTo(comparison(cv::Rect(src.cols, 0, src.cols, src.rows)));
    A_colored.copyTo(comparison(cv::Rect(0, src.rows, src.cols, src.rows)));
    B_colored.copyTo(comparison(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(comparison, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "L (Lightness)", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "A (Green-Red)", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "B (Blue-Yellow)", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("LAB Color Space", cv::WINDOW_AUTOSIZE);
    cv::imshow("LAB Color Space", comparison);
    
    // Calculate statistics
    cv::Scalar L_stats = cv::mean(L);
    cv::Scalar A_stats = cv::mean(A);
    cv::Scalar B_stats = cv::mean(B);
    
    std::cout << "LAB color space statistics:" << std::endl;
    std::cout << "  Average Lightness (L): " << L_stats[0] << " (0-100 range)" << std::endl;
    std::cout << "  Average A (Green-Red): " << A_stats[0] << " (128=neutral)" << std::endl;
    std::cout << "  Average B (Blue-Yellow): " << B_stats[0] << " (128=neutral)" << std::endl;
    std::cout << "LAB is perceptually uniform - equal distances represent equal perceptual differences." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateColorConstancy(const cv::Mat& src) {
    std::cout << "\n=== Color Constancy and White Balance ===" << std::endl;
    
    // Simulate different lighting conditions
    cv::Mat warm_light = src.clone();
    cv::Mat cool_light = src.clone();
    cv::Mat green_light = src.clone();
    
    // Apply color casts
    std::vector<cv::Mat> bgr_channels;
    cv::split(src, bgr_channels);
    
    // Warm light (more red/yellow)
    std::vector<cv::Mat> warm_channels = bgr_channels;
    warm_channels[2] = warm_channels[2] * 1.3;  // Increase red
    warm_channels[1] = warm_channels[1] * 1.1;  // Slightly increase green
    cv::merge(warm_channels, warm_light);
    
    // Cool light (more blue)
    std::vector<cv::Mat> cool_channels = bgr_channels;
    cool_channels[0] = cool_channels[0] * 1.3;  // Increase blue
    cv::merge(cool_channels, cool_light);
    
    // Green light
    std::vector<cv::Mat> green_channels = bgr_channels;
    green_channels[1] = green_channels[1] * 1.4;  // Increase green
    cv::merge(green_channels, green_light);
    
    // Simple white balance correction using gray world assumption
    auto whiteBalance = [](const cv::Mat& img) -> cv::Mat {
        cv::Mat result;
        std::vector<cv::Mat> channels;
        cv::split(img, channels);
        
        cv::Scalar means = cv::mean(img);
        double avg_mean = (means[0] + means[1] + means[2]) / 3.0;
        
        for (int i = 0; i < 3; i++) {
            if (means[i] > 0) {
                channels[i] = channels[i] * (avg_mean / means[i]);
            }
        }
        
        cv::merge(channels, result);
        return result;
    };
    
    // Apply white balance correction
    cv::Mat warm_corrected = whiteBalance(warm_light);
    cv::Mat cool_corrected = whiteBalance(cool_light);
    cv::Mat green_corrected = whiteBalance(green_light);
    
    // Create visualization
    cv::Mat result = cv::Mat::zeros(src.rows * 2, src.cols * 4, CV_8UC3);
    
    // Top row: original lighting conditions
    src.copyTo(result(cv::Rect(0, 0, src.cols, src.rows)));
    warm_light.copyTo(result(cv::Rect(src.cols, 0, src.cols, src.rows)));
    cool_light.copyTo(result(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    green_light.copyTo(result(cv::Rect(src.cols * 3, 0, src.cols, src.rows)));
    
    // Bottom row: white balance corrected
    src.copyTo(result(cv::Rect(0, src.rows, src.cols, src.rows)));
    warm_corrected.copyTo(result(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    cool_corrected.copyTo(result(cv::Rect(src.cols * 2, src.rows, src.cols, src.rows)));
    green_corrected.copyTo(result(cv::Rect(src.cols * 3, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(result, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "Warm Light", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "Cool Light", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "Green Light", cv::Point(src.cols * 3 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    cv::putText(result, "Corrected", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "WB Corrected", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "WB Corrected", cv::Point(src.cols * 2 + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "WB Corrected", cv::Point(src.cols * 3 + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Color Constancy", cv::WINDOW_AUTOSIZE);
    cv::imshow("Color Constancy", result);
    
    std::cout << "Color constancy demonstration:" << std::endl;
    std::cout << "  Applied different lighting conditions to simulate real-world scenarios" << std::endl;
    std::cout << "  Gray world white balance attempts to correct color casts" << std::endl;
    std::cout << "  Top row: Various lighting conditions" << std::endl;
    std::cout << "  Bottom row: White balance corrected versions" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Color Spaces ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createColorTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg");
    if (!real_image.empty()) {
        std::cout << "Using loaded image for some demonstrations." << std::endl;
        
        // Use real image for most demonstrations
        demonstrateBasicColorConversions(real_image);
        demonstrateHSVChannelSeparation(real_image);
        demonstrateColorMasking(test_image);        // Use synthetic for clear color patches
        demonstrateLABColorSpace(real_image);
        demonstrateColorConstancy(real_image);
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all color space operations
        demonstrateBasicColorConversions(test_image);
        demonstrateHSVChannelSeparation(test_image);
        demonstrateColorMasking(test_image);
        demonstrateLABColorSpace(test_image);
        demonstrateColorConstancy(test_image);
    }
    
    std::cout << "\n✓ Color Spaces demonstration complete!" << std::endl;
    std::cout << "Understanding color spaces is crucial for robust computer vision applications." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. BGR is OpenCV's default, RGB is standard display format
 * 2. HSV separates color information from intensity, better for color-based segmentation
 * 3. LAB is perceptually uniform, good for color difference calculations
 * 4. YUV separates luminance from chrominance, used in video compression
 * 5. Color masking in HSV is more robust than in BGR/RGB
 * 6. Different color spaces emphasize different aspects of color information
 * 7. White balance correction is essential for color constancy
 * 8. Choice of color space depends on the specific application requirements
 */

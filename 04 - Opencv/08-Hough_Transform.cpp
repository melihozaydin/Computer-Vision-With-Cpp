/**
 * 08-Hough_Transform.cpp
 * 
 * Line and circle detection using Hough transform.
 * 
 * Concepts covered:
 * - Hough line detection
 * - Hough circle detection
 * - Probabilistic Hough transform
 * - Parameter tuning
 * - Post-processing and filtering
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat createLineTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC1);
    
    // Draw various lines and shapes
    cv::line(image, cv::Point(50, 50), cv::Point(550, 50), cv::Scalar(255), 3);      // Horizontal
    cv::line(image, cv::Point(100, 100), cv::Point(100, 350), cv::Scalar(255), 3);   // Vertical
    cv::line(image, cv::Point(200, 100), cv::Point(500, 300), cv::Scalar(255), 3);   // Diagonal
    cv::line(image, cv::Point(150, 350), cv::Point(450, 150), cv::Scalar(255), 3);   // Diagonal
    
    // Add some rectangles (which have lines)
    cv::rectangle(image, cv::Point(300, 200), cv::Point(400, 300), cv::Scalar(255), 2);
    
    // Add noise
    cv::Mat noise(image.size(), CV_8UC1);
    cv::randu(noise, 0, 30);
    cv::add(image, noise, image);
    
    return image;
}

cv::Mat createCircleTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC1);
    
    // Draw various circles
    cv::circle(image, cv::Point(150, 150), 60, cv::Scalar(255), 3);
    cv::circle(image, cv::Point(350, 150), 40, cv::Scalar(255), 2);
    cv::circle(image, cv::Point(450, 250), 80, cv::Scalar(255), 4);
    cv::circle(image, cv::Point(200, 300), 50, cv::Scalar(255), 3);
    
    // Add some ellipses (partial circles)
    cv::ellipse(image, cv::Point(500, 100), cv::Size(30, 50), 0, 0, 180, cv::Scalar(255), 2);
    
    // Add noise
    cv::Mat noise(image.size(), CV_8UC1);
    cv::randu(noise, 0, 20);
    cv::add(image, noise, image);
    
    return image;
}

void demonstrateHoughLines(const cv::Mat& src) {
    std::cout << "\n=== Hough Line Detection ===" << std::endl;
    
    // Apply edge detection first
    cv::Mat edges;
    cv::Canny(src, edges, 50, 150, 3);
    
    // Standard Hough Transform
    std::vector<cv::Vec2f> lines_standard;
    cv::HoughLines(edges, lines_standard, 1, CV_PI/180, 100);
    
    // Probabilistic Hough Transform
    std::vector<cv::Vec4i> lines_prob;
    cv::HoughLinesP(edges, lines_prob, 1, CV_PI/180, 50, 30, 10);
    
    // Draw standard Hough lines
    cv::Mat result_standard;
    cv::cvtColor(src, result_standard, cv::COLOR_GRAY2BGR);
    
    for (size_t i = 0; i < lines_standard.size() && i < 20; i++) {
        float rho = lines_standard[i][0];
        float theta = lines_standard[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        cv::line(result_standard, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    }
    
    // Draw probabilistic Hough lines
    cv::Mat result_prob;
    cv::cvtColor(src, result_prob, cv::COLOR_GRAY2BGR);
    
    for (size_t i = 0; i < lines_prob.size(); i++) {
        cv::Vec4i l = lines_prob[i];
        cv::line(result_prob, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 2);
    }
    
    // Different thresholds
    std::vector<cv::Vec4i> lines_low, lines_high;
    cv::HoughLinesP(edges, lines_low, 1, CV_PI/180, 30, 20, 5);   // Lower threshold
    cv::HoughLinesP(edges, lines_high, 1, CV_PI/180, 80, 50, 20); // Higher threshold
    
    cv::Mat result_low, result_high;
    cv::cvtColor(src, result_low, cv::COLOR_GRAY2BGR);
    cv::cvtColor(src, result_high, cv::COLOR_GRAY2BGR);
    
    for (size_t i = 0; i < lines_low.size(); i++) {
        cv::Vec4i l = lines_low[i];
        cv::line(result_low, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 2);
    }
    
    for (size_t i = 0; i < lines_high.size(); i++) {
        cv::Vec4i l = lines_high[i];
        cv::line(result_high, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 255), 2);
    }
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Edges", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Standard Hough", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Probabilistic Hough", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Edges", edges);
    cv::imshow("Standard Hough", result_standard);
    cv::imshow("Probabilistic Hough", result_prob);
    
    std::cout << "Standard Hough found " << lines_standard.size() << " lines" << std::endl;
    std::cout << "Probabilistic Hough found " << lines_prob.size() << " line segments" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Show threshold comparison
    cv::namedWindow("Low Threshold", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("High Threshold", cv::WINDOW_AUTOSIZE);
    cv::imshow("Low Threshold", result_low);
    cv::imshow("High Threshold", result_high);
    
    std::cout << "Low threshold found " << lines_low.size() << " lines" << std::endl;
    std::cout << "High threshold found " << lines_high.size() << " lines" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateHoughCircles(const cv::Mat& src) {
    std::cout << "\n=== Hough Circle Detection ===" << std::endl;
    
    // Apply some blur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(9, 9), 2, 2);
    
    // Detect circles with different parameters
    std::vector<cv::Vec3f> circles_default, circles_sensitive, circles_strict;
    
    // Default parameters
    cv::HoughCircles(blurred, circles_default, cv::HOUGH_GRADIENT, 1, 50, 100, 30, 10, 100);
    
    // More sensitive (lower thresholds)
    cv::HoughCircles(blurred, circles_sensitive, cv::HOUGH_GRADIENT, 1, 30, 80, 20, 10, 100);
    
    // More strict (higher thresholds)
    cv::HoughCircles(blurred, circles_strict, cv::HOUGH_GRADIENT, 1, 80, 120, 50, 20, 80);
    
    // Draw circles
    cv::Mat result_default, result_sensitive, result_strict;
    cv::cvtColor(src, result_default, cv::COLOR_GRAY2BGR);
    cv::cvtColor(src, result_sensitive, cv::COLOR_GRAY2BGR);
    cv::cvtColor(src, result_strict, cv::COLOR_GRAY2BGR);
    
    // Draw default circles
    for (size_t i = 0; i < circles_default.size(); i++) {
        cv::Vec3i c = circles_default[i];
        cv::Point center = cv::Point(c[0], c[1]);
        int radius = c[2];
        cv::circle(result_default, center, 3, cv::Scalar(0, 255, 0), -1);  // Center
        cv::circle(result_default, center, radius, cv::Scalar(0, 0, 255), 3); // Circle
    }
    
    // Draw sensitive circles
    for (size_t i = 0; i < circles_sensitive.size(); i++) {
        cv::Vec3i c = circles_sensitive[i];
        cv::Point center = cv::Point(c[0], c[1]);
        int radius = c[2];
        cv::circle(result_sensitive, center, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(result_sensitive, center, radius, cv::Scalar(255, 0, 0), 3);
    }
    
    // Draw strict circles
    for (size_t i = 0; i < circles_strict.size(); i++) {
        cv::Vec3i c = circles_strict[i];
        cv::Point center = cv::Point(c[0], c[1]);
        int radius = c[2];
        cv::circle(result_strict, center, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(result_strict, center, radius, cv::Scalar(0, 255, 255), 3);
    }
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Blurred", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Default Parameters", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Sensitive Detection", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Strict Detection", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Blurred", blurred);
    cv::imshow("Default Parameters", result_default);
    cv::imshow("Sensitive Detection", result_sensitive);
    cv::imshow("Strict Detection", result_strict);
    
    std::cout << "Default parameters found " << circles_default.size() << " circles" << std::endl;
    std::cout << "Sensitive parameters found " << circles_sensitive.size() << " circles" << std::endl;
    std::cout << "Strict parameters found " << circles_strict.size() << " circles" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateParameterTuning() {
    std::cout << "\n=== Parameter Tuning Demo ===" << std::endl;
    
    // Create a test image with known lines and circles
    cv::Mat test_image = cv::Mat::zeros(300, 400, CV_8UC1);
    
    // Add perfect lines
    cv::line(test_image, cv::Point(50, 50), cv::Point(350, 50), cv::Scalar(255), 3);
    cv::line(test_image, cv::Point(200, 100), cv::Point(200, 250), cv::Scalar(255), 3);
    
    // Add perfect circles
    cv::circle(test_image, cv::Point(150, 150), 40, cv::Scalar(255), 3);
    cv::circle(test_image, cv::Point(300, 200), 30, cv::Scalar(255), 3);
    
    // Test different accumulator thresholds for lines
    cv::Mat edges;
    cv::Canny(test_image, edges, 50, 150);
    
    std::vector<int> thresholds = {20, 50, 80, 120};
    
    std::cout << "Line detection with different thresholds:" << std::endl;
    for (int thresh : thresholds) {
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI/180, thresh, 30, 10);
        std::cout << "Threshold " << thresh << ": " << lines.size() << " lines found" << std::endl;
    }
    
    // Test different parameters for circles
    cv::Mat blurred;
    cv::GaussianBlur(test_image, blurred, cv::Size(5, 5), 1);
    
    std::vector<int> circle_thresholds = {10, 20, 30, 50};
    
    std::cout << "\nCircle detection with different thresholds:" << std::endl;
    for (int thresh : circle_thresholds) {
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1, 50, 100, thresh, 10, 60);
        std::cout << "Threshold " << thresh << ": " << circles.size() << " circles found" << std::endl;
    }
    
    cv::namedWindow("Test Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Edges for Lines", cv::WINDOW_AUTOSIZE);
    cv::imshow("Test Image", test_image);
    cv::imshow("Edges for Lines", edges);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstratePostProcessing() {
    std::cout << "\n=== Post-processing and Filtering ===" << std::endl;
    
    cv::Mat image = createLineTestImage();
    cv::Mat edges;
    cv::Canny(image, edges, 50, 150);
    
    // Detect lines with lower threshold (more detections)
    std::vector<cv::Vec4i> raw_lines;
    cv::HoughLinesP(edges, raw_lines, 1, CV_PI/180, 30, 20, 5);
    
    // Filter lines by length
    std::vector<cv::Vec4i> long_lines;
    for (const auto& line : raw_lines) {
        double length = sqrt(pow(line[2] - line[0], 2) + pow(line[3] - line[1], 2));
        if (length > 50) {  // Keep only lines longer than 50 pixels
            long_lines.push_back(line);
        }
    }
    
    // Filter lines by angle (keep only horizontal and vertical)
    std::vector<cv::Vec4i> filtered_lines;
    for (const auto& line : long_lines) {
        double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI;
        angle = abs(angle);
        if (angle < 10 || angle > 80) {  // Horizontal or vertical lines
            filtered_lines.push_back(line);
        }
    }
    
    // Merge nearby parallel lines
    std::vector<cv::Vec4i> merged_lines;
    std::vector<bool> used(filtered_lines.size(), false);
    
    for (size_t i = 0; i < filtered_lines.size(); i++) {
        if (used[i]) continue;
        
        cv::Vec4i base_line = filtered_lines[i];
        std::vector<cv::Vec4i> similar_lines;
        similar_lines.push_back(base_line);
        used[i] = true;
        
        for (size_t j = i + 1; j < filtered_lines.size(); j++) {
            if (used[j]) continue;
            
            cv::Vec4i test_line = filtered_lines[j];
            
            // Check if lines are parallel and close
            double angle1 = atan2(base_line[3] - base_line[1], base_line[2] - base_line[0]);
            double angle2 = atan2(test_line[3] - test_line[1], test_line[2] - test_line[0]);
            double angle_diff = abs(angle1 - angle2) * 180.0 / CV_PI;
            
            if (angle_diff < 10) {  // Lines are parallel
                similar_lines.push_back(test_line);
                used[j] = true;
            }
        }
        
        // Average the similar lines to create one merged line
        if (similar_lines.size() > 1) {
            int avg_x1 = 0, avg_y1 = 0, avg_x2 = 0, avg_y2 = 0;
            for (const auto& line : similar_lines) {
                avg_x1 += line[0]; avg_y1 += line[1];
                avg_x2 += line[2]; avg_y2 += line[3];
            }
            avg_x1 /= similar_lines.size(); avg_y1 /= similar_lines.size();
            avg_x2 /= similar_lines.size(); avg_y2 /= similar_lines.size();
            merged_lines.push_back(cv::Vec4i(avg_x1, avg_y1, avg_x2, avg_y2));
        } else {
            merged_lines.push_back(base_line);
        }
    }
    
    // Draw comparison
    cv::Mat result_raw, result_filtered, result_merged;
    cv::cvtColor(image, result_raw, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image, result_filtered, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image, result_merged, cv::COLOR_GRAY2BGR);
    
    // Draw raw lines
    for (const auto& line : raw_lines) {
        cv::line(result_raw, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2);
    }
    
    // Draw filtered lines
    for (const auto& line : filtered_lines) {
        cv::line(result_filtered, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2);
    }
    
    // Draw merged lines
    for (const auto& line : merged_lines) {
        cv::line(result_merged, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 0, 0), 3);
    }
    
    cv::namedWindow("Raw Detection", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Length & Angle Filtered", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Merged Similar Lines", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Raw Detection", result_raw);
    cv::imshow("Length & Angle Filtered", result_filtered);
    cv::imshow("Merged Similar Lines", result_merged);
    
    std::cout << "Raw detections: " << raw_lines.size() << " lines" << std::endl;
    std::cout << "After filtering: " << filtered_lines.size() << " lines" << std::endl;
    std::cout << "After merging: " << merged_lines.size() << " lines" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Hough Transform ===" << std::endl;
    
    // Create test images
    cv::Mat line_image = createLineTestImage();
    cv::Mat circle_image = createCircleTestImage();
    
    // Try to load real images if available
    cv::Mat real_image = cv::imread("../data/test.jpg", cv::IMREAD_GRAYSCALE);
    if (!real_image.empty()) {
        std::cout << "Using loaded image for some demonstrations." << std::endl;
    }
    
    // Demonstrate Hough line detection
    demonstrateHoughLines(line_image);
    
    // Demonstrate Hough circle detection
    demonstrateHoughCircles(circle_image);
    
    // Show parameter tuning effects
    demonstrateParameterTuning();
    
    // Demonstrate post-processing techniques
    demonstratePostProcessing();
    
    std::cout << "\n✓ Hough Transform demonstration complete!" << std::endl;
    std::cout << "Hough transforms are powerful for detecting geometric shapes in noisy images." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Standard Hough Transform: detects infinite lines, returns rho and theta
 * 2. Probabilistic Hough Transform: detects line segments, more efficient
 * 3. Hough Circle Detection: uses gradient information, sensitive to parameters
 * 4. Parameter tuning is crucial for good results
 * 5. Pre-processing (edge detection, blur) significantly affects results
 * 6. Post-processing can filter and merge similar detections
 * 7. Different accumulator thresholds control sensitivity vs false positives
 * 8. Always validate results with ground truth when possible
 */

/**
 * 12-Corner_Detection.cpp
 * 
 * Detecting corners and keypoints in images.
 * 
 * Concepts covered:
 * - Harris corner detector
 * - Shi-Tomasi corners
 * - goodFeaturesToTrack
 * - FAST corner detector
 * - Corner refinement
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

cv::Mat createCornerTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // Create various shapes and patterns that will produce corners
    
    // Rectangle with clear corners
    cv::rectangle(image, cv::Point(50, 50), cv::Point(150, 120), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(image, cv::Point(70, 70), cv::Point(130, 100), cv::Scalar(0, 0, 0), -1);
    
    // Triangle
    std::vector<cv::Point> triangle = {cv::Point(200, 50), cv::Point(150, 120), cv::Point(250, 120)};
    cv::fillPoly(image, triangle, cv::Scalar(255, 255, 255));
    
    // Cross pattern
    cv::rectangle(image, cv::Point(320, 80), cv::Point(380, 100), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(image, cv::Point(340, 60), cv::Point(360, 120), cv::Scalar(255, 255, 255), -1);
    
    // L-shape
    cv::rectangle(image, cv::Point(450, 50), cv::Point(470, 120), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(image, cv::Point(450, 100), cv::Point(520, 120), cv::Scalar(255, 255, 255), -1);
    
    // Checkerboard pattern
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 4; j++) {
            if ((i + j) % 2 == 0) {
                cv::rectangle(image, cv::Point(50 + i * 20, 200 + j * 20), 
                             cv::Point(70 + i * 20, 220 + j * 20), cv::Scalar(255, 255, 255), -1);
            }
        }
    }
    
    // Circular patterns with corners
    cv::circle(image, cv::Point(300, 250), 30, cv::Scalar(255, 255, 255), -1);
    cv::circle(image, cv::Point(300, 250), 15, cv::Scalar(0, 0, 0), -1);
    
    // Text (creates many corners)
    cv::putText(image, "CORNERS", cv::Point(400, 280), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // Add some noise and lines
    cv::line(image, cv::Point(100, 350), cv::Point(500, 320), cv::Scalar(255, 255, 255), 2);
    cv::line(image, cv::Point(150, 300), cv::Point(200, 380), cv::Scalar(255, 255, 255), 2);
    
    return image;
}

void demonstrateHarrisCorners(const cv::Mat& src) {
    std::cout << "\n=== Harris Corner Detection ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Harris corner detection
    cv::Mat harris_response;
    cv::cornerHarris(gray, harris_response, 2, 3, 0.04);
    
    // Normalize the response
    cv::Mat harris_norm;
    cv::normalize(harris_response, harris_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    // Threshold to find corners
    cv::Mat corners_binary;
    cv::threshold(harris_norm, corners_binary, 50, 255, cv::THRESH_BINARY);
    
    // Find corner locations
    std::vector<cv::Point2f> corners;
    for (int y = 0; y < harris_norm.rows; y++) {
        for (int x = 0; x < harris_norm.cols; x++) {
            if (harris_norm.at<uchar>(y, x) > 50) {
                corners.push_back(cv::Point2f(x, y));
            }
        }
    }
    
    // Non-maximum suppression
    std::vector<cv::Point2f> filtered_corners;
    for (const auto& corner : corners) {
        bool is_local_max = true;
        for (const auto& other : corners) {
            if (corner != other && cv::norm(corner - other) < 10) {
                if (harris_response.at<float>(corner.y, corner.x) < 
                    harris_response.at<float>(other.y, other.x)) {
                    is_local_max = false;
                    break;
                }
            }
        }
        if (is_local_max) {
            filtered_corners.push_back(corner);
        }
    }
    
    // Visualize results
    cv::Mat result = src.clone();
    cv::Mat harris_colored;
    cv::applyColorMap(harris_norm, harris_colored, cv::COLORMAP_JET);
    
    // Draw corners
    for (const auto& corner : filtered_corners) {
        cv::circle(result, corner, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(result, corner, 6, cv::Scalar(0, 255, 0), 1);
    }
    
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Harris Response", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Harris Corners", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Harris Response", harris_colored);
    cv::imshow("Harris Corners", result);
    
    std::cout << "Harris corner detector found " << filtered_corners.size() << " corners" << std::endl;
    std::cout << "Harris response shows corner strength at each pixel" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateShiTomasiCorners(const cv::Mat& src) {
    std::cout << "\n=== Shi-Tomasi Corner Detection ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Shi-Tomasi corner detection using goodFeaturesToTrack
    std::vector<cv::Point2f> corners;
    double quality_level = 0.01;
    double min_distance = 10;
    int max_corners = 100;
    
    cv::goodFeaturesToTrack(gray, corners, max_corners, quality_level, min_distance);
    
    // Test different parameters
    std::vector<cv::Point2f> corners_high_quality, corners_low_distance;
    cv::goodFeaturesToTrack(gray, corners_high_quality, max_corners, 0.05, min_distance);  // Higher quality
    cv::goodFeaturesToTrack(gray, corners_low_distance, max_corners, quality_level, 5);    // Lower distance
    
    // Visualize results
    cv::Mat result1 = src.clone();
    cv::Mat result2 = src.clone();
    cv::Mat result3 = src.clone();
    
    // Draw corners with different colors for different parameter sets
    for (const auto& corner : corners) {
        cv::circle(result1, corner, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(result1, corner, 6, cv::Scalar(0, 0, 255), 1);
    }
    
    for (const auto& corner : corners_high_quality) {
        cv::circle(result2, corner, 3, cv::Scalar(255, 0, 0), -1);
        cv::circle(result2, corner, 6, cv::Scalar(0, 255, 255), 1);
    }
    
    for (const auto& corner : corners_low_distance) {
        cv::circle(result3, corner, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(result3, corner, 6, cv::Scalar(255, 255, 0), 1);
    }
    
    cv::namedWindow("Shi-Tomasi (quality=0.01, dist=10)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Shi-Tomasi (quality=0.05, dist=10)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Shi-Tomasi (quality=0.01, dist=5)", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Shi-Tomasi (quality=0.01, dist=10)", result1);
    cv::imshow("Shi-Tomasi (quality=0.05, dist=10)", result2);
    cv::imshow("Shi-Tomasi (quality=0.01, dist=5)", result3);
    
    std::cout << "Shi-Tomasi corners found:" << std::endl;
    std::cout << "  Standard params: " << corners.size() << " corners" << std::endl;
    std::cout << "  High quality: " << corners_high_quality.size() << " corners" << std::endl;
    std::cout << "  Low min distance: " << corners_low_distance.size() << " corners" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateFASTCorners(const cv::Mat& src) {
    std::cout << "\n=== FAST Corner Detection ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // FAST corner detection
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(gray, keypoints, 10, true);  // threshold=10, non-maximum suppression=true
    
    // Test different thresholds
    std::vector<cv::KeyPoint> keypoints_low, keypoints_high;
    cv::FAST(gray, keypoints_low, 5, true);   // Lower threshold (more corners)
    cv::FAST(gray, keypoints_high, 20, true); // Higher threshold (fewer corners)
    
    // Test without non-maximum suppression
    std::vector<cv::KeyPoint> keypoints_no_nms;
    cv::FAST(gray, keypoints_no_nms, 10, false);
    
    // Visualize results
    cv::Mat result1, result2, result3, result4;
    cv::drawKeypoints(src, keypoints, result1, cv::Scalar(0, 255, 0));
    cv::drawKeypoints(src, keypoints_low, result2, cv::Scalar(255, 0, 0));
    cv::drawKeypoints(src, keypoints_high, result3, cv::Scalar(0, 0, 255));
    cv::drawKeypoints(src, keypoints_no_nms, result4, cv::Scalar(255, 255, 0));
    
    cv::namedWindow("FAST (threshold=10)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("FAST (threshold=5)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("FAST (threshold=20)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("FAST (no NMS)", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("FAST (threshold=10)", result1);
    cv::imshow("FAST (threshold=5)", result2);
    cv::imshow("FAST (threshold=20)", result3);
    cv::imshow("FAST (no NMS)", result4);
    
    std::cout << "FAST corner detection results:" << std::endl;
    std::cout << "  Threshold 10: " << keypoints.size() << " corners" << std::endl;
    std::cout << "  Threshold 5: " << keypoints_low.size() << " corners" << std::endl;
    std::cout << "  Threshold 20: " << keypoints_high.size() << " corners" << std::endl;
    std::cout << "  No NMS: " << keypoints_no_nms.size() << " corners" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateCornerRefinement(const cv::Mat& src) {
    std::cout << "\n=== Corner Refinement ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Get initial corners
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, 50, 0.01, 10);
    
    // Refine corners using cornerSubPix
    std::vector<cv::Point2f> refined_corners = corners;
    cv::Size win_size(5, 5);
    cv::Size zero_zone(-1, -1);
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
    
    cv::cornerSubPix(gray, refined_corners, win_size, zero_zone, criteria);
    
    // Calculate refinement displacement
    std::vector<double> displacements;
    for (size_t i = 0; i < corners.size(); i++) {
        double dist = cv::norm(corners[i] - refined_corners[i]);
        displacements.push_back(dist);
    }
    
    // Visualize results
    cv::Mat result = src.clone();
    
    // Draw original corners in red
    for (const auto& corner : corners) {
        cv::circle(result, corner, 4, cv::Scalar(0, 0, 255), -1);
    }
    
    // Draw refined corners in green
    for (const auto& corner : refined_corners) {
        cv::circle(result, corner, 2, cv::Scalar(0, 255, 0), -1);
    }
    
    // Draw displacement vectors
    for (size_t i = 0; i < corners.size(); i++) {
        if (displacements[i] > 0.5) {  // Only show significant displacements
            cv::line(result, corners[i], refined_corners[i], cv::Scalar(255, 255, 0), 1);
            cv::circle(result, refined_corners[i], 6, cv::Scalar(255, 255, 0), 1);
        }
    }
    
    cv::namedWindow("Corner Refinement", cv::WINDOW_AUTOSIZE);
    cv::imshow("Corner Refinement", result);
    
    // Calculate statistics
    double avg_displacement = 0;
    double max_displacement = 0;
    for (double disp : displacements) {
        avg_displacement += disp;
        max_displacement = std::max(max_displacement, disp);
    }
    avg_displacement /= displacements.size();
    
    std::cout << "Corner refinement statistics:" << std::endl;
    std::cout << "  Number of corners: " << corners.size() << std::endl;
    std::cout << "  Average displacement: " << avg_displacement << " pixels" << std::endl;
    std::cout << "  Maximum displacement: " << max_displacement << " pixels" << std::endl;
    std::cout << "Red circles: original corners" << std::endl;
    std::cout << "Green circles: refined corners" << std::endl;
    std::cout << "Yellow lines: displacement vectors" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateCornerComparison(const cv::Mat& src) {
    std::cout << "\n=== Corner Detection Comparison ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Harris corners
    cv::Mat harris_response;
    cv::cornerHarris(gray, harris_response, 2, 3, 0.04);
    cv::Mat harris_norm;
    cv::normalize(harris_response, harris_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    std::vector<cv::Point2f> harris_corners;
    for (int y = 0; y < harris_norm.rows; y++) {
        for (int x = 0; x < harris_norm.cols; x++) {
            if (harris_norm.at<uchar>(y, x) > 50) {
                harris_corners.push_back(cv::Point2f(x, y));
            }
        }
    }
    
    // Shi-Tomasi corners
    std::vector<cv::Point2f> shitomasi_corners;
    cv::goodFeaturesToTrack(gray, shitomasi_corners, 100, 0.01, 10);
    
    // FAST corners
    std::vector<cv::KeyPoint> fast_keypoints;
    cv::FAST(gray, fast_keypoints, 10, true);
    
    // Convert FAST keypoints to points
    std::vector<cv::Point2f> fast_corners;
    for (const auto& kp : fast_keypoints) {
        fast_corners.push_back(kp.pt);
    }
    
    // Create comparison visualization
    cv::Mat comparison = cv::Mat::zeros(src.rows, src.cols * 3, CV_8UC3);
    
    // Harris
    cv::Mat harris_result = src.clone();
    for (const auto& corner : harris_corners) {
        cv::circle(harris_result, corner, 3, cv::Scalar(0, 0, 255), -1);
    }
    harris_result.copyTo(comparison(cv::Rect(0, 0, src.cols, src.rows)));
    cv::putText(comparison, "Harris (" + std::to_string(harris_corners.size()) + ")",
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // Shi-Tomasi
    cv::Mat shitomasi_result = src.clone();
    for (const auto& corner : shitomasi_corners) {
        cv::circle(shitomasi_result, corner, 3, cv::Scalar(0, 255, 0), -1);
    }
    shitomasi_result.copyTo(comparison(cv::Rect(src.cols, 0, src.cols, src.rows)));
    cv::putText(comparison, "Shi-Tomasi (" + std::to_string(shitomasi_corners.size()) + ")",
               cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // FAST
    cv::Mat fast_result = src.clone();
    for (const auto& corner : fast_corners) {
        cv::circle(fast_result, corner, 3, cv::Scalar(255, 0, 0), -1);
    }
    fast_result.copyTo(comparison(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    cv::putText(comparison, "FAST (" + std::to_string(fast_corners.size()) + ")",
               cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Corner Detection Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("Corner Detection Comparison", comparison);
    
    std::cout << "Corner detection comparison:" << std::endl;
    std::cout << "  Harris: " << harris_corners.size() << " corners (red)" << std::endl;
    std::cout << "  Shi-Tomasi: " << shitomasi_corners.size() << " corners (green)" << std::endl;
    std::cout << "  FAST: " << fast_corners.size() << " corners (blue)" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Corner Detection ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createCornerTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg");
    if (!real_image.empty()) {
        std::cout << "Using loaded image for some demonstrations." << std::endl;
        
        // Use real image for some demonstrations
        demonstrateHarrisCorners(real_image);
        demonstrateShiTomasiCorners(real_image);
        demonstrateFASTCorners(test_image);       // Use synthetic for FAST
        demonstrateCornerRefinement(real_image);
        demonstrateCornerComparison(test_image);  // Use synthetic for comparison
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all corner detection methods
        demonstrateHarrisCorners(test_image);
        demonstrateShiTomasiCorners(test_image);
        demonstrateFASTCorners(test_image);
        demonstrateCornerRefinement(test_image);
        demonstrateCornerComparison(test_image);
    }
    
    std::cout << "\n✓ Corner Detection demonstration complete!" << std::endl;
    std::cout << "Corner detection is fundamental for feature-based computer vision." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Harris corner detector uses local gradients to find corners
 * 2. Shi-Tomasi is an improved version of Harris corner detection
 * 3. goodFeaturesToTrack implements Shi-Tomasi with quality control
 * 4. FAST is optimized for real-time corner detection
 * 5. Corner refinement improves accuracy using sub-pixel precision
 * 6. Different detectors have different strengths and characteristics
 * 7. Parameter tuning significantly affects detection results
 * 8. Corner detection is preprocessing for feature matching and tracking
 */

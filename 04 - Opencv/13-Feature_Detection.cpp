/**
 * 13-Feature_Detection.cpp
 * 
 * Advanced feature detection algorithms.
 * 
 * Concepts covered:
 * - SIFT features
 * - SURF features
 * - ORB features
 * - AKAZE features
 * - Feature comparison
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

cv::Mat createFeatureTestImage() {
    cv::Mat image = cv::Mat::zeros(500, 700, CV_8UC3);
    
    // Create various textures and patterns for feature detection
    
    // Brick pattern
    for (int y = 50; y < 150; y += 20) {
        for (int x = 50; x < 200; x += 40) {
            cv::rectangle(image, cv::Point(x, y), cv::Point(x + 35, y + 15), cv::Scalar(180, 180, 180), -1);
            cv::rectangle(image, cv::Point(x, y), cv::Point(x + 35, y + 15), cv::Scalar(120, 120, 120), 1);
        }
    }
    
    // Checkerboard pattern
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 6; j++) {
            if ((i + j) % 2 == 0) {
                cv::rectangle(image, cv::Point(250 + i * 20, 50 + j * 20), 
                             cv::Point(270 + i * 20, 70 + j * 20), cv::Scalar(255, 255, 255), -1);
            }
        }
    }
    
    // Circular patterns
    for (int i = 0; i < 5; i++) {
        cv::circle(image, cv::Point(500 + i * 30, 100), 10 + i * 2, cv::Scalar(200, 200, 200), -1);
        cv::circle(image, cv::Point(500 + i * 30, 100), 10 + i * 2, cv::Scalar(100, 100, 100), 2);
    }
    
    // Complex geometric shapes
    std::vector<cv::Point> star = {
        cv::Point(150, 250), cv::Point(160, 280), cv::Point(190, 280),
        cv::Point(170, 300), cv::Point(180, 330), cv::Point(150, 315),
        cv::Point(120, 330), cv::Point(130, 300), cv::Point(110, 280),
        cv::Point(140, 280)
    };
    cv::fillPoly(image, star, cv::Scalar(255, 255, 255));
    
    // Text with various fonts and sizes
    cv::putText(image, "FEATURES", cv::Point(300, 250), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
    cv::putText(image, "detect", cv::Point(320, 280), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(200, 200, 200), 1);
    
    // Random noise pattern
    for (int i = 0; i < 200; i++) {
        cv::Point pt(rand() % 200 + 50, rand() % 100 + 350);
        cv::circle(image, pt, rand() % 3 + 1, cv::Scalar(rand() % 255, rand() % 255, rand() % 255), -1);
    }
    
    // Grid pattern
    for (int x = 350; x < 550; x += 10) {
        cv::line(image, cv::Point(x, 350), cv::Point(x, 450), cv::Scalar(150, 150, 150), 1);
    }
    for (int y = 350; y < 450; y += 10) {
        cv::line(image, cv::Point(350, y), cv::Point(550, y), cv::Scalar(150, 150, 150), 1);
    }
    
    return image;
}

void demonstrateORBFeatures(const cv::Mat& src) {
    std::cout << "\n=== ORB Feature Detection ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Create ORB detector with different parameters
    cv::Ptr<cv::ORB> orb_default = cv::ORB::create();
    cv::Ptr<cv::ORB> orb_many = cv::ORB::create(1000);  // More features
    cv::Ptr<cv::ORB> orb_few = cv::ORB::create(100);    // Fewer features
    
    std::vector<cv::KeyPoint> keypoints_default, keypoints_many, keypoints_few;
    cv::Mat descriptors_default, descriptors_many, descriptors_few;
    
    // Detect features
    orb_default->detectAndCompute(gray, cv::noArray(), keypoints_default, descriptors_default);
    orb_many->detectAndCompute(gray, cv::noArray(), keypoints_many, descriptors_many);
    orb_few->detectAndCompute(gray, cv::noArray(), keypoints_few, descriptors_few);
    
    // Visualize results
    cv::Mat result_default, result_many, result_few;
    cv::drawKeypoints(src, keypoints_default, result_default, cv::Scalar(0, 255, 0));
    cv::drawKeypoints(src, keypoints_many, result_many, cv::Scalar(255, 0, 0));
    cv::drawKeypoints(src, keypoints_few, result_few, cv::Scalar(0, 0, 255));
    
    // Create comparison image
    cv::Mat comparison = cv::Mat::zeros(src.rows, src.cols * 3, CV_8UC3);
    result_default.copyTo(comparison(cv::Rect(0, 0, src.cols, src.rows)));
    result_many.copyTo(comparison(cv::Rect(src.cols, 0, src.cols, src.rows)));
    result_few.copyTo(comparison(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(comparison, "Default (" + std::to_string(keypoints_default.size()) + ")",
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Many (" + std::to_string(keypoints_many.size()) + ")",
               cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Few (" + std::to_string(keypoints_few.size()) + ")",
               cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("ORB Features Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("ORB Features Comparison", comparison);
    
    std::cout << "ORB feature detection results:" << std::endl;
    std::cout << "  Default: " << keypoints_default.size() << " features" << std::endl;
    std::cout << "  Many features: " << keypoints_many.size() << " features" << std::endl;
    std::cout << "  Few features: " << keypoints_few.size() << " features" << std::endl;
    std::cout << "  Descriptor size: " << descriptors_default.cols << " bytes per feature" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateAKAZEFeatures(const cv::Mat& src) {
    std::cout << "\n=== AKAZE Feature Detection ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Create AKAZE detector
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    // Detect and compute features
    akaze->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    
    // Draw features with orientation and scale
    cv::Mat result;
    cv::drawKeypoints(src, keypoints, result, cv::Scalar(0, 255, 255), 
                     cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    // Analyze keypoint properties
    std::vector<float> scales, responses;
    for (const auto& kp : keypoints) {
        scales.push_back(kp.size);
        responses.push_back(kp.response);
    }
    
    if (!scales.empty()) {
        auto minmax_scale = std::minmax_element(scales.begin(), scales.end());
        auto minmax_response = std::minmax_element(responses.begin(), responses.end());
        
        std::cout << "AKAZE feature analysis:" << std::endl;
        std::cout << "  Total features: " << keypoints.size() << std::endl;
        std::cout << "  Scale range: " << *minmax_scale.first << " - " << *minmax_scale.second << std::endl;
        std::cout << "  Response range: " << *minmax_response.first << " - " << *minmax_response.second << std::endl;
        std::cout << "  Descriptor size: " << descriptors.cols << " bytes per feature" << std::endl;
    }
    
    cv::namedWindow("AKAZE Features", cv::WINDOW_AUTOSIZE);
    cv::imshow("AKAZE Features", result);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateSIFTFeatures(const cv::Mat& src) {
    std::cout << "\n=== SIFT Feature Detection ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Try to create SIFT detector (may not be available in all OpenCV builds)
    cv::Ptr<cv::SIFT> sift;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    try {
        sift = cv::SIFT::create();
        sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
        
        // Draw features with rich keypoints (shows scale and orientation)
        cv::Mat result;
        cv::drawKeypoints(src, keypoints, result, cv::Scalar(255, 0, 255), 
                         cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        cv::namedWindow("SIFT Features", cv::WINDOW_AUTOSIZE);
        cv::imshow("SIFT Features", result);
        
        std::cout << "SIFT feature detection:" << std::endl;
        std::cout << "  Total features: " << keypoints.size() << std::endl;
        std::cout << "  Descriptor size: " << descriptors.cols << " floats per feature" << std::endl;
        
        cv::waitKey(0);
        cv::destroyAllWindows();
        
    } catch (const cv::Exception& e) {
        std::cout << "SIFT not available in this OpenCV build: " << e.what() << std::endl;
    }
}

void demonstrateBRISKFeatures(const cv::Mat& src) {
    std::cout << "\n=== BRISK Feature Detection ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Create BRISK detector
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    // Detect and compute features
    brisk->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    
    // Draw features
    cv::Mat result;
    cv::drawKeypoints(src, keypoints, result, cv::Scalar(0, 255, 0));
    
    cv::namedWindow("BRISK Features", cv::WINDOW_AUTOSIZE);
    cv::imshow("BRISK Features", result);
    
    std::cout << "BRISK feature detection:" << std::endl;
    std::cout << "  Total features: " << keypoints.size() << std::endl;
    std::cout << "  Descriptor size: " << descriptors.cols << " bytes per feature" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateFeatureComparison(const cv::Mat& src) {
    std::cout << "\n=== Feature Detection Comparison ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Create different detectors
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    
    // Detect features with each detector
    std::vector<cv::KeyPoint> orb_kpts, akaze_kpts, brisk_kpts;
    cv::Mat orb_desc, akaze_desc, brisk_desc;
    
    auto start = cv::getTickCount();
    orb->detectAndCompute(gray, cv::noArray(), orb_kpts, orb_desc);
    auto orb_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
    
    start = cv::getTickCount();
    akaze->detectAndCompute(gray, cv::noArray(), akaze_kpts, akaze_desc);
    auto akaze_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
    
    start = cv::getTickCount();
    brisk->detectAndCompute(gray, cv::noArray(), brisk_kpts, brisk_desc);
    auto brisk_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
    
    // Create comparison visualization
    cv::Mat comparison = cv::Mat::zeros(src.rows, src.cols * 3, CV_8UC3);
    
    cv::Mat orb_result, akaze_result, brisk_result;
    cv::drawKeypoints(src, orb_kpts, orb_result, cv::Scalar(0, 255, 0));
    cv::drawKeypoints(src, akaze_kpts, akaze_result, cv::Scalar(255, 0, 0));
    cv::drawKeypoints(src, brisk_kpts, brisk_result, cv::Scalar(0, 0, 255));
    
    orb_result.copyTo(comparison(cv::Rect(0, 0, src.cols, src.rows)));
    akaze_result.copyTo(comparison(cv::Rect(src.cols, 0, src.cols, src.rows)));
    brisk_result.copyTo(comparison(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Add labels with statistics
    cv::putText(comparison, "ORB: " + std::to_string(orb_kpts.size()) + " (" + 
               std::to_string(static_cast<int>(orb_time)) + "ms)",
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "AKAZE: " + std::to_string(akaze_kpts.size()) + " (" + 
               std::to_string(static_cast<int>(akaze_time)) + "ms)",
               cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "BRISK: " + std::to_string(brisk_kpts.size()) + " (" + 
               std::to_string(static_cast<int>(brisk_time)) + "ms)",
               cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Feature Detection Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("Feature Detection Comparison", comparison);
    
    // Print detailed comparison
    std::cout << "Feature detection comparison:" << std::endl;
    std::cout << "┌──────────┬──────────┬──────────┬──────────┬──────────┐" << std::endl;
    std::cout << "│ Method   │ Features │ Desc Size│ Time (ms)│ Type     │" << std::endl;
    std::cout << "├──────────┼──────────┼──────────┼──────────┼──────────┤" << std::endl;
    std::cout << "│ ORB      │ " << std::setw(8) << orb_kpts.size() 
              << " │ " << std::setw(8) << orb_desc.cols 
              << " │ " << std::setw(8) << static_cast<int>(orb_time)
              << " │ Binary   │" << std::endl;
    std::cout << "│ AKAZE    │ " << std::setw(8) << akaze_kpts.size() 
              << " │ " << std::setw(8) << akaze_desc.cols 
              << " │ " << std::setw(8) << static_cast<int>(akaze_time)
              << " │ Binary   │" << std::endl;
    std::cout << "│ BRISK    │ " << std::setw(8) << brisk_kpts.size() 
              << " │ " << std::setw(8) << brisk_desc.cols 
              << " │ " << std::setw(8) << static_cast<int>(brisk_time)
              << " │ Binary   │" << std::endl;
    std::cout << "└──────────┴──────────┴──────────┴──────────┴──────────┘" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateKeypointFiltering(const cv::Mat& src) {
    std::cout << "\n=== Keypoint Filtering and Analysis ===" << std::endl;
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Detect features with ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
    std::vector<cv::KeyPoint> keypoints;
    orb->detect(gray, keypoints);
    
    // Filter keypoints by response (strength)
    std::vector<cv::KeyPoint> strong_kpts, weak_kpts;
    float response_threshold = 20.0f;
    
    for (const auto& kp : keypoints) {
        if (kp.response > response_threshold) {
            strong_kpts.push_back(kp);
        } else {
            weak_kpts.push_back(kp);
        }
    }
    
    // Filter keypoints by size (scale)
    std::vector<cv::KeyPoint> large_kpts, small_kpts;
    float size_threshold = 10.0f;
    
    for (const auto& kp : keypoints) {
        if (kp.size > size_threshold) {
            large_kpts.push_back(kp);
        } else {
            small_kpts.push_back(kp);
        }
    }
    
    // Create visualization
    cv::Mat result = cv::Mat::zeros(src.rows, src.cols * 2, CV_8UC3);
    
    // Left: Response-based filtering
    cv::Mat response_vis = src.clone();
    for (const auto& kp : strong_kpts) {
        cv::circle(response_vis, kp.pt, 3, cv::Scalar(0, 255, 0), -1);  // Green for strong
    }
    for (const auto& kp : weak_kpts) {
        cv::circle(response_vis, kp.pt, 2, cv::Scalar(0, 0, 255), -1);  // Red for weak
    }
    response_vis.copyTo(result(cv::Rect(0, 0, src.cols, src.rows)));
    
    // Right: Size-based filtering
    cv::Mat size_vis = src.clone();
    for (const auto& kp : large_kpts) {
        cv::circle(size_vis, kp.pt, static_cast<int>(kp.size/2), cv::Scalar(255, 0, 0), 1);  // Blue circle for large
        cv::circle(size_vis, kp.pt, 2, cv::Scalar(255, 0, 0), -1);
    }
    for (const auto& kp : small_kpts) {
        cv::circle(size_vis, kp.pt, 1, cv::Scalar(0, 255, 255), -1);  // Yellow for small
    }
    size_vis.copyTo(result(cv::Rect(src.cols, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(result, "Response Filter (>20)", cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(result, "Size Filter (>10)", cv::Point(src.cols + 10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Keypoint Filtering", cv::WINDOW_AUTOSIZE);
    cv::imshow("Keypoint Filtering", result);
    
    std::cout << "Keypoint filtering results:" << std::endl;
    std::cout << "  Total keypoints: " << keypoints.size() << std::endl;
    std::cout << "  Strong response (>" << response_threshold << "): " << strong_kpts.size() << std::endl;
    std::cout << "  Weak response: " << weak_kpts.size() << std::endl;
    std::cout << "  Large size (>" << size_threshold << "): " << large_kpts.size() << std::endl;
    std::cout << "  Small size: " << small_kpts.size() << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Feature Detection ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createFeatureTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg");
    if (!real_image.empty()) {
        std::cout << "Using loaded image for some demonstrations." << std::endl;
        
        // Use real image for most demonstrations
        demonstrateORBFeatures(real_image);
        demonstrateAKAZEFeatures(real_image);
        demonstrateSIFTFeatures(real_image);
        demonstrateBRISKFeatures(test_image);        // Use synthetic for BRISK
        demonstrateFeatureComparison(real_image);
        demonstrateKeypointFiltering(real_image);
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all feature detection methods
        demonstrateORBFeatures(test_image);
        demonstrateAKAZEFeatures(test_image);
        demonstrateSIFTFeatures(test_image);
        demonstrateBRISKFeatures(test_image);
        demonstrateFeatureComparison(test_image);
        demonstrateKeypointFiltering(test_image);
    }
    
    std::cout << "\n✓ Feature Detection demonstration complete!" << std::endl;
    std::cout << "Feature detection is essential for image matching, recognition, and tracking." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. ORB is fast and efficient for real-time applications
 * 2. AKAZE provides good balance between speed and accuracy
 * 3. SIFT is scale and rotation invariant but computationally expensive
 * 4. BRISK is binary descriptor, fast for matching
 * 5. Different detectors excel in different scenarios
 * 6. Keypoint filtering improves quality and reduces computational load
 * 7. Feature response indicates strength/distinctiveness
 * 8. Feature scale provides information about local structure size
 */

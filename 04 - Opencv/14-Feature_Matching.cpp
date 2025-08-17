/**
 * 14-Feature_Matching.cpp
 * 
 * Matching features between images.
 * 
 * Concepts covered:
 * - Brute-force matcher
 * - FLANN matcher
 * - Cross-check filtering
 * - Ratio test
 * - Geometric verification
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

std::pair<cv::Mat, cv::Mat> createMatchingTestImages() {
    // Create two related images for feature matching
    cv::Mat img1 = cv::Mat::zeros(400, 600, CV_8UC3);
    cv::Mat img2 = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // Base pattern in both images
    cv::rectangle(img1, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(255, 255, 255), -1);
    cv::circle(img1, cv::Point(350, 150), 50, cv::Scalar(200, 200, 200), -1);
    cv::putText(img1, "MATCH", cv::Point(250, 300), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // Add checkerboard pattern
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 6; j++) {
            if ((i + j) % 2 == 0) {
                cv::rectangle(img1, cv::Point(450 + i * 15, 50 + j * 15), 
                             cv::Point(465 + i * 15, 65 + j * 15), cv::Scalar(255, 255, 255), -1);
            }
        }
    }
    
    // Second image: transformed version of first
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(cv::Point2f(300, 200), 15, 0.9);
    cv::warpAffine(img1, img2, rotation_matrix, img1.size());
    
    // Add some different elements to img2
    cv::rectangle(img2, cv::Point(50, 50), cv::Point(80, 80), cv::Scalar(255, 255, 255), -1);
    cv::circle(img2, cv::Point(500, 350), 30, cv::Scalar(180, 180, 180), -1);
    
    // Add noise to both images
    cv::Mat noise1(img1.size(), CV_8UC3);
    cv::Mat noise2(img2.size(), CV_8UC3);
    cv::randn(noise1, 0, 10);
    cv::randn(noise2, 0, 10);
    img1 += noise1;
    img2 += noise2;
    
    return std::make_pair(img1, img2);
}

void demonstrateBruteForceMatching(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n=== Brute Force Matching ===" << std::endl;
    
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    
    // Detect features using ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    
    orb->detectAndCompute(gray1, cv::noArray(), kpts1, desc1);
    orb->detectAndCompute(gray2, cv::noArray(), kpts2, desc2);
    
    // Brute force matcher
    cv::BFMatcher bf_matcher(cv::NORM_HAMMING, false);  // For ORB descriptors
    
    std::vector<cv::DMatch> matches;
    bf_matcher.match(desc1, desc2, matches);
    
    // Sort matches by distance (quality)
    std::sort(matches.begin(), matches.end(), 
              [](const cv::DMatch& a, const cv::DMatch& b) {
                  return a.distance < b.distance;
              });
    
    // Keep only good matches (top 50%)
    std::vector<cv::DMatch> good_matches;
    int num_good = static_cast<int>(matches.size() * 0.5);
    good_matches.assign(matches.begin(), matches.begin() + num_good);
    
    // Draw matches
    cv::Mat match_img;
    cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, match_img);
    
    cv::namedWindow("Brute Force Matches", cv::WINDOW_AUTOSIZE);
    cv::imshow("Brute Force Matches", match_img);
    
    std::cout << "Brute force matching results:" << std::endl;
    std::cout << "  Features in img1: " << kpts1.size() << std::endl;
    std::cout << "  Features in img2: " << kpts2.size() << std::endl;
    std::cout << "  Total matches: " << matches.size() << std::endl;
    std::cout << "  Good matches (top 50%): " << good_matches.size() << std::endl;
    
    if (!matches.empty()) {
        std::cout << "  Best match distance: " << matches[0].distance << std::endl;
        std::cout << "  Worst match distance: " << matches.back().distance << std::endl;
    }
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateCrossCheckMatching(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n=== Cross-Check Matching ===" << std::endl;
    
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    
    // Detect features
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    
    orb->detectAndCompute(gray1, cv::noArray(), kpts1, desc1);
    orb->detectAndCompute(gray2, cv::noArray(), kpts2, desc2);
    
    // Regular matching
    cv::BFMatcher bf_regular(cv::NORM_HAMMING, false);
    std::vector<cv::DMatch> matches_regular;
    bf_regular.match(desc1, desc2, matches_regular);
    
    // Cross-check matching
    cv::BFMatcher bf_crosscheck(cv::NORM_HAMMING, true);  // Enable cross-check
    std::vector<cv::DMatch> matches_crosscheck;
    bf_crosscheck.match(desc1, desc2, matches_crosscheck);
    
    // Draw both results for comparison
    cv::Mat regular_img, crosscheck_img;
    cv::drawMatches(img1, kpts1, img2, kpts2, matches_regular, regular_img);
    cv::drawMatches(img1, kpts1, img2, kpts2, matches_crosscheck, crosscheck_img);
    
    // Create comparison image
    cv::Mat comparison = cv::Mat::zeros(std::max(regular_img.rows, crosscheck_img.rows), 
                                       regular_img.cols + crosscheck_img.cols, CV_8UC3);
    
    regular_img.copyTo(comparison(cv::Rect(0, 0, regular_img.cols, regular_img.rows)));
    crosscheck_img.copyTo(comparison(cv::Rect(regular_img.cols, 0, crosscheck_img.cols, crosscheck_img.rows)));
    
    // Add labels
    cv::putText(comparison, "Regular Matching (" + std::to_string(matches_regular.size()) + ")",
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Cross-Check (" + std::to_string(matches_crosscheck.size()) + ")",
               cv::Point(regular_img.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Cross-Check Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("Cross-Check Comparison", comparison);
    
    std::cout << "Cross-check matching comparison:" << std::endl;
    std::cout << "  Regular matches: " << matches_regular.size() << std::endl;
    std::cout << "  Cross-check matches: " << matches_crosscheck.size() << std::endl;
    std::cout << "  Filtered out: " << (matches_regular.size() - matches_crosscheck.size()) << std::endl;
    std::cout << "  Retention rate: " << static_cast<float>(matches_crosscheck.size()) / matches_regular.size() * 100 << "%" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateRatioTest(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n=== Ratio Test (Lowe's Test) ===" << std::endl;
    
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    
    // Detect features
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    
    orb->detectAndCompute(gray1, cv::noArray(), kpts1, desc1);
    orb->detectAndCompute(gray2, cv::noArray(), kpts2, desc2);
    
    // Use kNN matcher to find 2 best matches for each descriptor
    cv::BFMatcher bf_matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf_matcher.knnMatch(desc1, desc2, knn_matches, 2);
    
    // Apply ratio test
    std::vector<cv::DMatch> good_matches;
    float ratio_threshold = 0.75f;
    
    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() >= 2) {
            if (match_pair[0].distance < ratio_threshold * match_pair[1].distance) {
                good_matches.push_back(match_pair[0]);
            }
        }
    }
    
    // Compare with different ratio thresholds
    std::vector<cv::DMatch> strict_matches, loose_matches;
    float strict_ratio = 0.6f;
    float loose_ratio = 0.9f;
    
    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() >= 2) {
            if (match_pair[0].distance < strict_ratio * match_pair[1].distance) {
                strict_matches.push_back(match_pair[0]);
            }
            if (match_pair[0].distance < loose_ratio * match_pair[1].distance) {
                loose_matches.push_back(match_pair[0]);
            }
        }
    }
    
    // Visualize different ratio thresholds
    cv::Mat good_img, strict_img, loose_img;
    cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, good_img);
    cv::drawMatches(img1, kpts1, img2, kpts2, strict_matches, strict_img);
    cv::drawMatches(img1, kpts1, img2, kpts2, loose_matches, loose_img);
    
    cv::namedWindow("Ratio 0.75", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ratio 0.6 (Strict)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ratio 0.9 (Loose)", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Ratio 0.75", good_img);
    cv::imshow("Ratio 0.6 (Strict)", strict_img);
    cv::imshow("Ratio 0.9 (Loose)", loose_img);
    
    std::cout << "Ratio test results:" << std::endl;
    std::cout << "  Total kNN matches: " << knn_matches.size() << std::endl;
    std::cout << "  Strict ratio (0.6): " << strict_matches.size() << " matches" << std::endl;
    std::cout << "  Standard ratio (0.75): " << good_matches.size() << " matches" << std::endl;
    std::cout << "  Loose ratio (0.9): " << loose_matches.size() << " matches" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateFLANNMatcher(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n=== FLANN Matcher ===" << std::endl;
    
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    
    // Use AKAZE for float descriptors (better for FLANN)
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    
    akaze->detectAndCompute(gray1, cv::noArray(), kpts1, desc1);
    akaze->detectAndCompute(gray2, cv::noArray(), kpts2, desc2);
    
    if (desc1.empty() || desc2.empty()) {
        std::cout << "No descriptors found for FLANN matching." << std::endl;
        return;
    }
    
    // Convert descriptors to float if they're not already
    if (desc1.type() != CV_32F) {
        desc1.convertTo(desc1, CV_32F);
        desc2.convertTo(desc2, CV_32F);
    }
    
    // FLANN matcher
    cv::FlannBasedMatcher flann_matcher;
    
    std::vector<std::vector<cv::DMatch>> knn_matches;
    
    auto start = cv::getTickCount();
    flann_matcher.knnMatch(desc1, desc2, knn_matches, 2);
    auto flann_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
    
    // Brute force matcher for comparison
    cv::BFMatcher bf_matcher;
    std::vector<std::vector<cv::DMatch>> bf_knn_matches;
    
    start = cv::getTickCount();
    bf_matcher.knnMatch(desc1, desc2, bf_knn_matches, 2);
    auto bf_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
    
    // Apply ratio test to both
    std::vector<cv::DMatch> flann_good, bf_good;
    float ratio = 0.75f;
    
    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() >= 2 && match_pair[0].distance < ratio * match_pair[1].distance) {
            flann_good.push_back(match_pair[0]);
        }
    }
    
    for (const auto& match_pair : bf_knn_matches) {
        if (match_pair.size() >= 2 && match_pair[0].distance < ratio * match_pair[1].distance) {
            bf_good.push_back(match_pair[0]);
        }
    }
    
    // Visualize results
    cv::Mat flann_img, bf_img;
    cv::drawMatches(img1, kpts1, img2, kpts2, flann_good, flann_img);
    cv::drawMatches(img1, kpts1, img2, kpts2, bf_good, bf_img);
    
    cv::namedWindow("FLANN Matches", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Brute Force Matches", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("FLANN Matches", flann_img);
    cv::imshow("Brute Force Matches", bf_img);
    
    std::cout << "FLANN vs Brute Force comparison:" << std::endl;
    std::cout << "  FLANN matches: " << flann_good.size() << " (time: " << flann_time << "ms)" << std::endl;
    std::cout << "  Brute Force matches: " << bf_good.size() << " (time: " << bf_time << "ms)" << std::endl;
    std::cout << "  Speed improvement: " << bf_time / flann_time << "x" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateGeometricVerification(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n=== Geometric Verification ===" << std::endl;
    
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    
    // Detect features
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    
    orb->detectAndCompute(gray1, cv::noArray(), kpts1, desc1);
    orb->detectAndCompute(gray2, cv::noArray(), kpts2, desc2);
    
    // Match features
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);
    
    // Apply ratio test
    std::vector<cv::DMatch> good_matches;
    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() >= 2 && match_pair[0].distance < 0.75f * match_pair[1].distance) {
            good_matches.push_back(match_pair[0]);
        }
    }
    
    if (good_matches.size() < 4) {
        std::cout << "Not enough matches for geometric verification." << std::endl;
        return;
    }
    
    // Extract matched points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : good_matches) {
        pts1.push_back(kpts1[match.queryIdx].pt);
        pts2.push_back(kpts2[match.trainIdx].pt);
    }
    
    // Find homography using RANSAC
    std::vector<uchar> inlier_mask;
    cv::Mat homography = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, inlier_mask);
    
    // Separate inliers and outliers
    std::vector<cv::DMatch> inlier_matches, outlier_matches;
    for (size_t i = 0; i < good_matches.size(); i++) {
        if (inlier_mask[i]) {
            inlier_matches.push_back(good_matches[i]);
        } else {
            outlier_matches.push_back(good_matches[i]);
        }
    }
    
    // Visualize results
    cv::Mat all_matches_img, inlier_img;
    cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, all_matches_img);
    cv::drawMatches(img1, kpts1, img2, kpts2, inlier_matches, inlier_img);
    
    // Draw bounding box transformation
    if (!homography.empty()) {
        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(0, 0);
        corners[1] = cv::Point2f(img1.cols, 0);
        corners[2] = cv::Point2f(img1.cols, img1.rows);
        corners[3] = cv::Point2f(0, img1.rows);
        
        std::vector<cv::Point2f> transformed_corners(4);
        cv::perspectiveTransform(corners, transformed_corners, homography);
        
        // Draw transformed bounding box on the second image
        for (size_t i = 0; i < 4; i++) {
            transformed_corners[i].x += img1.cols;  // Offset for concatenated image
        }
        
        cv::polylines(inlier_img, transformed_corners, true, cv::Scalar(0, 255, 0), 3);
    }
    
    cv::namedWindow("All Matches", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Geometric Inliers", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("All Matches", all_matches_img);
    cv::imshow("Geometric Inliers", inlier_img);
    
    std::cout << "Geometric verification results:" << std::endl;
    std::cout << "  Initial matches: " << good_matches.size() << std::endl;
    std::cout << "  Geometric inliers: " << inlier_matches.size() << std::endl;
    std::cout << "  Outliers removed: " << outlier_matches.size() << std::endl;
    std::cout << "  Inlier ratio: " << static_cast<float>(inlier_matches.size()) / good_matches.size() * 100 << "%" << std::endl;
    
    if (!homography.empty()) {
        std::cout << "  Homography found successfully" << std::endl;
    } else {
        std::cout << "  Could not find valid homography" << std::endl;
    }
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Feature Matching ===" << std::endl;
    
    // Create test images
    auto test_images = createMatchingTestImages();
    cv::Mat img1 = test_images.first;
    cv::Mat img2 = test_images.second;
    
    // Try to load real images for additional testing
    cv::Mat real_img1 = cv::imread("../data/test.jpg");
    cv::Mat real_img2;
    
    if (!real_img1.empty()) {
        // Create a transformed version of the real image
        cv::Mat rotation_matrix = cv::getRotationMatrix2D(
            cv::Point2f(real_img1.cols/2, real_img1.rows/2), 20, 0.8);
        cv::warpAffine(real_img1, real_img2, rotation_matrix, real_img1.size());
        
        std::cout << "Using loaded images for some demonstrations." << std::endl;
        
        // Use real images for some demonstrations
        demonstrateBruteForceMatching(real_img1, real_img2);
        demonstrateCrossCheckMatching(img1, img2);      // Use synthetic for clear comparison
        demonstrateRatioTest(real_img1, real_img2);
        demonstrateFLANNMatcher(img1, img2);            // Use synthetic for FLANN
        demonstrateGeometricVerification(real_img1, real_img2);
    } else {
        std::cout << "Using synthetic test images." << std::endl;
        
        // Demonstrate all feature matching techniques
        demonstrateBruteForceMatching(img1, img2);
        demonstrateCrossCheckMatching(img1, img2);
        demonstrateRatioTest(img1, img2);
        demonstrateFLANNMatcher(img1, img2);
        demonstrateGeometricVerification(img1, img2);
    }
    
    std::cout << "\n✓ Feature Matching demonstration complete!" << std::endl;
    std::cout << "Feature matching is essential for image registration, object recognition, and tracking." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Brute force matcher tests all possible descriptor pairs
 * 2. Cross-check filtering improves match quality by bidirectional verification
 * 3. Ratio test (Lowe's test) filters ambiguous matches using distance ratios
 * 4. FLANN is faster for large descriptor sets using approximate search
 * 5. Geometric verification using RANSAC removes spatial outliers
 * 6. Different matching strategies trade off between speed and accuracy
 * 7. Match filtering is crucial for robust feature-based applications
 * 8. Homography estimation validates geometric consistency of matches
 */

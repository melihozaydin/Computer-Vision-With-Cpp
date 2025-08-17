/**
 * 31-Homography_RANSAC.cpp
 * 
 * Robust geometric transformations using RANSAC.
 * 
 * Concepts covered:
 * - Homography estimation
 * - RANSAC algorithm
 * - Outlier rejection
 * - Fundamental matrix
 * - Essential matrix estimation
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <random>

cv::Mat createTestScene() {
    cv::Mat scene = cv::Mat::zeros(600, 800, CV_8UC3);
    
    // Create a scene with recognizable patterns
    // Checkerboard pattern
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 6; j++) {
            if ((i + j) % 2 == 0) {
                cv::rectangle(scene, cv::Point(j * 50 + 100, i * 50 + 100), 
                             cv::Point((j + 1) * 50 + 100, (i + 1) * 50 + 100), 
                             cv::Scalar(255, 255, 255), -1);
            }
        }
    }
    
    // Add some geometric shapes
    cv::circle(scene, cv::Point(600, 150), 40, cv::Scalar(0, 255, 0), -1);
    cv::rectangle(scene, cv::Point(550, 250), cv::Point(650, 350), cv::Scalar(255, 0, 0), -1);
    
    // Draw triangle using polygon
    std::vector<cv::Point> triangle_pts = {cv::Point(600, 400), cv::Point(550, 500), cv::Point(650, 500)};
    cv::fillPoly(scene, triangle_pts, cv::Scalar(0, 0, 255));
    
    // Add text for reference
    cv::putText(scene, "RANSAC", cv::Point(150, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 0), 3);
    
    return scene;
}

cv::Mat transformImage(const cv::Mat& src, const cv::Mat& homography) {
    cv::Mat transformed;
    cv::warpPerspective(src, transformed, homography, src.size());
    return transformed;
}

void demonstrateBasicHomography(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n=== Basic Homography Estimation ===" << std::endl;
    
    // Detect ORB features
    cv::Ptr<cv::ORB> detector = cv::ORB::create(1000);
    
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    
    detector->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    detector->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    
    // Match features
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);
    
    // Sort matches by distance
    std::sort(matches.begin(), matches.end(), 
              [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
    
    // Keep only good matches
    const int num_good_matches = std::min(100, static_cast<int>(matches.size()));
    std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + num_good_matches);
    
    // Extract matched points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : good_matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }
    
    if (pts1.size() < 4) {
        std::cout << "Not enough matches found for homography estimation." << std::endl;
        return;
    }
    
    // Estimate homography without RANSAC
    cv::Mat H_basic = cv::findHomography(pts1, pts2, 0);  // 0 = no robust method
    
    // Estimate homography with RANSAC
    cv::Mat H_ransac = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0);
    
    // Transform images
    cv::Mat warped_basic, warped_ransac;
    cv::warpPerspective(img1, warped_basic, H_basic, img1.size());
    cv::warpPerspective(img1, warped_ransac, H_ransac, img1.size());
    
    // Create comparison display
    cv::Mat display;
    cv::hconcat(img2, warped_basic, display);
    cv::Mat bottom_row;
    cv::hconcat(warped_ransac, img1, bottom_row);
    cv::vconcat(display, bottom_row, display);
    
    // Add labels
    cv::putText(display, "Target", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Basic Homography", cv::Point(img1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "RANSAC Homography", cv::Point(10, img1.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Original", cv::Point(img1.cols + 10, img1.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::namedWindow("Homography Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("Homography Comparison", display);
    
    std::cout << "Homography estimation results:" << std::endl;
    std::cout << "  - Total matches found: " << matches.size() << std::endl;
    std::cout << "  - Good matches used: " << good_matches.size() << std::endl;
    std::cout << "  - RANSAC provides more robust estimation" << std::endl;
    std::cout << "  - Basic method sensitive to outliers" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateRANSACOutlierRejection(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n=== RANSAC Outlier Rejection ===" << std::endl;
    
    // Detect features
    cv::Ptr<cv::ORB> detector = cv::ORB::create(1000);
    
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    
    detector->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    detector->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    
    // Match features
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);
    
    // Extract points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }
    
    if (pts1.size() < 4) {
        std::cout << "Not enough matches for RANSAC demonstration." << std::endl;
        return;
    }
    
    // Find homography with RANSAC
    cv::Mat mask;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);
    
    // Count inliers and outliers
    int inliers = cv::countNonZero(mask);
    int outliers = mask.rows - inliers;
    
    // Draw matches with inlier/outlier classification
    cv::Mat img_matches;
    std::vector<cv::DMatch> inlier_matches, outlier_matches;
    
    for (size_t i = 0; i < matches.size(); i++) {
        if (mask.at<uchar>(i)) {
            inlier_matches.push_back(matches[i]);
        } else {
            outlier_matches.push_back(matches[i]);
        }
    }
    
    // Draw inliers in green
    cv::drawMatches(img1, kp1, img2, kp2, inlier_matches, img_matches,
                   cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // Draw outliers in red (overlay)
    cv::Mat img_outliers;
    cv::drawMatches(img1, kp1, img2, kp2, outlier_matches, img_outliers,
                   cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 255),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // Combine inliers and outliers
    cv::addWeighted(img_matches, 0.7, img_outliers, 0.3, 0, img_matches);
    
    cv::namedWindow("RANSAC Inliers/Outliers", cv::WINDOW_AUTOSIZE);
    cv::imshow("RANSAC Inliers/Outliers", img_matches);
    
    // Add text overlay
    cv::putText(img_matches, "Green: Inliers", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(img_matches, "Red: Outliers", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    cv::putText(img_matches, "Inliers: " + std::to_string(inliers), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(img_matches, "Outliers: " + std::to_string(outliers), cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    std::cout << "RANSAC outlier rejection results:" << std::endl;
    std::cout << "  - Total matches: " << matches.size() << std::endl;
    std::cout << "  - Inliers: " << inliers << " (" << (100.0 * inliers / matches.size()) << "%)" << std::endl;
    std::cout << "  - Outliers: " << outliers << " (" << (100.0 * outliers / matches.size()) << "%)" << std::endl;
    std::cout << "  - RANSAC threshold: 3.0 pixels" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateRANSACParameters(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n=== RANSAC Parameter Effects ===" << std::endl;
    
    // Detect features
    cv::Ptr<cv::ORB> detector = cv::ORB::create(500);
    
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    
    detector->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    detector->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    
    // Match features
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);
    
    // Extract points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }
    
    if (pts1.size() < 4) {
        std::cout << "Not enough matches for parameter demonstration." << std::endl;
        return;
    }
    
    // Test different RANSAC thresholds
    std::vector<double> thresholds = {1.0, 3.0, 5.0, 10.0};
    cv::Mat display = cv::Mat::zeros(img1.rows * 2, img1.cols * 2, CV_8UC3);
    
    for (size_t i = 0; i < thresholds.size(); i++) {
        cv::Mat mask;
        cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, thresholds[i], mask);
        
        int inliers = cv::countNonZero(mask);
        
        // Transform image
        cv::Mat warped;
        cv::warpPerspective(img1, warped, H, img1.size());
        
        // Place in grid
        int row = i / 2;
        int col = i % 2;
        cv::Rect roi(col * img1.cols, row * img1.rows, img1.cols, img1.rows);
        warped.copyTo(display(roi));
        
        // Add label
        std::string label = "T:" + std::to_string(thresholds[i]) + " I:" + std::to_string(inliers);
        cv::putText(display, label, 
                   cv::Point(col * img1.cols + 10, row * img1.rows + 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    
    cv::namedWindow("RANSAC Parameter Effects", cv::WINDOW_AUTOSIZE);
    cv::imshow("RANSAC Parameter Effects", display);
    
    std::cout << "RANSAC parameter effects:" << std::endl;
    std::cout << "  - Lower threshold = fewer inliers, more precise" << std::endl;
    std::cout << "  - Higher threshold = more inliers, less precise" << std::endl;
    std::cout << "  - Threshold should match expected noise level" << std::endl;
    std::cout << "  - Too low: insufficient inliers for estimation" << std::endl;
    std::cout << "  - Too high: includes outliers, degrades quality" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateFundamentalMatrix(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n=== Fundamental Matrix Estimation ===" << std::endl;
    
    // Detect features
    cv::Ptr<cv::ORB> detector = cv::ORB::create(500);
    
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    
    detector->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    detector->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    
    // Match features
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);
    
    // Keep only good matches
    std::sort(matches.begin(), matches.end(), 
              [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
    
    const int num_good = std::min(100, static_cast<int>(matches.size()));
    std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + num_good);
    
    // Extract points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : good_matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }
    
    if (pts1.size() < 8) {
        std::cout << "Not enough matches for fundamental matrix estimation." << std::endl;
        return;
    }
    
    // Estimate fundamental matrix
    cv::Mat mask;
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, mask);
    
    // Draw epipolar lines
    cv::Mat img_epi1, img_epi2;
    img1.copyTo(img_epi1);
    img2.copyTo(img_epi2);
    
    // Select a few points to draw epipolar lines
    std::vector<cv::Point2f> sample_pts1, sample_pts2;
    for (size_t i = 0; i < pts1.size(); i += 10) {  // Every 10th point
        if (mask.at<uchar>(i)) {
            sample_pts1.push_back(pts1[i]);
            sample_pts2.push_back(pts2[i]);
        }
    }
    
    if (!sample_pts1.empty()) {
        // Compute epipolar lines
        std::vector<cv::Vec3f> lines1, lines2;
        cv::computeCorrespondEpilines(sample_pts2, 2, F, lines1);  // Lines in img1 for points in img2
        cv::computeCorrespondEpilines(sample_pts1, 1, F, lines2);  // Lines in img2 for points in img1
        
        // Draw epipolar lines and points
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        
        for (size_t i = 0; i < sample_pts1.size() && i < 10; i++) {
            cv::Scalar color(dis(gen), dis(gen), dis(gen));
            
            // Draw points
            cv::circle(img_epi1, sample_pts1[i], 5, color, -1);
            cv::circle(img_epi2, sample_pts2[i], 5, color, -1);
            
            // Draw epipolar lines
            cv::Vec3f line1 = lines1[i];
            cv::Vec3f line2 = lines2[i];
            
            // Line equation: ax + by + c = 0
            // y = -(ax + c) / b
            cv::Point2f pt1_1(0, -line1[2] / line1[1]);
            cv::Point2f pt1_2(img_epi1.cols, -(line1[0] * img_epi1.cols + line1[2]) / line1[1]);
            cv::line(img_epi1, pt1_1, pt1_2, color, 2);
            
            cv::Point2f pt2_1(0, -line2[2] / line2[1]);
            cv::Point2f pt2_2(img_epi2.cols, -(line2[0] * img_epi2.cols + line2[2]) / line2[1]);
            cv::line(img_epi2, pt2_1, pt2_2, color, 2);
        }
    }
    
    // Create display
    cv::Mat display;
    cv::hconcat(img_epi1, img_epi2, display);
    
    cv::putText(display, "Epipolar Lines", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Corresponding Points", cv::Point(img1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Fundamental Matrix", cv::WINDOW_AUTOSIZE);
    cv::imshow("Fundamental Matrix", display);
    
    std::cout << "Fundamental matrix characteristics:" << std::endl;
    std::cout << "  - Describes epipolar geometry between two views" << std::endl;
    std::cout << "  - Points lie on corresponding epipolar lines" << std::endl;
    std::cout << "  - Rank 2 matrix (3x3 with determinant = 0)" << std::endl;
    std::cout << "  - Independent of camera calibration" << std::endl;
    std::cout << "  - Used in stereo vision and structure from motion" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateRobustEstimationComparison() {
    std::cout << "\n=== Robust Estimation Methods Comparison ===" << std::endl;
    
    // Create synthetic point correspondences with outliers
    std::vector<cv::Point2f> pts1, pts2;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> noise(-2.0, 2.0);
    std::uniform_real_distribution<> outlier(-50.0, 50.0);
    
    // Generate inlier correspondences
    cv::Mat true_H = (cv::Mat_<double>(3, 3) << 
                      1.2, 0.1, 50,
                      -0.1, 1.1, 30,
                      0.001, -0.001, 1);
    
    for (int i = 0; i < 50; i++) {
        cv::Point2f pt1(100 + i * 8, 100 + (i % 10) * 20);
        
        // Transform point
        cv::Mat pt_homo = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Mat pt2_homo = true_H * pt_homo;
        cv::Point2f pt2(pt2_homo.at<double>(0) / pt2_homo.at<double>(2),
                        pt2_homo.at<double>(1) / pt2_homo.at<double>(2));
        
        // Add noise
        pt2.x += noise(gen);
        pt2.y += noise(gen);
        
        pts1.push_back(pt1);
        pts2.push_back(pt2);
    }
    
    // Add outliers
    for (int i = 0; i < 20; i++) {
        cv::Point2f pt1(100 + outlier(gen), 100 + outlier(gen));
        cv::Point2f pt2(200 + outlier(gen), 150 + outlier(gen));
        pts1.push_back(pt1);
        pts2.push_back(pt2);
    }
    
    // Test different methods
    std::vector<std::string> methods = {"None", "RANSAC", "LMEDS", "RHO"};
    std::vector<int> method_flags = {0, cv::RANSAC, cv::LMEDS, cv::RHO};
    
    cv::Mat results = cv::Mat::zeros(200, 800, CV_8UC3);
    
    for (size_t i = 0; i < methods.size(); i++) {
        cv::Mat mask;
        cv::Mat H;
        
        if (method_flags[i] == 0) {
            H = cv::findHomography(pts1, pts2, 0);  // No robust method
        } else {
            H = cv::findHomography(pts1, pts2, method_flags[i], 3.0, mask);
        }
        
        // Calculate reprojection error
        std::vector<cv::Point2f> projected_pts;
        cv::perspectiveTransform(pts1, projected_pts, H);
        
        double total_error = 0;
        int valid_points = 0;
        for (size_t j = 0; j < pts2.size(); j++) {
            double error = cv::norm(pts2[j] - projected_pts[j]);
            if (error < 100) {  // Ignore extreme outliers for average calculation
                total_error += error;
                valid_points++;
            }
        }
        double avg_error = total_error / valid_points;
        
        int inliers = method_flags[i] == 0 ? static_cast<int>(pts1.size()) : cv::countNonZero(mask);
        
        // Draw results
        int x_offset = i * 200;
        cv::putText(results, methods[i], cv::Point(x_offset + 10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(results, "Inliers: " + std::to_string(inliers), 
                   cv::Point(x_offset + 10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        cv::putText(results, "Error: " + std::to_string(avg_error).substr(0, 5), 
                   cv::Point(x_offset + 10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        
        // Draw quality indicator
        cv::Scalar color = avg_error < 5.0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(results, cv::Point(x_offset + 10, 100), cv::Point(x_offset + 180, 180), color, 3);
    }
    
    cv::namedWindow("Robust Estimation Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("Robust Estimation Comparison", results);
    
    std::cout << "Robust estimation methods comparison:" << std::endl;
    std::cout << "  - None: Least squares, sensitive to outliers" << std::endl;
    std::cout << "  - RANSAC: Random sampling, most popular" << std::endl;
    std::cout << "  - LMEDS: Least median of squares, robust" << std::endl;
    std::cout << "  - RHO: PROSAC variant, faster convergence" << std::endl;
    std::cout << "  - Total points: " << pts1.size() << " (70 inliers, 20 outliers)" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Homography and RANSAC Demonstration ===" << std::endl;
    
    // Create test images
    cv::Mat img1 = createTestScene();
    
    // Create a transformed version
    cv::Mat H_transform = (cv::Mat_<double>(3, 3) << 
                          0.9, 0.2, 50,
                          -0.1, 1.1, 30,
                          0.001, -0.0005, 1);
    
    cv::Mat img2;
    cv::warpPerspective(img1, img2, H_transform, img1.size());
    
    // Add some noise to make matching more challenging
    cv::Mat noise(img2.size(), CV_8UC3);
    cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(20));
    cv::add(img2, noise, img2);
    
    // Try to load real images if available
    cv::Mat real_img1 = cv::imread("../data/homography1.jpg");
    cv::Mat real_img2 = cv::imread("../data/homography2.jpg");
    
    if (!real_img1.empty() && !real_img2.empty()) {
        std::cout << "Using real images for demonstration." << std::endl;
        img1 = real_img1;
        img2 = real_img2;
        
        // Resize if necessary
        if (img1.cols > 800 || img1.rows > 600) {
            cv::resize(img1, img1, cv::Size(800, 600));
            cv::resize(img2, img2, cv::Size(800, 600));
        }
    } else {
        std::cout << "Using synthetic images for demonstration." << std::endl;
    }
    
    // Demonstrate RANSAC techniques
    demonstrateBasicHomography(img1, img2);
    demonstrateRANSACOutlierRejection(img1, img2);
    demonstrateRANSACParameters(img1, img2);
    demonstrateFundamentalMatrix(img1, img2);
    demonstrateRobustEstimationComparison();
    
    std::cout << "\n✓ Homography and RANSAC demonstration complete!" << std::endl;
    std::cout << "RANSAC provides robust geometric estimation in the presence of outliers." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. RANSAC iteratively samples minimal sets to find consensus
 * 2. Homography requires at least 4 point correspondences
 * 3. Outlier rejection improves geometric estimation quality
 * 4. Threshold parameter balances precision vs inclusion
 * 5. Fundamental matrix describes epipolar geometry
 * 6. More iterations increase probability of finding best model
 * 7. Inlier ratio affects RANSAC convergence speed
 * 8. Different robust methods suit different scenarios
 * 9. Feature matching quality directly affects results
 * 10. RANSAC is widely applicable beyond homography estimation
 */

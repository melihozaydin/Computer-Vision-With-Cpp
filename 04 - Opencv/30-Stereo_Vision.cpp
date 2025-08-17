/**
 * 30-Stereo_Vision.cpp
 * 
 * Depth estimation from stereo image pairs.
 * 
 * Concepts covered:
 * - Stereo rectification
 * - Disparity map computation
 * - Depth map generation
 * - 3D point cloud creation
 * - Stereo matching algorithms
 */

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <random>

// Create synthetic stereo pair for demonstration
std::pair<cv::Mat, cv::Mat> createStereoTestImages() {
    cv::Mat left = cv::Mat::zeros(400, 600, CV_8UC3);
    cv::Mat right = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // Create scene with objects at different depths
    // Background (far)
    cv::rectangle(left, cv::Point(0, 200), cv::Point(600, 400), cv::Scalar(50, 100, 150), -1);
    cv::rectangle(right, cv::Point(5, 200), cv::Point(605, 400), cv::Scalar(50, 100, 150), -1);
    
    // Mid-ground objects
    cv::circle(left, cv::Point(150, 150), 40, cv::Scalar(0, 200, 0), -1);
    cv::circle(right, cv::Point(145, 150), 40, cv::Scalar(0, 200, 0), -1);
    
    cv::rectangle(left, cv::Point(300, 100), cv::Point(400, 200), cv::Scalar(200, 100, 0), -1);
    cv::rectangle(right, cv::Point(290, 100), cv::Point(390, 200), cv::Scalar(200, 100, 0), -1);
    
    // Foreground objects (close)
    cv::circle(left, cv::Point(450, 120), 30, cv::Scalar(200, 0, 200), -1);
    cv::circle(right, cv::Point(435, 120), 30, cv::Scalar(200, 0, 200), -1);
    
    cv::rectangle(left, cv::Point(80, 250), cv::Point(180, 350), cv::Scalar(255, 255, 0), -1);
    cv::rectangle(right, cv::Point(65, 250), cv::Point(165, 350), cv::Scalar(255, 255, 0), -1);
    
    // Add texture for better matching
    cv::Mat noise_left(left.size(), CV_8UC3);
    cv::Mat noise_right(right.size(), CV_8UC3);
    cv::randu(noise_left, cv::Scalar::all(0), cv::Scalar::all(30));
    cv::randu(noise_right, cv::Scalar::all(0), cv::Scalar::all(30));
    cv::add(left, noise_left, left);
    cv::add(right, noise_right, right);
    
    return std::make_pair(left, right);
}

void demonstrateBasicDisparity(const cv::Mat& left, const cv::Mat& right) {
    std::cout << "\n=== Basic Disparity Computation ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    
    // Create StereoBM object
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
    
    // Set parameters
    int numDisparities = 64;  // Must be divisible by 16
    int blockSize = 15;       // Must be odd
    
    stereo->setNumDisparities(numDisparities);
    stereo->setBlockSize(blockSize);
    stereo->setPreFilterCap(31);
    stereo->setMinDisparity(0);
    stereo->setUniquenessRatio(10);
    stereo->setSpeckleWindowSize(100);
    stereo->setSpeckleRange(32);
    stereo->setDisp12MaxDiff(1);
    
    // Compute disparity
    cv::Mat disparity;
    stereo->compute(left_gray, right_gray, disparity);
    
    // Normalize disparity for visualization
    cv::Mat disparity_vis;
    cv::normalize(disparity, disparity_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    // Create display
    cv::Mat display;
    cv::hconcat(left, right, display);
    
    cv::Mat disparity_colored;
    cv::applyColorMap(disparity_vis, disparity_colored, cv::COLORMAP_JET);
    
    cv::Mat bottom_row;
    cv::hconcat(disparity_colored, disparity_colored, bottom_row);
    
    cv::vconcat(display, bottom_row, display);
    
    // Add labels
    cv::putText(display, "Left Image", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Right Image", cv::Point(left.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Disparity Map", cv::Point(10, left.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Stereo Disparity", cv::WINDOW_AUTOSIZE);
    cv::imshow("Stereo Disparity", display);
    
    std::cout << "Basic disparity computation parameters:" << std::endl;
    std::cout << "  - Number of disparities: " << numDisparities << std::endl;
    std::cout << "  - Block size: " << blockSize << std::endl;
    std::cout << "  - Algorithm: StereoBM (Block Matching)" << std::endl;
    std::cout << "  - Closer objects appear brighter in disparity map" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateSGBMStereo(const cv::Mat& left, const cv::Mat& right) {
    std::cout << "\n=== Semi-Global Block Matching (SGBM) ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    
    // Create StereoSGBM object
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
    
    // Set parameters for better quality
    int numDisparities = 64;
    int blockSize = 5;
    
    stereo->setMinDisparity(0);
    stereo->setNumDisparities(numDisparities);
    stereo->setBlockSize(blockSize);
    stereo->setP1(8 * blockSize * blockSize);
    stereo->setP2(32 * blockSize * blockSize);
    stereo->setDisp12MaxDiff(1);
    stereo->setUniquenessRatio(10);
    stereo->setSpeckleWindowSize(100);
    stereo->setSpeckleRange(32);
    stereo->setPreFilterCap(63);
    stereo->setMode(cv::StereoSGBM::MODE_SGBM);
    
    // Compute disparity
    cv::Mat disparity;
    stereo->compute(left_gray, right_gray, disparity);
    
    // Convert to proper format and normalize
    cv::Mat disparity_8u;
    disparity.convertTo(disparity_8u, CV_8U, 255.0 / (numDisparities * 16.0));
    
    // Apply color map
    cv::Mat disparity_colored;
    cv::applyColorMap(disparity_8u, disparity_colored, cv::COLORMAP_TURBO);
    
    // Create comparison display
    cv::Mat display;
    cv::hconcat(left, disparity_colored, display);
    
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "SGBM Disparity", cv::Point(left.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("SGBM Stereo", cv::WINDOW_AUTOSIZE);
    cv::imshow("SGBM Stereo", display);
    
    std::cout << "SGBM advantages:" << std::endl;
    std::cout << "  - Better quality than basic block matching" << std::endl;
    std::cout << "  - Handles textureless regions better" << std::endl;
    std::cout << "  - More robust to noise and illumination changes" << std::endl;
    std::cout << "  - Smoother disparity maps with fewer holes" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateDepthFromDisparity(const cv::Mat& left, const cv::Mat& right) {
    std::cout << "\n=== Depth Map Generation ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    
    // Compute disparity using SGBM
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
    int numDisparities = 64;
    int blockSize = 5;
    
    stereo->setMinDisparity(0);
    stereo->setNumDisparities(numDisparities);
    stereo->setBlockSize(blockSize);
    stereo->setP1(8 * blockSize * blockSize);
    stereo->setP2(32 * blockSize * blockSize);
    
    cv::Mat disparity;
    stereo->compute(left_gray, right_gray, disparity);
    
    // Convert disparity to depth
    // For real stereo setup: depth = (focal_length * baseline) / disparity
    // For this demo, we'll use simplified conversion
    cv::Mat depth;
    cv::Mat disparity_float;
    disparity.convertTo(disparity_float, CV_32F, 1.0/16.0);
    
    // Simulate depth calculation (focal_length * baseline = 1000 for demo)
    cv::divide(1000.0, disparity_float, depth, CV_32F);
    
    // Handle invalid disparities (set to maximum depth)
    cv::Mat mask = disparity_float <= 0;
    depth.setTo(100.0, mask);
    
    // Normalize depth for visualization
    cv::Mat depth_vis;
    cv::normalize(depth, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    // Create depth visualization with different color maps
    cv::Mat depth_jet, depth_hot, depth_cool;
    cv::applyColorMap(depth_vis, depth_jet, cv::COLORMAP_JET);
    cv::applyColorMap(depth_vis, depth_hot, cv::COLORMAP_HOT);
    cv::applyColorMap(depth_vis, depth_cool, cv::COLORMAP_COOL);
    
    // Create display grid
    cv::Mat top_row, bottom_row, display;
    cv::hconcat(left, depth_jet, top_row);
    cv::hconcat(depth_hot, depth_cool, bottom_row);
    cv::vconcat(top_row, bottom_row, display);
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Depth (JET)", cv::Point(left.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Depth (HOT)", cv::Point(10, left.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Depth (COOL)", cv::Point(left.cols + 10, left.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Depth Maps", cv::WINDOW_AUTOSIZE);
    cv::imshow("Depth Maps", display);
    
    // Print depth statistics
    double min_depth, max_depth;
    cv::minMaxLoc(depth, &min_depth, &max_depth, nullptr, nullptr, ~mask);
    
    std::cout << "Depth map statistics:" << std::endl;
    std::cout << "  - Min depth: " << min_depth << " units" << std::endl;
    std::cout << "  - Max depth: " << max_depth << " units" << std::endl;
    std::cout << "  - Depth formula: depth = (focal_length * baseline) / disparity" << std::endl;
    std::cout << "  - Closer objects = smaller disparity = larger depth values" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrate3DPointCloud(const cv::Mat& left, const cv::Mat& right) {
    std::cout << "\n=== 3D Point Cloud Generation ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    
    // Compute disparity
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
    stereo->setMinDisparity(0);
    stereo->setNumDisparities(64);
    stereo->setBlockSize(5);
    
    cv::Mat disparity;
    stereo->compute(left_gray, right_gray, disparity);
    
    // Camera parameters (simulated for demonstration)
    cv::Mat Q = (cv::Mat_<double>(4, 4) << 
                 1, 0, 0, -left.cols/2,
                 0, 1, 0, -left.rows/2,
                 0, 0, 0, 500,  // focal length
                 0, 0, 1.0/50, 0);  // 1/baseline
    
    // Generate 3D points
    cv::Mat points_3d;
    cv::reprojectImageTo3D(disparity, points_3d, Q);
    
    // Create point cloud visualization (top-down view)
    cv::Mat cloud_vis = cv::Mat::zeros(400, 400, CV_8UC3);
    
    for (int y = 0; y < points_3d.rows; y += 5) {  // Sample every 5th point
        for (int x = 0; x < points_3d.cols; x += 5) {
            cv::Vec3f point = points_3d.at<cv::Vec3f>(y, x);
            
            if (point[2] > 0 && point[2] < 100) {  // Valid depth range
                int vis_x = static_cast<int>(point[0] * 2 + 200);
                int vis_y = static_cast<int>(point[2] * 4);
                
                if (vis_x >= 0 && vis_x < 400 && vis_y >= 0 && vis_y < 400) {
                    // Color based on depth
                    int depth_color = static_cast<int>(point[2] * 2.55);
                    depth_color = cv::saturate_cast<uchar>(depth_color);
                    cloud_vis.at<cv::Vec3b>(vis_y, vis_x) = cv::Vec3b(depth_color, 255 - depth_color, 128);
                }
            }
        }
    }
    
    // Create side view
    cv::Mat cloud_side = cv::Mat::zeros(400, 400, CV_8UC3);
    
    for (int y = 0; y < points_3d.rows; y += 5) {
        for (int x = 0; x < points_3d.cols; x += 5) {
            cv::Vec3f point = points_3d.at<cv::Vec3f>(y, x);
            
            if (point[2] > 0 && point[2] < 100) {
                int vis_x = static_cast<int>(point[2] * 4);
                int vis_y = static_cast<int>(-point[1] + 200);
                
                if (vis_x >= 0 && vis_x < 400 && vis_y >= 0 && vis_y < 400) {
                    int depth_color = static_cast<int>(point[0] + 128);
                    depth_color = cv::saturate_cast<uchar>(depth_color);
                    cloud_side.at<cv::Vec3b>(vis_y, vis_x) = cv::Vec3b(128, depth_color, 255 - depth_color);
                }
            }
        }
    }
    
    // Create display
    cv::Mat display;
    cv::hconcat(cloud_vis, cloud_side, display);
    
    cv::putText(display, "Top View (X-Z)", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Side View (Z-Y)", cv::Point(410, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("3D Point Cloud Views", cv::WINDOW_AUTOSIZE);
    cv::imshow("3D Point Cloud Views", display);
    
    std::cout << "3D point cloud characteristics:" << std::endl;
    std::cout << "  - Each pixel becomes a 3D point (X, Y, Z)" << std::endl;
    std::cout << "  - Requires camera calibration matrix (Q)" << std::endl;
    std::cout << "  - Top view shows spatial distribution" << std::endl;
    std::cout << "  - Side view shows depth profile" << std::endl;
    std::cout << "  - Colors represent different dimensions" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateParameterTuning(const cv::Mat& left, const cv::Mat& right) {
    std::cout << "\n=== Stereo Parameter Tuning ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    
    // Test different parameter combinations
    std::vector<int> block_sizes = {5, 11, 21};
    std::vector<int> num_disparities = {32, 64, 96};
    
    cv::Mat display = cv::Mat::zeros(left.rows * 3, left.cols * 3, CV_8UC3);
    
    for (size_t i = 0; i < block_sizes.size(); i++) {
        for (size_t j = 0; j < num_disparities.size(); j++) {
            cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
            
            int blockSize = block_sizes[i];
            int numDisp = num_disparities[j];
            
            stereo->setMinDisparity(0);
            stereo->setNumDisparities(numDisp);
            stereo->setBlockSize(blockSize);
            stereo->setP1(8 * blockSize * blockSize);
            stereo->setP2(32 * blockSize * blockSize);
            
            cv::Mat disparity;
            stereo->compute(left_gray, right_gray, disparity);
            
            // Normalize and color
            cv::Mat disparity_vis;
            cv::normalize(disparity, disparity_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::Mat disparity_colored;
            cv::applyColorMap(disparity_vis, disparity_colored, cv::COLORMAP_JET);
            
            // Place in grid
            cv::Rect roi(j * left.cols, i * left.rows, left.cols, left.rows);
            disparity_colored.copyTo(display(roi));
            
            // Add parameter labels
            std::string label = "B:" + std::to_string(blockSize) + " D:" + std::to_string(numDisp);
            cv::putText(display, label, 
                       cv::Point(j * left.cols + 10, i * left.rows + 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    cv::namedWindow("Parameter Tuning", cv::WINDOW_AUTOSIZE);
    cv::imshow("Parameter Tuning", display);
    
    std::cout << "Parameter effects:" << std::endl;
    std::cout << "  - Block Size: Larger = smoother, smaller = more detail" << std::endl;
    std::cout << "  - Num Disparities: More = larger depth range" << std::endl;
    std::cout << "  - Trade-off between accuracy and computational cost" << std::endl;
    std::cout << "  - Scene-dependent optimization required" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Stereo Vision Demonstration ===" << std::endl;
    
    // Try to load real stereo images
    cv::Mat left_real = cv::imread("../data/stereo_left.jpg");
    cv::Mat right_real = cv::imread("../data/stereo_right.jpg");
    
    cv::Mat left, right;
    
    if (!left_real.empty() && !right_real.empty()) {
        std::cout << "Using real stereo images." << std::endl;
        left = left_real;
        right = right_real;
        
        // Ensure same size
        if (left.size() != right.size()) {
            cv::Size common_size(std::min(left.cols, right.cols), std::min(left.rows, right.rows));
            cv::resize(left, left, common_size);
            cv::resize(right, right, common_size);
        }
    } else {
        std::cout << "Using synthetic stereo images." << std::endl;
        auto stereo_pair = createStereoTestImages();
        left = stereo_pair.first;
        right = stereo_pair.second;
    }
    
    // Resize if too large
    if (left.cols > 800 || left.rows > 600) {
        cv::Size new_size(800, 600);
        cv::resize(left, left, new_size);
        cv::resize(right, right, new_size);
    }
    
    // Demonstrate stereo vision techniques
    demonstrateBasicDisparity(left, right);
    demonstrateSGBMStereo(left, right);
    demonstrateDepthFromDisparity(left, right);
    demonstrate3DPointCloud(left, right);
    demonstrateParameterTuning(left, right);
    
    std::cout << "\n✓ Stereo Vision demonstration complete!" << std::endl;
    std::cout << "Stereo vision enables 3D depth perception from two 2D images." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Stereo vision estimates depth from image pair disparities
 * 2. StereoBM uses simple block matching for fast computation
 * 3. StereoSGBM provides better quality through semi-global optimization
 * 4. Disparity is inversely proportional to depth
 * 5. Camera calibration is crucial for accurate 3D reconstruction
 * 6. Parameter tuning balances quality vs computational cost
 * 7. Texture-rich regions produce better disparity estimates
 * 8. 3D point clouds can be generated from disparity maps
 * 9. Stereo rectification ensures epipolar geometry
 * 10. Real-world applications require careful camera setup and calibration
 */

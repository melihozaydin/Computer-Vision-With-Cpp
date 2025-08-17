/**
 * 34-Image_Stitching.cpp
 * 
 * Creating panoramas from multiple images.
 * 
 * Concepts covered:
 * - Feature-based stitching
 * - Homography estimation
 * - Image blending
 * - Seam finding
 * - Bundle adjustment
 */

#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

// Create a sequence of overlapping images for stitching
std::vector<cv::Mat> createPanoramaSequence() {
    std::vector<cv::Mat> images;
    
    // Base scene dimensions
    int scene_width = 800;
    int scene_height = 400;
    int image_width = 400;
    int image_height = 300;
    
    // Create a wide synthetic scene
    cv::Mat full_scene = cv::Mat::zeros(scene_height, scene_width, CV_8UC3);
    
    // Create background gradient
    for (int y = 0; y < scene_height; y++) {
        for (int x = 0; x < scene_width; x++) {
            int r = 100 + (x * 155) / scene_width;
            int g = 150 + (y * 105) / scene_height;
            int b = 200 - (x * 100) / scene_width;
            full_scene.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }
    
    // Add distinctive features across the scene
    
    // Left region features
    cv::circle(full_scene, cv::Point(100, 150), 40, cv::Scalar(255, 0, 0), -1);
    cv::rectangle(full_scene, cv::Point(50, 250), cv::Point(150, 350), cv::Scalar(0, 255, 0), -1);
    cv::putText(full_scene, "LEFT", cv::Point(60, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // Center region features
    cv::ellipse(full_scene, cv::Point(400, 120), cv::Size(60, 30), 45, 0, 360, cv::Scalar(0, 0, 255), -1);
    cv::circle(full_scene, cv::Point(350, 250), 35, cv::Scalar(255, 255, 0), -1);
    cv::circle(full_scene, cv::Point(450, 250), 35, cv::Scalar(255, 0, 255), -1);
    cv::putText(full_scene, "CENTER", cv::Point(320, 350), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // Right region features
    cv::rectangle(full_scene, cv::Point(650, 100), cv::Point(750, 200), cv::Scalar(0, 255, 255), -1);
    cv::circle(full_scene, cv::Point(700, 280), 45, cv::Scalar(128, 0, 128), -1);
    cv::putText(full_scene, "RIGHT", cv::Point(650, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // Add noise for more realistic matching
    cv::Mat noise(full_scene.size(), CV_8UC3);
    cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(20));
    cv::add(full_scene, noise, full_scene);
    
    // Extract overlapping images
    std::vector<cv::Rect> image_regions = {
        cv::Rect(0, 50, image_width, image_height),        // Left image
        cv::Rect(200, 50, image_width, image_height),      // Center-left image  
        cv::Rect(400, 50, image_width, image_height)       // Right image
    };
    
    for (const auto& region : image_regions) {
        // Ensure region is within bounds
        cv::Rect safe_region = region & cv::Rect(0, 0, scene_width, scene_height);
        cv::Mat cropped = full_scene(safe_region).clone();
        
        // Add slight variations to make stitching more challenging
        cv::Mat variation;
        cropped.convertTo(variation, -1, 0.95 + 0.1 * (rand() % 100) / 100.0, 
                         -5 + (rand() % 10));
        
        images.push_back(variation);
    }
    
    return images;
}

void demonstrateBasicStitching(const std::vector<cv::Mat>& images) {
    std::cout << "\n=== Basic Image Stitching ===" << std::endl;
    
    if (images.size() < 2) {
        std::cout << "Need at least 2 images for stitching." << std::endl;
        return;
    }
    
    // Create stitcher
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
    
    // Perform stitching
    cv::Mat panorama;
    cv::Stitcher::Status status = stitcher->stitch(images, panorama);
    
    if (status == cv::Stitcher::OK) {
        // Create comparison display
        cv::Mat input_display;
        
        // Resize inputs for display
        std::vector<cv::Mat> resized_inputs;
        for (const auto& img : images) {
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(200, 150));
            resized_inputs.push_back(resized);
        }
        
        // Concatenate input images
        cv::hconcat(resized_inputs, input_display);
        
        // Resize panorama for display
        cv::Mat panorama_display;
        double scale = std::min(800.0 / panorama.cols, 300.0 / panorama.rows);
        cv::resize(panorama, panorama_display, cv::Size(), scale, scale);
        
        // Make sure both images have the same width for vconcat
        int target_width = std::max(input_display.cols, panorama_display.cols);
        
        cv::Mat input_padded = cv::Mat::zeros(input_display.rows, target_width, input_display.type());
        cv::Mat panorama_padded = cv::Mat::zeros(panorama_display.rows, target_width, panorama_display.type());
        
        input_display.copyTo(input_padded(cv::Rect(0, 0, input_display.cols, input_display.rows)));
        panorama_display.copyTo(panorama_padded(cv::Rect(0, 0, panorama_display.cols, panorama_display.rows)));
        
        // Create final display
        cv::Mat display;
        cv::vconcat(input_padded, panorama_padded, display);
        
        // Add labels
        cv::putText(display, "Input Images", cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "Stitched Panorama", cv::Point(10, input_display.rows + 50), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        cv::namedWindow("Basic Stitching", cv::WINDOW_AUTOSIZE);
        cv::imshow("Basic Stitching", display);
        
        std::cout << "Stitching successful!" << std::endl;
        std::cout << "  - Input images: " << images.size() << std::endl;
        std::cout << "  - Panorama size: " << panorama.cols << "x" << panorama.rows << std::endl;
        std::cout << "  - Automatic feature detection and matching" << std::endl;
        std::cout << "  - Homography estimation and blending" << std::endl;
    } else {
        std::cout << "Stitching failed with status: " << static_cast<int>(status) << std::endl;
        
        // Display input images anyway
        cv::Mat input_display;
        std::vector<cv::Mat> resized_inputs;
        for (const auto& img : images) {
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(200, 150));
            resized_inputs.push_back(resized);
        }
        cv::hconcat(resized_inputs, input_display);
        
        cv::putText(input_display, "Stitching Failed", cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        
        cv::namedWindow("Basic Stitching", cv::WINDOW_AUTOSIZE);
        cv::imshow("Basic Stitching", input_display);
    }
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateManualStitching(const std::vector<cv::Mat>& images) {
    std::cout << "\n=== Manual Feature-Based Stitching ===" << std::endl;
    
    if (images.size() < 2) {
        std::cout << "Need at least 2 images for manual stitching." << std::endl;
        return;
    }
    
    // Take first two images for manual stitching
    cv::Mat img1 = images[0];
    cv::Mat img2 = images[1];
    
    // Feature detection
    cv::Ptr<cv::ORB> detector = cv::ORB::create(1000);
    
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    
    detector->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    detector->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    
    // Feature matching
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);
    
    // Filter good matches
    std::sort(matches.begin(), matches.end(), 
              [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
    
    const int num_good_matches = std::min(50, static_cast<int>(matches.size()));
    std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + num_good_matches);
    
    // Extract matched points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : good_matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }
    
    if (pts1.size() < 4) {
        std::cout << "Not enough good matches for homography estimation." << std::endl;
        return;
    }
    
    // Estimate homography
    cv::Mat mask;
    cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC, 3.0, mask);
    
    // Create output canvas
    int canvas_width = img1.cols + img2.cols;
    int canvas_height = std::max(img1.rows, img2.rows);
    cv::Mat canvas = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    
    // Place first image
    img1.copyTo(canvas(cv::Rect(0, 0, img1.cols, img1.rows)));
    
    // Warp and blend second image
    cv::Mat warped;
    cv::warpPerspective(img2, warped, H, cv::Size(canvas_width, canvas_height));
    
    // Simple blending in overlap region
    for (int y = 0; y < canvas.rows; y++) {
        for (int x = 0; x < canvas.cols; x++) {
            cv::Vec3b canvas_pixel = canvas.at<cv::Vec3b>(y, x);
            cv::Vec3b warped_pixel = warped.at<cv::Vec3b>(y, x);
            
            if (warped_pixel != cv::Vec3b(0, 0, 0)) {  // Non-black pixel in warped image
                if (canvas_pixel != cv::Vec3b(0, 0, 0)) {  // Overlap region
                    // Alpha blending
                    canvas.at<cv::Vec3b>(y, x) = canvas_pixel * 0.5 + warped_pixel * 0.5;
                } else {  // Only warped image
                    canvas.at<cv::Vec3b>(y, x) = warped_pixel;
                }
            }
        }
    }
    
    // Draw matches
    cv::Mat matches_display;
    cv::drawMatches(img1, kp1, img2, kp2, good_matches, matches_display,
                   cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // Create final display
    cv::Mat top_display = matches_display;
    cv::Mat bottom_display = canvas;
    
    // Resize for display
    if (bottom_display.cols > 800) {
        double scale = 800.0 / bottom_display.cols;
        cv::resize(bottom_display, bottom_display, cv::Size(), scale, scale);
    }
    if (top_display.cols > 800) {
        double scale = 800.0 / top_display.cols;
        cv::resize(top_display, top_display, cv::Size(), scale, scale);
    }
    
    cv::Mat display;
    
    // Make sure both images have the same width for vconcat
    int target_width = std::max(top_display.cols, bottom_display.cols);
    
    cv::Mat top_padded = cv::Mat::zeros(top_display.rows, target_width, top_display.type());
    cv::Mat bottom_padded = cv::Mat::zeros(bottom_display.rows, target_width, bottom_display.type());
    
    top_display.copyTo(top_padded(cv::Rect(0, 0, top_display.cols, top_display.rows)));
    bottom_display.copyTo(bottom_padded(cv::Rect(0, 0, bottom_display.cols, bottom_display.rows)));
    
    cv::vconcat(top_padded, bottom_padded, display);
    
    cv::namedWindow("Manual Stitching", cv::WINDOW_AUTOSIZE);
    cv::imshow("Manual Stitching", display);
    
    int inliers = cv::countNonZero(mask);
    std::cout << "Manual stitching results:" << std::endl;
    std::cout << "  - Features detected: " << kp1.size() << " and " << kp2.size() << std::endl;
    std::cout << "  - Total matches: " << matches.size() << std::endl;
    std::cout << "  - Good matches: " << good_matches.size() << std::endl;
    std::cout << "  - RANSAC inliers: " << inliers << std::endl;
    std::cout << "  - Final canvas size: " << canvas.cols << "x" << canvas.rows << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateStitchingModes(const std::vector<cv::Mat>& images) {
    std::cout << "\n=== Different Stitching Modes ===" << std::endl;
    
    if (images.size() < 2) {
        std::cout << "Need at least 2 images for mode comparison." << std::endl;
        return;
    }
    
    // Test different stitching modes
    std::vector<cv::Stitcher::Mode> modes = {
        cv::Stitcher::PANORAMA,
        cv::Stitcher::SCANS
    };
    
    std::vector<std::string> mode_names = {"PANORAMA", "SCANS"};
    
    cv::Mat display = cv::Mat::zeros(400, 800, CV_8UC3);
    
    for (size_t i = 0; i < modes.size(); i++) {
        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(modes[i]);
        
        cv::Mat panorama;
        cv::Stitcher::Status status = stitcher->stitch(images, panorama);
        
        cv::Mat mode_result;
        if (status == cv::Stitcher::OK) {
            // Resize for display
            double scale = std::min(390.0 / panorama.cols, 190.0 / panorama.rows);
            cv::resize(panorama, mode_result, cv::Size(), scale, scale);
        } else {
            mode_result = cv::Mat::zeros(190, 390, CV_8UC3);
            cv::putText(mode_result, "FAILED", cv::Point(150, 100), 
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
        
        // Place in display
        cv::Rect roi(5, i * 200 + 5, mode_result.cols, mode_result.rows);
        mode_result.copyTo(display(roi));
        
        // Add mode label
        cv::putText(display, mode_names[i], cv::Point(10, i * 200 + 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        std::cout << mode_names[i] << " mode: " << 
                     (status == cv::Stitcher::OK ? "SUCCESS" : "FAILED") << std::endl;
    }
    
    cv::namedWindow("Stitching Modes", cv::WINDOW_AUTOSIZE);
    cv::imshow("Stitching Modes", display);
    
    std::cout << "\nStitching mode differences:" << std::endl;
    std::cout << "  - PANORAMA: For typical panoramic photography" << std::endl;
    std::cout << "  - SCANS: For scanned documents or planar scenes" << std::endl;
    std::cout << "  - Mode affects feature detection and blending strategies" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateBlendingMethods(const std::vector<cv::Mat>& images) {
    std::cout << "\n=== Blending Methods Comparison ===" << std::endl;
    
    if (images.size() < 2) {
        std::cout << "Need at least 2 images for blending demonstration." << std::endl;
        return;
    }
    
    // Manual implementation of different blending methods
    cv::Mat img1 = images[0];
    cv::Mat img2 = images[1];
    
    // Simple homography for demonstration
    cv::Mat H = (cv::Mat_<double>(3, 3) << 
                 1.0, 0.0, img1.cols * 0.7,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0);
    
    // Create canvas
    int canvas_width = img1.cols + static_cast<int>(img1.cols * 0.7);
    int canvas_height = img1.rows;
    
    // Method 1: No blending (overlay)
    cv::Mat no_blend = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    img1.copyTo(no_blend(cv::Rect(0, 0, img1.cols, img1.rows)));
    cv::Mat warped1;
    cv::warpPerspective(img2, warped1, H, cv::Size(canvas_width, canvas_height));
    
    for (int y = 0; y < canvas_height; y++) {
        for (int x = 0; x < canvas_width; x++) {
            cv::Vec3b warped_pixel = warped1.at<cv::Vec3b>(y, x);
            if (warped_pixel != cv::Vec3b(0, 0, 0)) {
                no_blend.at<cv::Vec3b>(y, x) = warped_pixel;
            }
        }
    }
    
    // Method 2: Alpha blending
    cv::Mat alpha_blend = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    img1.copyTo(alpha_blend(cv::Rect(0, 0, img1.cols, img1.rows)));
    cv::Mat warped2;
    cv::warpPerspective(img2, warped2, H, cv::Size(canvas_width, canvas_height));
    
    for (int y = 0; y < canvas_height; y++) {
        for (int x = 0; x < canvas_width; x++) {
            cv::Vec3b canvas_pixel = alpha_blend.at<cv::Vec3b>(y, x);
            cv::Vec3b warped_pixel = warped2.at<cv::Vec3b>(y, x);
            
            if (warped_pixel != cv::Vec3b(0, 0, 0) && canvas_pixel != cv::Vec3b(0, 0, 0)) {
                // Overlap region - blend
                alpha_blend.at<cv::Vec3b>(y, x) = canvas_pixel * 0.5 + warped_pixel * 0.5;
            } else if (warped_pixel != cv::Vec3b(0, 0, 0)) {
                alpha_blend.at<cv::Vec3b>(y, x) = warped_pixel;
            }
        }
    }
    
    // Method 3: Feathering (distance-weighted blending)
    cv::Mat feather_blend = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    img1.copyTo(feather_blend(cv::Rect(0, 0, img1.cols, img1.rows)));
    cv::Mat warped3;
    cv::warpPerspective(img2, warped3, H, cv::Size(canvas_width, canvas_height));
    
    // Find overlap region
    cv::Mat mask1 = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC1);
    mask1(cv::Rect(0, 0, img1.cols, img1.rows)) = 255;
    
    cv::Mat mask2 = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC1);
    for (int y = 0; y < canvas_height; y++) {
        for (int x = 0; x < canvas_width; x++) {
            if (warped3.at<cv::Vec3b>(y, x) != cv::Vec3b(0, 0, 0)) {
                mask2.at<uchar>(y, x) = 255;
            }
        }
    }
    
    cv::Mat overlap_mask;
    cv::bitwise_and(mask1, mask2, overlap_mask);
    
    // Distance transform for feathering
    cv::Mat dist1, dist2;
    cv::distanceTransform(mask1, dist1, cv::DIST_L2, 3);
    cv::distanceTransform(mask2, dist2, cv::DIST_L2, 3);
    
    for (int y = 0; y < canvas_height; y++) {
        for (int x = 0; x < canvas_width; x++) {
            if (overlap_mask.at<uchar>(y, x) > 0) {
                float d1 = dist1.at<float>(y, x);
                float d2 = dist2.at<float>(y, x);
                float weight = d1 / (d1 + d2);
                
                cv::Vec3b p1 = feather_blend.at<cv::Vec3b>(y, x);
                cv::Vec3b p2 = warped3.at<cv::Vec3b>(y, x);
                
                feather_blend.at<cv::Vec3b>(y, x) = p1 * (1 - weight) + p2 * weight;
            } else if (warped3.at<cv::Vec3b>(y, x) != cv::Vec3b(0, 0, 0)) {
                feather_blend.at<cv::Vec3b>(y, x) = warped3.at<cv::Vec3b>(y, x);
            }
        }
    }
    
    // Create comparison display
    std::vector<cv::Mat> blend_results = {no_blend, alpha_blend, feather_blend};
    std::vector<std::string> method_names = {"No Blending", "Alpha Blend", "Feathering"};
    
    cv::Mat display = cv::Mat::zeros(canvas_height * 3, canvas_width, CV_8UC3);
    
    for (size_t i = 0; i < blend_results.size(); i++) {
        cv::Rect roi(0, i * canvas_height, canvas_width, canvas_height);
        blend_results[i].copyTo(display(roi));
        
        cv::putText(display, method_names[i], cv::Point(10, i * canvas_height + 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    
    // Resize for display
    if (display.cols > 800) {
        double scale = 800.0 / display.cols;
        cv::resize(display, display, cv::Size(), scale, scale);
    }
    
    cv::namedWindow("Blending Methods", cv::WINDOW_AUTOSIZE);
    cv::imshow("Blending Methods", display);
    
    std::cout << "Blending method comparison:" << std::endl;
    std::cout << "  - No blending: Visible seams and discontinuities" << std::endl;
    std::cout << "  - Alpha blending: Simple averaging, reduces seams" << std::endl;
    std::cout << "  - Feathering: Distance-weighted, smoother transitions" << std::endl;
    std::cout << "  - Advanced methods use multi-band blending" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Image Stitching Demonstration ===" << std::endl;
    
    // Try to load real images for stitching
    std::vector<cv::Mat> real_images;
    for (int i = 1; i <= 5; i++) {
        std::string filename = "../data/stitch" + std::to_string(i) + ".jpg";
        cv::Mat img = cv::imread(filename);
        if (!img.empty()) {
            // Resize if too large
            if (img.cols > 600 || img.rows > 400) {
                cv::resize(img, img, cv::Size(600, 400));
            }
            real_images.push_back(img);
        }
    }
    
    std::vector<cv::Mat> images_to_use;
    
    if (real_images.size() >= 2) {
        std::cout << "Using " << real_images.size() << " real images for stitching." << std::endl;
        images_to_use = real_images;
    } else {
        std::cout << "Creating synthetic image sequence for stitching." << std::endl;
        images_to_use = createPanoramaSequence();
    }
    
    // Demonstrate various stitching techniques
    demonstrateBasicStitching(images_to_use);
    demonstrateManualStitching(images_to_use);
    demonstrateStitchingModes(images_to_use);
    demonstrateBlendingMethods(images_to_use);
    
    std::cout << "\n✓ Image Stitching demonstration complete!" << std::endl;
    std::cout << "Image stitching enables creation of wide panoramas from multiple overlapping images." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Stitching requires overlapping images with similar exposure
 * 2. Feature-based methods detect and match keypoints automatically
 * 3. Homography estimation aligns images geometrically
 * 4. RANSAC removes outlier matches for robust estimation
 * 5. Image blending reduces visible seams in overlap regions
 * 6. Bundle adjustment optimizes global consistency
 * 7. Seam finding algorithms determine optimal blend boundaries
 * 8. Different modes suit different scene types
 * 9. Multi-band blending preserves both low and high frequencies
 * 10. Modern methods handle parallax and moving objects
 */

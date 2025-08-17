/**
 * 10-Distance_Transform.cpp
 * 
 * Distance transform and skeletonization for shape analysis.
 * 
 * Concepts covered:
 * - Euclidean distance transform
 * - Manhattan distance transform
 * - Chessboard distance
 * - Skeletonization
 * - Watershed preparation
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

cv::Mat createDistanceTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC1);
    
    // Add various shapes for distance transform analysis
    cv::rectangle(image, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255), -1);
    cv::circle(image, cv::Point(300, 100), 50, cv::Scalar(255), -1);
    
    // Add a complex shape
    std::vector<cv::Point> contour = {
        cv::Point(450, 50), cv::Point(550, 50), cv::Point(550, 80),
        cv::Point(500, 80), cv::Point(500, 120), cv::Point(550, 120),
        cv::Point(550, 150), cv::Point(450, 150)
    };
    cv::fillPoly(image, contour, cv::Scalar(255));
    
    // Add thin structures
    cv::rectangle(image, cv::Point(100, 200), cv::Point(120, 350), cv::Scalar(255), -1);
    cv::rectangle(image, cv::Point(200, 250), cv::Point(350, 270), cv::Scalar(255), -1);
    
    // Add some connected components
    cv::circle(image, cv::Point(400, 250), 30, cv::Scalar(255), -1);
    cv::circle(image, cv::Point(450, 280), 25, cv::Scalar(255), -1);
    cv::circle(image, cv::Point(380, 300), 20, cv::Scalar(255), -1);
    
    return image;
}

void demonstrateDistanceTypes(const cv::Mat& binary) {
    std::cout << "\n=== Different Distance Transform Types ===" << std::endl;
    
    cv::Mat dist_euclidean, dist_manhattan, dist_chessboard;
    
    // Different distance types
    cv::distanceTransform(binary, dist_euclidean, cv::DIST_L2, 3);
    cv::distanceTransform(binary, dist_manhattan, cv::DIST_L1, 3);
    cv::distanceTransform(binary, dist_chessboard, cv::DIST_C, 3);
    
    // Normalize for visualization
    cv::Mat vis_euclidean, vis_manhattan, vis_chessboard;
    cv::normalize(dist_euclidean, vis_euclidean, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(dist_manhattan, vis_manhattan, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(dist_chessboard, vis_chessboard, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    // Apply color map for better visualization
    cv::Mat colored_euclidean, colored_manhattan, colored_chessboard;
    cv::applyColorMap(vis_euclidean, colored_euclidean, cv::COLORMAP_JET);
    cv::applyColorMap(vis_manhattan, colored_manhattan, cv::COLORMAP_JET);
    cv::applyColorMap(vis_chessboard, colored_chessboard, cv::COLORMAP_JET);
    
    // Set background to black in colored versions
    for (int y = 0; y < binary.rows; y++) {
        for (int x = 0; x < binary.cols; x++) {
            if (binary.at<uchar>(y, x) == 0) {
                colored_euclidean.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                colored_manhattan.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                colored_chessboard.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    
    cv::namedWindow("Original Binary", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Euclidean Distance", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Manhattan Distance", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Chessboard Distance", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original Binary", binary);
    cv::imshow("Euclidean Distance", colored_euclidean);
    cv::imshow("Manhattan Distance", colored_manhattan);
    cv::imshow("Chessboard Distance", colored_chessboard);
    
    // Print some statistics
    double min_val, max_val;
    cv::minMaxLoc(dist_euclidean, &min_val, &max_val);
    std::cout << "Euclidean - Max distance: " << max_val << std::endl;
    
    cv::minMaxLoc(dist_manhattan, &min_val, &max_val);
    std::cout << "Manhattan - Max distance: " << max_val << std::endl;
    
    cv::minMaxLoc(dist_chessboard, &min_val, &max_val);
    std::cout << "Chessboard - Max distance: " << max_val << std::endl;
    
    std::cout << "Distance transform measures distance to nearest background pixel." << std::endl;
    std::cout << "Different metrics give different distance measurements." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateSkeletonization(const cv::Mat& binary) {
    std::cout << "\n=== Skeletonization using Distance Transform ===" << std::endl;
    
    // Method 1: Simple skeleton using distance transform peaks
    cv::Mat dist_transform;
    cv::distanceTransform(binary, dist_transform, cv::DIST_L2, 3);
    
    // Find local maxima in distance transform
    cv::Mat dilated_dist;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(dist_transform, dilated_dist, kernel);
    
    cv::Mat skeleton_simple = (dist_transform == dilated_dist) & (dist_transform > 0);
    
    // Method 2: Morphological skeleton
    cv::Mat skeleton_morph = cv::Mat::zeros(binary.size(), CV_8UC1);
    cv::Mat temp = binary.clone();
    cv::Mat eroded;
    
    while (cv::countNonZero(temp) > 0) {
        cv::erode(temp, eroded, kernel);
        cv::Mat opened;
        cv::morphologyEx(eroded, opened, cv::MORPH_OPEN, kernel);
        cv::Mat subset = eroded - opened;
        skeleton_morph = skeleton_morph | subset;
        temp = eroded.clone();
    }
    
    // Method 3: Alternative thinning implementation (since ximgproc may not be available)
    cv::Mat skeleton_thin = cv::Mat::zeros(binary.size(), CV_8UC1);
    cv::Mat prev = cv::Mat::zeros(binary.size(), CV_8UC1);
    cv::Mat diff;
    
    // Iterative thinning using morphological operations
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    bool done = false;
    int iterations = 0;
    cv::Mat current = binary.clone();
    
    while (!done && iterations < 50) {
        cv::erode(current, skeleton_thin, element);
        cv::dilate(skeleton_thin, skeleton_thin, element);
        cv::subtract(current, skeleton_thin, diff);
        skeleton_thin = skeleton_thin | diff;
        
        cv::absdiff(current, prev, diff);
        done = (cv::countNonZero(diff) == 0);
        
        prev = current.clone();
        current = skeleton_thin.clone();
        iterations++;
    }
    
    // Visualize results
    cv::Mat dist_vis;
    cv::normalize(dist_transform, dist_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::Mat colored_dist;
    cv::applyColorMap(dist_vis, colored_dist, cv::COLORMAP_HOT);
    
    // Set background to black
    for (int y = 0; y < binary.rows; y++) {
        for (int x = 0; x < binary.cols; x++) {
            if (binary.at<uchar>(y, x) == 0) {
                colored_dist.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    
    cv::namedWindow("Original Binary", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Distance Transform", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Simple Skeleton", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Morphological Skeleton", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Thinning Skeleton", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original Binary", binary);
    cv::imshow("Distance Transform", colored_dist);
    cv::imshow("Simple Skeleton", skeleton_simple);
    cv::imshow("Morphological Skeleton", skeleton_morph);
    cv::imshow("Thinning Skeleton", skeleton_thin);
    
    std::cout << "Skeletonization reduces shapes to their essential structure." << std::endl;
    std::cout << "Different methods preserve different properties." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateWatershedPreparation(const cv::Mat& binary) {
    std::cout << "\n=== Watershed Preparation using Distance Transform ===" << std::endl;
    
    // Distance transform
    cv::Mat dist_transform;
    cv::distanceTransform(binary, dist_transform, cv::DIST_L2, 3);
    
    // Find local maxima (seeds for watershed)
    cv::Mat local_maxima;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::dilate(dist_transform, local_maxima, kernel);
    local_maxima = (dist_transform == local_maxima) & (dist_transform > 0);
    
    // Remove small maxima
    double min_val, max_val;
    cv::minMaxLoc(dist_transform, &min_val, &max_val);
    cv::Mat significant_maxima = (dist_transform > max_val * 0.3) & local_maxima;
    
    // Create markers for watershed
    cv::Mat markers;
    cv::connectedComponents(significant_maxima, markers);
    
    // Apply watershed (need 3-channel image)
    cv::Mat binary_3ch;
    cv::cvtColor(binary, binary_3ch, cv::COLOR_GRAY2BGR);
    cv::watershed(binary_3ch, markers);
    
    // Visualize the watershed regions
    cv::Mat watershed_vis = cv::Mat::zeros(markers.size(), CV_8UC3);
    std::vector<cv::Vec3b> colors;
    
    // Generate random colors for each region
    int max_label = 0;
    cv::minMaxLoc(markers, nullptr, reinterpret_cast<double*>(&max_label));
    
    for (int i = 0; i <= max_label; i++) {
        colors.push_back(cv::Vec3b(rand() % 255, rand() % 255, rand() % 255));
    }
    
    for (int y = 0; y < markers.rows; y++) {
        for (int x = 0; x < markers.cols; x++) {
            int label = markers.at<int>(y, x);
            if (label > 0 && label <= max_label) {
                watershed_vis.at<cv::Vec3b>(y, x) = colors[label];
            }
            if (label == -1) {  // Watershed boundaries
                watershed_vis.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
            }
        }
    }
    
    // Create visualization of distance transform with markers
    cv::Mat dist_vis;
    cv::normalize(dist_transform, dist_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::Mat colored_dist;
    cv::applyColorMap(dist_vis, colored_dist, cv::COLORMAP_JET);
    
    // Overlay markers on distance transform
    cv::Mat markers_vis = cv::Mat::zeros(binary.size(), CV_8UC1);
    significant_maxima.copyTo(markers_vis);
    markers_vis *= 255;
    
    cv::namedWindow("Original Binary", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Distance Transform", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Watershed Seeds", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Watershed Segmentation", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original Binary", binary);
    cv::imshow("Distance Transform", colored_dist);
    cv::imshow("Watershed Seeds", markers_vis);
    cv::imshow("Watershed Segmentation", watershed_vis);
    
    std::cout << "Distance transform provides natural seeds for watershed segmentation." << std::endl;
    std::cout << "Local maxima in distance transform indicate object centers." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateFeatureExtraction(const cv::Mat& binary) {
    std::cout << "\n=== Feature Extraction using Distance Transform ===" << std::endl;
    
    cv::Mat dist_transform;
    cv::distanceTransform(binary, dist_transform, cv::DIST_L2, 3);
    
    // Find object centers (local maxima)
    cv::Mat local_maxima;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
    cv::dilate(dist_transform, local_maxima, kernel);
    local_maxima = (dist_transform == local_maxima) & (dist_transform > 2.0);
    
    // Find contours for shape analysis
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Create result image
    cv::Mat result;
    cv::cvtColor(binary, result, cv::COLOR_GRAY2BGR);
    
    // Analyze each object
    std::vector<cv::Point2f> centers;
    std::vector<float> radii;
    
    for (size_t i = 0; i < contours.size(); i++) {
        // Calculate moments
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 == 0) continue;
        
        cv::Point2f center(m.m10 / m.m00, m.m01 / m.m00);
        centers.push_back(center);
        
        // Find maximum distance value within this contour
        cv::Mat mask = cv::Mat::zeros(binary.size(), CV_8UC1);
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{contours[i]}, cv::Scalar(255));
        
        double min_val, max_val;
        cv::minMaxLoc(dist_transform, &min_val, &max_val, nullptr, nullptr, mask);
        radii.push_back(static_cast<float>(max_val));
        
        // Draw contour
        cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
        
        // Draw center
        cv::circle(result, center, 3, cv::Scalar(0, 0, 255), -1);
        
        // Draw inscribed circle (largest circle that fits inside)
        cv::circle(result, center, static_cast<int>(max_val), cv::Scalar(255, 0, 0), 2);
        
        // Add text with measurements
        std::string text = "A:" + std::to_string(static_cast<int>(cv::contourArea(contours[i]))) +
                          " R:" + std::to_string(static_cast<int>(max_val));
        cv::putText(result, text, cv::Point(center.x - 30, center.y - 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }
    
    // Find ridges (high distance values indicating thick parts)
    cv::Mat ridges;
    double min_val, max_val;
    cv::minMaxLoc(dist_transform, &min_val, &max_val);
    cv::threshold(dist_transform, ridges, max_val * 0.7, 255, cv::THRESH_BINARY);
    ridges.convertTo(ridges, CV_8UC1);
    
    // Create combined visualization
    cv::Mat dist_vis;
    cv::normalize(dist_transform, dist_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::Mat colored_dist;
    cv::applyColorMap(dist_vis, colored_dist, cv::COLORMAP_VIRIDIS);
    
    cv::namedWindow("Original Binary", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Distance Transform", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Feature Analysis", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ridges (Thick Parts)", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original Binary", binary);
    cv::imshow("Distance Transform", colored_dist);
    cv::imshow("Feature Analysis", result);
    cv::imshow("Ridges (Thick Parts)", ridges);
    
    std::cout << "Distance transform enables shape analysis:" << std::endl;
    std::cout << "- Object centers at distance transform maxima" << std::endl;
    std::cout << "- Thickness measurement from distance values" << std::endl;
    std::cout << "- Inscribed circle radius from maximum distance" << std::endl;
    std::cout << "Found " << centers.size() << " objects." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Distance Transform ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createDistanceTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg", cv::IMREAD_GRAYSCALE);
    if (!real_image.empty()) {
        cv::Mat binary_real;
        cv::threshold(real_image, binary_real, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        std::cout << "Using loaded image for some demonstrations." << std::endl;
        
        // Use real image for some demonstrations
        demonstrateDistanceTypes(binary_real);
        demonstrateSkeletonization(test_image);  // Use synthetic for skeleton
        demonstrateWatershedPreparation(binary_real);
        demonstrateFeatureExtraction(test_image);  // Use synthetic for features
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all distance transform operations
        demonstrateDistanceTypes(test_image);
        demonstrateSkeletonization(test_image);
        demonstrateWatershedPreparation(test_image);
        demonstrateFeatureExtraction(test_image);
    }
    
    std::cout << "\n✓ Distance Transform demonstration complete!" << std::endl;
    std::cout << "Distance transform is fundamental for shape analysis and segmentation preparation." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Distance transform measures distance to nearest background pixel
 * 2. Different distance metrics (L1, L2, Chebyshev) give different results
 * 3. Skeletonization preserves shape topology while reducing thickness
 * 4. Distance transform provides natural seeds for watershed segmentation
 * 5. Local maxima in distance transform indicate object centers
 * 6. Distance values encode shape thickness information
 * 7. Useful for shape analysis, object separation, and feature extraction
 * 8. Essential preprocessing step for many advanced algorithms
 */

/**
 * 19-Watershed_Segmentation.cpp
 * 
 * Watershed algorithm for image segmentation.
 * 
 * Concepts covered:
 * - Marker-based watershed
 * - Distance transform watershed
 * - Seed point selection
 * - Post-processing
 * - Object separation
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>

cv::Mat createWatershedTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);
    
    // Create overlapping circles to test watershed segmentation
    cv::circle(image, cv::Point(120, 120), 60, cv::Scalar(0, 255, 0), -1);    // Green circle
    cv::circle(image, cv::Point(200, 120), 60, cv::Scalar(0, 0, 255), -1);    // Red circle
    cv::circle(image, cv::Point(160, 200), 60, cv::Scalar(255, 0, 0), -1);    // Blue circle
    
    // Add some rectangular objects
    cv::rectangle(image, cv::Point(50, 250), cv::Point(120, 320), cv::Scalar(255, 255, 0), -1);  // Cyan rectangle
    cv::rectangle(image, cv::Point(280, 250), cv::Point(350, 320), cv::Scalar(255, 0, 255), -1); // Magenta rectangle
    
    // Add noise to make it more challenging
    cv::Mat noise(image.size(), CV_8UC3);
    cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(30));
    cv::add(image, noise, image);
    
    return image;
}

void demonstrateBasicWatershed(const cv::Mat& src) {
    std::cout << "\n=== Basic Watershed Segmentation ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Apply threshold to get binary image
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    // Remove noise using morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat opening;
    cv::morphologyEx(binary, opening, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);
    
    // Sure background area
    cv::Mat sure_bg;
    cv::dilate(opening, sure_bg, kernel, cv::Point(-1, -1), 3);
    
    // Sure foreground area using distance transform
    cv::Mat dist_transform;
    cv::distanceTransform(opening, dist_transform, cv::DIST_L2, 5);
    
    cv::Mat sure_fg;
    double maxVal;
    cv::minMaxLoc(dist_transform, nullptr, &maxVal);
    cv::threshold(dist_transform, sure_fg, 0.7 * maxVal, 255, cv::THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8U);
    
    // Unknown region
    cv::Mat unknown;
    cv::subtract(sure_bg, sure_fg, unknown);
    
    // Marker labelling
    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);
    
    // Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1;
    
    // Mark the region of unknown with zero
    markers.setTo(0, unknown == 255);
    
    // Apply watershed
    cv::Mat result = src.clone();
    cv::watershed(result, markers);
    
    // Create visualization
    cv::Mat watershed_result = src.clone();
    watershed_result.setTo(cv::Scalar(0, 0, 255), markers == -1);  // Mark boundaries in red
    
    // Color code the segments
    cv::Mat colored_segments = cv::Mat::zeros(src.size(), CV_8UC3);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    double minMarker, maxMarker;
    cv::minMaxLoc(markers, &minMarker, &maxMarker);
    
    for (int i = 2; i <= maxMarker; i++) {  // Start from 2 (background is 1, boundary is -1)
        cv::Scalar color(dis(gen), dis(gen), dis(gen));
        colored_segments.setTo(color, markers == i);
    }
    
    // Display intermediate steps
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 3, CV_8UC3);
    
    // Convert grayscale images to 3-channel for display
    cv::Mat binary_3ch, dist_3ch, sure_fg_3ch;
    cv::cvtColor(binary, binary_3ch, cv::COLOR_GRAY2BGR);
    cv::normalize(dist_transform, dist_transform, 0, 255, cv::NORM_MINMAX);
    dist_transform.convertTo(dist_transform, CV_8U);
    cv::cvtColor(dist_transform, dist_3ch, cv::COLOR_GRAY2BGR);
    cv::cvtColor(sure_fg, sure_fg_3ch, cv::COLOR_GRAY2BGR);
    
    // Top row
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    binary_3ch.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    dist_3ch.copyTo(display(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Bottom row
    sure_fg_3ch.copyTo(display(cv::Rect(0, src.rows, src.cols, src.rows)));
    watershed_result.copyTo(display(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    colored_segments.copyTo(display(cv::Rect(src.cols * 2, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Binary", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Distance Transform", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Sure Foreground", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Watershed Result", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Colored Segments", cv::Point(src.cols * 2 + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Basic Watershed", cv::WINDOW_AUTOSIZE);
    cv::imshow("Basic Watershed", display);
    
    std::cout << "Basic watershed steps:" << std::endl;
    std::cout << "  1. Binary thresholding with Otsu" << std::endl;
    std::cout << "  2. Morphological opening to remove noise" << std::endl;
    std::cout << "  3. Distance transform to find sure foreground" << std::endl;
    std::cout << "  4. Connected components for initial markers" << std::endl;
    std::cout << "  5. Watershed algorithm for final segmentation" << std::endl;
    std::cout << "  Number of segments found: " << maxMarker - 1 << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateMarkerBasedWatershed(const cv::Mat& src) {
    std::cout << "\n=== Marker-Based Watershed ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Apply Gaussian blur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);
    
    // Apply gradient magnitude (morphological gradient)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat gradient;
    cv::morphologyEx(blurred, gradient, cv::MORPH_GRADIENT, kernel);
    
    // Threshold to create binary image
    cv::Mat binary;
    cv::threshold(gradient, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    // Create markers manually by finding local maxima
    cv::Mat dist_transform;
    cv::distanceTransform(255 - binary, dist_transform, cv::DIST_L2, 5);
    
    // Find local maxima as seed points
    cv::Mat local_maxima;
    cv::dilate(dist_transform, local_maxima, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::compare(dist_transform, local_maxima, local_maxima, cv::CMP_EQ);
    
    // Filter small maxima
    cv::Mat filtered_maxima;
    cv::threshold(dist_transform, filtered_maxima, 0.3 * 255, 255, cv::THRESH_BINARY);
    filtered_maxima.convertTo(filtered_maxima, CV_8U);
    cv::bitwise_and(local_maxima, filtered_maxima, local_maxima);
    
    // Create markers from local maxima
    cv::Mat markers;
    cv::connectedComponents(local_maxima, markers);
    
    // Apply watershed to gradient image
    cv::Mat gradient_3ch;
    cv::cvtColor(gradient, gradient_3ch, cv::COLOR_GRAY2BGR);
    cv::watershed(gradient_3ch, markers);
    
    // Visualize results
    cv::Mat result = src.clone();
    
    // Color boundaries
    result.setTo(cv::Scalar(0, 0, 255), markers == -1);
    
    // Create colored segments
    cv::Mat colored_result = cv::Mat::zeros(src.size(), CV_8UC3);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(50, 255);
    
    double minMarker, maxMarker;
    cv::minMaxLoc(markers, &minMarker, &maxMarker);
    
    for (int i = 1; i <= maxMarker; i++) {
        cv::Scalar color(dis(gen), dis(gen), dis(gen));
        colored_result.setTo(color, markers == i);
    }
    
    // Add boundaries to colored result
    colored_result.setTo(cv::Scalar(255, 255, 255), markers == -1);
    
    // Display results
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    // Convert single channel images to 3-channel for display
    cv::Mat gradient_disp, maxima_disp;
    cv::cvtColor(gradient, gradient_disp, cv::COLOR_GRAY2BGR);
    cv::cvtColor(local_maxima, maxima_disp, cv::COLOR_GRAY2BGR);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    gradient_disp.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    maxima_disp.copyTo(display(cv::Rect(0, src.rows, src.cols, src.rows)));
    colored_result.copyTo(display(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Gradient", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Local Maxima", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Watershed Result", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Marker-Based Watershed", cv::WINDOW_AUTOSIZE);
    cv::imshow("Marker-Based Watershed", display);
    
    std::cout << "Marker-based watershed features:" << std::endl;
    std::cout << "  - Uses gradient image as input" << std::endl;
    std::cout << "  - Local maxima detection for seed points" << std::endl;
    std::cout << "  - Better control over segmentation" << std::endl;
    std::cout << "  - Reduces over-segmentation" << std::endl;
    std::cout << "  Number of regions: " << maxMarker << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateInteractiveWatershed(const cv::Mat& src) {
    std::cout << "\n=== Interactive Watershed ===" << std::endl;
    
    // For demonstration, we'll simulate interactive marker placement
    cv::Mat markers = cv::Mat::zeros(src.size(), CV_32S);
    
    // Simulate user-placed markers (in real application, these would come from mouse clicks)
    std::vector<cv::Point> foreground_seeds = {
        cv::Point(120, 120),  // Green circle center
        cv::Point(200, 120),  // Red circle center
        cv::Point(160, 200),  // Blue circle center
        cv::Point(85, 285),   // Cyan rectangle center
        cv::Point(315, 285)   // Magenta rectangle center
    };
    
    std::vector<cv::Point> background_seeds = {
        cv::Point(50, 50),    // Top-left background
        cv::Point(350, 50),   // Top-right background
        cv::Point(50, 350),   // Bottom-left background
        cv::Point(350, 350)   // Bottom-right background
    };
    
    // Create markers
    int marker_id = 1;
    
    // Background markers
    for (const auto& seed : background_seeds) {
        cv::circle(markers, seed, 10, cv::Scalar(marker_id), -1);
        marker_id++;
    }
    
    // Foreground markers
    for (const auto& seed : foreground_seeds) {
        cv::circle(markers, seed, 10, cv::Scalar(marker_id), -1);
        marker_id++;
    }
    
    // Apply watershed
    cv::Mat result = src.clone();
    cv::watershed(result, markers);
    
    // Visualize markers and result
    cv::Mat marker_vis = src.clone();
    
    // Draw background seeds in blue
    for (const auto& seed : background_seeds) {
        cv::circle(marker_vis, seed, 8, cv::Scalar(255, 0, 0), -1);
        cv::circle(marker_vis, seed, 12, cv::Scalar(255, 255, 255), 2);
    }
    
    // Draw foreground seeds in green
    for (const auto& seed : foreground_seeds) {
        cv::circle(marker_vis, seed, 8, cv::Scalar(0, 255, 0), -1);
        cv::circle(marker_vis, seed, 12, cv::Scalar(255, 255, 255), 2);
    }
    
    // Create colored segmentation result
    cv::Mat segmentation_result = cv::Mat::zeros(src.size(), CV_8UC3);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    double minMarker, maxMarker;
    cv::minMaxLoc(markers, &minMarker, &maxMarker);
    
    for (int i = 1; i <= maxMarker; i++) {
        cv::Scalar color(dis(gen), dis(gen), dis(gen));
        segmentation_result.setTo(color, markers == i);
    }
    
    // Mark boundaries in white
    segmentation_result.setTo(cv::Scalar(255, 255, 255), markers == -1);
    
    // Create display
    cv::Mat display = cv::Mat::zeros(src.rows, src.cols * 3, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    marker_vis.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    segmentation_result.copyTo(display(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Seeds (Blue=BG, Green=FG)", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Segmentation", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Interactive Watershed", cv::WINDOW_AUTOSIZE);
    cv::imshow("Interactive Watershed", display);
    
    std::cout << "Interactive watershed advantages:" << std::endl;
    std::cout << "  - User controls marker placement" << std::endl;
    std::cout << "  - Better semantic segmentation" << std::endl;
    std::cout << "  - Reduced over-segmentation" << std::endl;
    std::cout << "  - Suitable for complex scenes" << std::endl;
    std::cout << "  Number of user-defined regions: " << marker_id - 1 << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateWatershedPostProcessing(const cv::Mat& src) {
    std::cout << "\n=== Watershed Post-Processing ===" << std::endl;
    
    // Apply basic watershed first
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat opening;
    cv::morphologyEx(binary, opening, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);
    
    cv::Mat dist_transform;
    cv::distanceTransform(opening, dist_transform, cv::DIST_L2, 5);
    
    cv::Mat sure_fg;
    double maxVal;
    cv::minMaxLoc(dist_transform, nullptr, &maxVal);
    cv::threshold(dist_transform, sure_fg, 0.7 * maxVal, 255, cv::THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8U);
    
    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);
    markers = markers + 1;
    
    cv::Mat sure_bg;
    cv::dilate(opening, sure_bg, kernel, cv::Point(-1, -1), 3);
    cv::Mat unknown;
    cv::subtract(sure_bg, sure_fg, unknown);
    markers.setTo(0, unknown == 255);
    
    cv::Mat result = src.clone();
    cv::watershed(result, markers);
    
    // Post-processing: Remove small regions
    cv::Mat cleaned_markers = markers.clone();
    double minMarker, maxMarker;
    cv::minMaxLoc(markers, &minMarker, &maxMarker);
    
    int min_area = 100;  // Minimum area threshold
    for (int i = 2; i <= maxMarker; i++) {  // Skip background (1) and boundary (-1)
        cv::Mat region_mask = (markers == i);
        int area = cv::countNonZero(region_mask);
        if (area < min_area) {
            cleaned_markers.setTo(1, region_mask);  // Merge with background
        }
    }
    
    // Post-processing: Merge similar regions based on color
    cv::Mat merged_markers = cleaned_markers.clone();
    cv::minMaxLoc(cleaned_markers, &minMarker, &maxMarker);
    
    std::vector<cv::Scalar> region_colors;
    region_colors.resize(maxMarker + 1);
    
    // Calculate average color for each region
    for (int i = 2; i <= maxMarker; i++) {
        cv::Mat region_mask = (cleaned_markers == i);
        if (cv::countNonZero(region_mask) > 0) {
            region_colors[i] = cv::mean(src, region_mask);
        }
    }
    
    // Merge regions with similar colors
    double color_threshold = 50.0;
    for (int i = 2; i <= maxMarker; i++) {
        if (cv::countNonZero(cleaned_markers == i) == 0) continue;
        
        for (int j = i + 1; j <= maxMarker; j++) {
            if (cv::countNonZero(cleaned_markers == j) == 0) continue;
            
            double color_dist = cv::norm(region_colors[i] - region_colors[j]);
            if (color_dist < color_threshold) {
                merged_markers.setTo(i, cleaned_markers == j);  // Merge j into i
            }
        }
    }
    
    // Visualization
    auto createColoredSegmentation = [](const cv::Mat& markers_input) -> cv::Mat {
        cv::Mat colored = cv::Mat::zeros(markers_input.size(), CV_8UC3);
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for consistent colors
        std::uniform_int_distribution<> dis(50, 255);
        
        double minVal, maxVal;
        cv::minMaxLoc(markers_input, &minVal, &maxVal);
        
        for (int i = 2; i <= maxVal; i++) {
            if (cv::countNonZero(markers_input == i) > 0) {
                cv::Scalar color(dis(gen), dis(gen), dis(gen));
                colored.setTo(color, markers_input == i);
            }
        }
        colored.setTo(cv::Scalar(255, 255, 255), markers_input == -1);
        return colored;
    };
    
    cv::Mat original_seg = createColoredSegmentation(markers);
    cv::Mat cleaned_seg = createColoredSegmentation(cleaned_markers);
    cv::Mat merged_seg = createColoredSegmentation(merged_markers);
    
    // Create display
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    original_seg.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    cleaned_seg.copyTo(display(cv::Rect(0, src.rows, src.cols, src.rows)));
    merged_seg.copyTo(display(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Raw Watershed", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Size Filtered", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Color Merged", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Watershed Post-Processing", cv::WINDOW_AUTOSIZE);
    cv::imshow("Watershed Post-Processing", display);
    
    // Count regions in each step
    cv::minMaxLoc(markers, &minMarker, &maxMarker);
    int original_regions = maxMarker - 1;
    
    cv::minMaxLoc(cleaned_markers, &minMarker, &maxMarker);
    int cleaned_regions = 0;
    for (int i = 2; i <= maxMarker; i++) {
        if (cv::countNonZero(cleaned_markers == i) > 0) cleaned_regions++;
    }
    
    cv::minMaxLoc(merged_markers, &minMarker, &maxMarker);
    int merged_regions = 0;
    for (int i = 2; i <= maxMarker; i++) {
        if (cv::countNonZero(merged_markers == i) > 0) merged_regions++;
    }
    
    std::cout << "Post-processing effects:" << std::endl;
    std::cout << "  Original regions: " << original_regions << std::endl;
    std::cout << "  After size filtering: " << cleaned_regions << std::endl;
    std::cout << "  After color merging: " << merged_regions << std::endl;
    std::cout << "  Size reduction: " << (100.0 * (original_regions - merged_regions) / original_regions) << "%" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Watershed Segmentation ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createWatershedTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg");
    if (!real_image.empty()) {
        std::cout << "Using loaded image for demonstrations." << std::endl;
        
        // Use real image for demonstrations
        demonstrateBasicWatershed(real_image);
        demonstrateMarkerBasedWatershed(real_image);
        demonstrateInteractiveWatershed(real_image);
        demonstrateWatershedPostProcessing(real_image);
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all watershed techniques
        demonstrateBasicWatershed(test_image);
        demonstrateMarkerBasedWatershed(test_image);
        demonstrateInteractiveWatershed(test_image);
        demonstrateWatershedPostProcessing(test_image);
    }
    
    std::cout << "\n✓ Watershed Segmentation demonstration complete!" << std::endl;
    std::cout << "Watershed algorithm is powerful for separating touching objects in images." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Watershed treats images as topographical maps with intensity as elevation
 * 2. Distance transform helps identify sure foreground regions
 * 3. Proper marker initialization is crucial for good segmentation
 * 4. Interactive watershed provides user control over segmentation
 * 5. Post-processing helps reduce over-segmentation
 * 6. Morphological operations are essential for preprocessing
 * 7. Gradient images often work better than original images for watershed
 * 8. Connected components labeling creates initial markers
 */

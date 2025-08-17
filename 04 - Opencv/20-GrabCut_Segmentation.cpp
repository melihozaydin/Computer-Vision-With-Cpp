/**
 * 20-GrabCut_Segmentation.cpp
 * 
 * Interactive foreground extraction using GrabCut.
 * 
 * Concepts covered:
 * - GrabCut algorithm
 * - Interactive segmentation
 * - Mask refinement
 * - Background/foreground modeling
 * - Iterative optimization
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat createGrabCutTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // Create a complex scene with foreground and background
    // Background: gradient
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int intensity = (x + y) % 255;
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity/3, intensity/2, intensity);
        }
    }
    
    // Foreground objects
    // Main object: large circle
    cv::circle(image, cv::Point(300, 200), 80, cv::Scalar(0, 255, 0), -1);
    cv::circle(image, cv::Point(280, 180), 20, cv::Scalar(255, 255, 0), -1);
    
    // Secondary object: rectangle
    cv::rectangle(image, cv::Point(150, 150), cv::Point(220, 220), cv::Scalar(255, 0, 255), -1);
    
    // Add some texture to foreground
    for (int i = 0; i < 20; i++) {
        cv::Point pt1(250 + rand() % 100, 150 + rand() % 100);
        cv::Point pt2(pt1.x + 10, pt1.y + 10);
        cv::rectangle(image, pt1, pt2, cv::Scalar(rand() % 255, rand() % 255, rand() % 255), -1);
    }
    
    return image;
}

void demonstrateBasicGrabCut(const cv::Mat& src) {
    std::cout << "\n=== Basic GrabCut Segmentation ===" << std::endl;
    
    // Define rectangle around the object of interest
    cv::Rect rect(120, 120, 250, 160);  // x, y, width, height
    
    // Create mask and models
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Mat bgModel, fgModel;
    
    // Apply GrabCut
    cv::grabCut(src, mask, rect, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);
    
    // Create binary mask (foreground/background)
    cv::Mat mask2;
    cv::compare(mask, cv::GC_PR_FGD, mask2, cv::CMP_EQ);
    cv::Mat mask3;
    cv::compare(mask, cv::GC_FGD, mask3, cv::CMP_EQ);
    cv::bitwise_or(mask2, mask3, mask2);
    
    // Apply mask to extract foreground
    cv::Mat result;
    src.copyTo(result, mask2);
    
    // Visualization
    cv::Mat display = cv::Mat::zeros(src.rows, src.cols * 3, CV_8UC3);
    
    // Original with rectangle
    cv::Mat src_with_rect = src.clone();
    cv::rectangle(src_with_rect, rect, cv::Scalar(0, 255, 0), 3);
    
    // Mask visualization
    cv::Mat mask_vis;
    cv::cvtColor(mask2, mask_vis, cv::COLOR_GRAY2BGR);
    
    src_with_rect.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    mask_vis.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    result.copyTo(display(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original + Rectangle", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Extracted Mask", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Foreground", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Basic GrabCut", cv::WINDOW_AUTOSIZE);
    cv::imshow("Basic GrabCut", display);
    
    // Display mask values
    std::cout << "GrabCut mask values:" << std::endl;
    std::cout << "  GC_BGD (0): Definite background" << std::endl;
    std::cout << "  GC_FGD (1): Definite foreground" << std::endl;
    std::cout << "  GC_PR_BGD (2): Probable background" << std::endl;
    std::cout << "  GC_PR_FGD (3): Probable foreground" << std::endl;
    
    // Count pixels in each category
    int bgd_count = cv::countNonZero(mask == cv::GC_BGD);
    int fgd_count = cv::countNonZero(mask == cv::GC_FGD);
    int pr_bgd_count = cv::countNonZero(mask == cv::GC_PR_BGD);
    int pr_fgd_count = cv::countNonZero(mask == cv::GC_PR_FGD);
    
    std::cout << "Pixel classification:" << std::endl;
    std::cout << "  Definite background: " << bgd_count << std::endl;
    std::cout << "  Definite foreground: " << fgd_count << std::endl;
    std::cout << "  Probable background: " << pr_bgd_count << std::endl;
    std::cout << "  Probable foreground: " << pr_fgd_count << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateRefinedGrabCut(const cv::Mat& src) {
    std::cout << "\n=== Refined GrabCut with User Input ===" << std::endl;
    
    // Initial rectangle
    cv::Rect rect(120, 120, 250, 160);
    
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Mat bgModel, fgModel;
    
    // Initial GrabCut with rectangle
    cv::grabCut(src, mask, rect, bgModel, fgModel, 3, cv::GC_INIT_WITH_RECT);
    
    // Simulate user refinement (in real application, this would be interactive)
    cv::Mat refined_mask = mask.clone();
    
    // Mark some areas as definite foreground (simulate brush strokes)
    cv::circle(refined_mask, cv::Point(200, 180), 15, cv::GC_FGD, -1);
    cv::circle(refined_mask, cv::Point(300, 200), 20, cv::GC_FGD, -1);
    cv::circle(refined_mask, cv::Point(180, 190), 10, cv::GC_FGD, -1);
    
    // Mark some areas as definite background
    cv::circle(refined_mask, cv::Point(100, 100), 20, cv::GC_BGD, -1);
    cv::circle(refined_mask, cv::Point(400, 300), 20, cv::GC_BGD, -1);
    cv::circle(refined_mask, cv::Point(500, 100), 15, cv::GC_BGD, -1);
    
    // Refine with user input
    cv::grabCut(src, refined_mask, rect, bgModel, fgModel, 3, cv::GC_INIT_WITH_MASK);
    
    // Create binary masks for comparison
    cv::Mat mask_initial, mask_refined;
    cv::compare(mask, cv::GC_PR_FGD, mask_initial, cv::CMP_EQ);
    cv::Mat temp;
    cv::compare(mask, cv::GC_FGD, temp, cv::CMP_EQ);
    cv::bitwise_or(mask_initial, temp, mask_initial);
    
    cv::compare(refined_mask, cv::GC_PR_FGD, mask_refined, cv::CMP_EQ);
    cv::compare(refined_mask, cv::GC_FGD, temp, cv::CMP_EQ);
    cv::bitwise_or(mask_refined, temp, mask_refined);
    
    // Extract foregrounds
    cv::Mat result_initial, result_refined;
    src.copyTo(result_initial, mask_initial);
    src.copyTo(result_refined, mask_refined);
    
    // Visualization with user strokes
    cv::Mat src_with_strokes = src.clone();
    cv::rectangle(src_with_strokes, rect, cv::Scalar(0, 255, 0), 2);
    
    // Show foreground strokes in green
    cv::circle(src_with_strokes, cv::Point(200, 180), 15, cv::Scalar(0, 255, 0), 3);
    cv::circle(src_with_strokes, cv::Point(300, 200), 20, cv::Scalar(0, 255, 0), 3);
    cv::circle(src_with_strokes, cv::Point(180, 190), 10, cv::Scalar(0, 255, 0), 3);
    
    // Show background strokes in red
    cv::circle(src_with_strokes, cv::Point(100, 100), 20, cv::Scalar(0, 0, 255), 3);
    cv::circle(src_with_strokes, cv::Point(400, 300), 20, cv::Scalar(0, 0, 255), 3);
    cv::circle(src_with_strokes, cv::Point(500, 100), 15, cv::Scalar(0, 0, 255), 3);
    
    // Create display
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    src_with_strokes.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    result_initial.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    
    cv::Mat mask_initial_vis, mask_refined_vis;
    cv::cvtColor(mask_initial, mask_initial_vis, cv::COLOR_GRAY2BGR);
    cv::cvtColor(mask_refined, mask_refined_vis, cv::COLOR_GRAY2BGR);
    
    mask_initial_vis.copyTo(display(cv::Rect(0, src.rows, src.cols, src.rows)));
    result_refined.copyTo(display(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "User Input (G=FG, R=BG)", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Initial Result", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Initial Mask", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Refined Result", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Refined GrabCut", cv::WINDOW_AUTOSIZE);
    cv::imshow("Refined GrabCut", display);
    
    std::cout << "Refinement process:" << std::endl;
    std::cout << "  1. Initial segmentation with rectangle" << std::endl;
    std::cout << "  2. User marks additional foreground/background areas" << std::endl;
    std::cout << "  3. GrabCut refines segmentation with user constraints" << std::endl;
    std::cout << "  4. Iterative process until satisfactory result" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateIterativeGrabCut(const cv::Mat& src) {
    std::cout << "\n=== Iterative GrabCut Optimization ===" << std::endl;
    
    cv::Rect rect(120, 120, 250, 160);
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Mat bgModel, fgModel;
    
    // Show progression of iterations
    std::vector<cv::Mat> iteration_results;
    std::vector<int> iteration_counts = {1, 3, 5, 10};
    
    for (int iter_count : iteration_counts) {
        cv::Mat current_mask = cv::Mat::zeros(src.size(), CV_8UC1);
        cv::Mat current_bgModel, current_fgModel;
        
        cv::grabCut(src, current_mask, rect, current_bgModel, current_fgModel, iter_count, cv::GC_INIT_WITH_RECT);
        
        // Create binary mask
        cv::Mat binary_mask;
        cv::compare(current_mask, cv::GC_PR_FGD, binary_mask, cv::CMP_EQ);
        cv::Mat temp;
        cv::compare(current_mask, cv::GC_FGD, temp, cv::CMP_EQ);
        cv::bitwise_or(binary_mask, temp, binary_mask);
        
        // Extract foreground
        cv::Mat result;
        src.copyTo(result, binary_mask);
        iteration_results.push_back(result.clone());
    }
    
    // Create comparison display
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    for (size_t i = 0; i < iteration_results.size(); i++) {
        int row = i / 2;
        int col = i % 2;
        iteration_results[i].copyTo(display(cv::Rect(col * src.cols, row * src.rows, src.cols, src.rows)));
        
        // Add iteration count label
        std::string label = std::to_string(iteration_counts[i]) + " iterations";
        cv::putText(display, label, cv::Point(col * src.cols + 10, row * src.rows + 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
    
    cv::namedWindow("Iterative GrabCut", cv::WINDOW_AUTOSIZE);
    cv::imshow("Iterative GrabCut", display);
    
    std::cout << "Iterative optimization effects:" << std::endl;
    std::cout << "  - More iterations generally improve boundary accuracy" << std::endl;
    std::cout << "  - Convergence typically occurs within 5-10 iterations" << std::endl;
    std::cout << "  - Diminishing returns after initial iterations" << std::endl;
    std::cout << "  - Computational cost increases linearly with iterations" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateGrabCutModes(const cv::Mat& src) {
    std::cout << "\n=== GrabCut Initialization Modes ===" << std::endl;
    
    cv::Rect rect(120, 120, 250, 160);
    
    // Mode 1: GC_INIT_WITH_RECT
    cv::Mat mask1 = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Mat bgModel1, fgModel1;
    cv::grabCut(src, mask1, rect, bgModel1, fgModel1, 5, cv::GC_INIT_WITH_RECT);
    
    // Mode 2: GC_INIT_WITH_MASK
    cv::Mat mask2 = cv::Mat::zeros(src.size(), CV_8UC1);
    
    // Create initial mask manually
    mask2.setTo(cv::GC_PR_BGD);  // Everything is probable background
    cv::circle(mask2, cv::Point(200, 180), 50, cv::GC_PR_FGD, -1);  // Probable foreground
    cv::circle(mask2, cv::Point(300, 200), 40, cv::GC_FGD, -1);     // Definite foreground
    cv::circle(mask2, cv::Point(100, 100), 30, cv::GC_BGD, -1);     // Definite background
    
    cv::Mat bgModel2, fgModel2;
    cv::grabCut(src, mask2, rect, bgModel2, fgModel2, 5, cv::GC_INIT_WITH_MASK);
    
    // Mode 3: GC_EVAL (refinement)
    cv::Mat mask3 = mask1.clone();
    // Add some user corrections
    cv::circle(mask3, cv::Point(250, 250), 20, cv::GC_BGD, -1);
    cv::circle(mask3, cv::Point(200, 150), 15, cv::GC_FGD, -1);
    
    cv::Mat bgModel3 = bgModel1.clone();
    cv::Mat fgModel3 = fgModel1.clone();
    cv::grabCut(src, mask3, rect, bgModel3, fgModel3, 3, cv::GC_EVAL);
    
    // Create binary masks and results
    auto createResult = [&src](const cv::Mat& mask) -> cv::Mat {
        cv::Mat binary_mask;
        cv::compare(mask, cv::GC_PR_FGD, binary_mask, cv::CMP_EQ);
        cv::Mat temp;
        cv::compare(mask, cv::GC_FGD, temp, cv::CMP_EQ);
        cv::bitwise_or(binary_mask, temp, binary_mask);
        
        cv::Mat result;
        src.copyTo(result, binary_mask);
        return result;
    };
    
    cv::Mat result1 = createResult(mask1);
    cv::Mat result2 = createResult(mask2);
    cv::Mat result3 = createResult(mask3);
    
    // Visualization
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    // Original with rectangle
    cv::Mat src_with_rect = src.clone();
    cv::rectangle(src_with_rect, rect, cv::Scalar(0, 255, 0), 3);
    
    src_with_rect.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    result1.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    result2.copyTo(display(cv::Rect(0, src.rows, src.cols, src.rows)));
    result3.copyTo(display(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original + Rect", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "INIT_WITH_RECT", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "INIT_WITH_MASK", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "GC_EVAL (refined)", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("GrabCut Modes", cv::WINDOW_AUTOSIZE);
    cv::imshow("GrabCut Modes", display);
    
    std::cout << "GrabCut initialization modes:" << std::endl;
    std::cout << "  GC_INIT_WITH_RECT: Initialize with bounding rectangle" << std::endl;
    std::cout << "  GC_INIT_WITH_MASK: Initialize with user-provided mask" << std::endl;
    std::cout << "  GC_EVAL: Refine existing segmentation (no initialization)" << std::endl;
    std::cout << "  GC_INIT_WITH_MASK allows more precise initial constraints" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== GrabCut Segmentation ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createGrabCutTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg");
    if (!real_image.empty()) {
        std::cout << "Using loaded image for demonstrations." << std::endl;
        
        // Use real image for demonstrations
        demonstrateBasicGrabCut(real_image);
        demonstrateRefinedGrabCut(real_image);
        demonstrateIterativeGrabCut(real_image);
        demonstrateGrabCutModes(real_image);
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all GrabCut techniques
        demonstrateBasicGrabCut(test_image);
        demonstrateRefinedGrabCut(test_image);
        demonstrateIterativeGrabCut(test_image);
        demonstrateGrabCutModes(test_image);
    }
    
    std::cout << "\n✓ GrabCut Segmentation demonstration complete!" << std::endl;
    std::cout << "GrabCut provides powerful interactive foreground extraction with minimal user input." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. GrabCut combines graph cuts with Gaussian Mixture Models
 * 2. Requires minimal user input - just a bounding rectangle
 * 3. Can be refined iteratively with user brush strokes
 * 4. Works well for objects with distinct color distributions
 * 5. Three initialization modes: rectangle, mask, and refinement
 * 6. Iterative optimization improves boundary accuracy
 * 7. Mask values: BGD(0), FGD(1), PR_BGD(2), PR_FGD(3)
 * 8. Particularly effective for natural images with clear foreground/background distinction
 */

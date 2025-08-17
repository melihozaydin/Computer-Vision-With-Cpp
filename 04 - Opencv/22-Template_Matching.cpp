/**
 * 22-Template_Matching.cpp
 * 
 * Template matching for object detection.
 * 
 * Concepts covered:
 * - Normalized cross-correlation
 * - Multiple template matching
 * - Multi-scale matching
 * - Rotation-invariant matching
 * - Template selection
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

cv::Mat createTemplateMatchingTestImage() {
    cv::Mat image = cv::Mat::zeros(600, 800, CV_8UC3);
    
    // Create background pattern
    for (int y = 0; y < image.rows; y += 20) {
        for (int x = 0; x < image.cols; x += 20) {
            cv::rectangle(image, cv::Point(x, y), cv::Point(x + 10, y + 10), 
                         cv::Scalar(50, 50, 50), -1);
        }
    }
    
    // Add objects to find (these will be our templates)
    // Object 1: Red circle
    cv::circle(image, cv::Point(150, 150), 40, cv::Scalar(0, 0, 255), -1);
    cv::circle(image, cv::Point(150, 150), 25, cv::Scalar(255, 255, 255), -1);
    
    // Object 2: Green rectangle
    cv::rectangle(image, cv::Point(400, 100), cv::Point(480, 180), cv::Scalar(0, 255, 0), -1);
    cv::rectangle(image, cv::Point(420, 120), cv::Point(460, 160), cv::Scalar(255, 255, 255), -1);
    
    // Object 3: Blue triangle (using lines)
    std::vector<cv::Point> triangle = {cv::Point(600, 200), cv::Point(650, 280), cv::Point(550, 280)};
    cv::fillPoly(image, triangle, cv::Scalar(255, 0, 0));
    
    // Add some similar objects at different locations
    cv::circle(image, cv::Point(350, 350), 40, cv::Scalar(0, 0, 255), -1);
    cv::circle(image, cv::Point(350, 350), 25, cv::Scalar(255, 255, 255), -1);
    
    cv::rectangle(image, cv::Point(200, 400), cv::Point(280, 480), cv::Scalar(0, 255, 0), -1);
    cv::rectangle(image, cv::Point(220, 420), cv::Point(260, 460), cv::Scalar(255, 255, 255), -1);
    
    // Add scaled version
    cv::circle(image, cv::Point(600, 450), 25, cv::Scalar(0, 0, 255), -1);
    cv::circle(image, cv::Point(600, 450), 15, cv::Scalar(255, 255, 255), -1);
    
    return image;
}

cv::Mat extractTemplate(const cv::Mat& src, cv::Rect roi) {
    return src(roi).clone();
}

void demonstrateBasicTemplateMatching(const cv::Mat& src) {
    std::cout << "\n=== Basic Template Matching ===" << std::endl;
    
    // Extract template from the image (red circle)
    cv::Rect template_roi(110, 110, 80, 80);
    cv::Mat template_img = extractTemplate(src, template_roi);
    
    // Apply different matching methods
    std::vector<int> methods = {
        cv::TM_CCOEFF,
        cv::TM_CCOEFF_NORMED,
        cv::TM_CCORR,
        cv::TM_CCORR_NORMED,
        cv::TM_SQDIFF,
        cv::TM_SQDIFF_NORMED
    };
    
    std::vector<std::string> method_names = {
        "TM_CCOEFF",
        "TM_CCOEFF_NORMED",
        "TM_CCORR",
        "TM_CCORR_NORMED",
        "TM_SQDIFF",
        "TM_SQDIFF_NORMED"
    };
    
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 3, CV_8UC3);
    
    // Original image with template region marked
    cv::Mat src_marked = src.clone();
    cv::rectangle(src_marked, template_roi, cv::Scalar(255, 255, 0), 3);
    src_marked.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    
    // Template
    cv::Mat template_display;
    cv::resize(template_img, template_display, cv::Size(200, 200));
    template_display.copyTo(display(cv::Rect(src.cols, 0, 200, 200)));
    
    // Test one method in detail (TM_CCOEFF_NORMED)
    cv::Mat result;
    cv::matchTemplate(src, template_img, result, cv::TM_CCOEFF_NORMED);
    
    // Find locations with good match
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
    
    // Threshold for multiple detections
    cv::Mat thresh_result;
    cv::threshold(result, thresh_result, 0.8, 1.0, cv::THRESH_BINARY);
    
    // Find all good matches
    std::vector<cv::Point> locations;
    cv::Mat result_8u;
    thresh_result.convertTo(result_8u, CV_8U, 255);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(result_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Mat result_display = src.clone();
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        cv::Point center(rect.x + rect.width/2, rect.y + rect.height/2);
        locations.push_back(center);
        
        cv::rectangle(result_display, 
                     cv::Point(center.x - template_img.cols/2, center.y - template_img.rows/2),
                     cv::Point(center.x + template_img.cols/2, center.y + template_img.rows/2),
                     cv::Scalar(0, 255, 0), 3);
    }
    
    result_display.copyTo(display(cv::Rect(0, src.rows, src.cols, src.rows)));
    
    // Visualize correlation result
    cv::Mat result_norm;
    cv::normalize(result, result_norm, 0, 255, cv::NORM_MINMAX);
    result_norm.convertTo(result_norm, CV_8U);
    cv::Mat result_colored;
    cv::applyColorMap(result_norm, result_colored, cv::COLORMAP_JET);
    cv::resize(result_colored, result_colored, src.size());
    result_colored.copyTo(display(cv::Rect(src.cols, src.rows, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original + Template ROI", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Template", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Detections", cv::Point(10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Correlation Map", cv::Point(src.cols + 10, src.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Basic Template Matching", cv::WINDOW_AUTOSIZE);
    cv::imshow("Basic Template Matching", display);
    
    std::cout << "Template matching results:" << std::endl;
    std::cout << "  Template size: " << template_img.cols << "x" << template_img.rows << std::endl;
    std::cout << "  Best match score: " << max_val << std::endl;
    std::cout << "  Best match location: (" << max_loc.x << ", " << max_loc.y << ")" << std::endl;
    std::cout << "  Number of detections (threshold=0.8): " << locations.size() << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateMultipleTemplateMatching(const cv::Mat& src) {
    std::cout << "\n=== Multiple Template Matching ===" << std::endl;
    
    // Extract multiple templates
    std::vector<cv::Rect> template_rois = {
        cv::Rect(110, 110, 80, 80),  // Red circle
        cv::Rect(380, 80, 100, 100), // Green rectangle
        cv::Rect(530, 180, 120, 120) // Blue triangle
    };
    
    std::vector<cv::Mat> templates;
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),    // Red
        cv::Scalar(0, 255, 0),    // Green
        cv::Scalar(255, 0, 0)     // Blue
    };
    
    for (const auto& roi : template_rois) {
        templates.push_back(extractTemplate(src, roi));
    }
    
    cv::Mat result_display = src.clone();
    
    // Match each template
    for (size_t i = 0; i < templates.size(); i++) {
        cv::Mat result;
        cv::matchTemplate(src, templates[i], result, cv::TM_CCOEFF_NORMED);
        
        // Find multiple matches
        cv::Mat thresh_result;
        cv::threshold(result, thresh_result, 0.7, 1.0, cv::THRESH_BINARY);
        
        cv::Mat result_8u;
        thresh_result.convertTo(result_8u, CV_8U, 255);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(result_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        for (const auto& contour : contours) {
            cv::Rect rect = cv::boundingRect(contour);
            cv::Point center(rect.x + rect.width/2, rect.y + rect.height/2);
            
            cv::rectangle(result_display, 
                         cv::Point(center.x - templates[i].cols/2, center.y - templates[i].rows/2),
                         cv::Point(center.x + templates[i].cols/2, center.y + templates[i].rows/2),
                         colors[i], 3);
            
            // Add template index
            cv::putText(result_display, std::to_string(i), 
                       cv::Point(center.x - 10, center.y), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2);
        }
    }
    
    // Create display with templates
    cv::Mat display = cv::Mat::zeros(src.rows, src.cols + 300, CV_8UC3);
    result_display.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    
    // Show templates
    int y_offset = 20;
    for (size_t i = 0; i < templates.size(); i++) {
        cv::Mat template_resized;
        cv::resize(templates[i], template_resized, cv::Size(80, 80));
        
        if (y_offset + 80 < src.rows) {
            template_resized.copyTo(display(cv::Rect(src.cols + 10, y_offset, 80, 80)));
            cv::putText(display, "Template " + std::to_string(i), 
                       cv::Point(src.cols + 100, y_offset + 40), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2);
            y_offset += 100;
        }
    }
    
    cv::namedWindow("Multiple Template Matching", cv::WINDOW_AUTOSIZE);
    cv::imshow("Multiple Template Matching", display);
    
    std::cout << "Multiple template matching features:" << std::endl;
    std::cout << "  - Each template uses different color for visualization" << std::endl;
    std::cout << "  - Threshold of 0.7 used for all templates" << std::endl;
    std::cout << "  - Numbers indicate template index" << std::endl;
    std::cout << "  - Can detect multiple instances of each template" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateMultiScaleTemplateMatching(const cv::Mat& src) {
    std::cout << "\n=== Multi-Scale Template Matching ===" << std::endl;
    
    // Extract template
    cv::Rect template_roi(110, 110, 80, 80);
    cv::Mat template_img = extractTemplate(src, template_roi);
    
    cv::Mat result_display = src.clone();
    
    // Test different scales
    std::vector<double> scales = {0.5, 0.75, 1.0, 1.25, 1.5};
    std::vector<cv::Scalar> scale_colors = {
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Cyan
        cv::Scalar(255, 255, 0),    // Yellow
        cv::Scalar(255, 128, 0),    // Orange
        cv::Scalar(128, 0, 255)     // Purple
    };
    
    struct Detection {
        cv::Point location;
        double scale;
        double score;
        int scale_index;
    };
    
    std::vector<Detection> all_detections;
    
    for (size_t scale_idx = 0; scale_idx < scales.size(); scale_idx++) {
        double scale = scales[scale_idx];
        
        // Scale template
        cv::Mat scaled_template;
        cv::resize(template_img, scaled_template, 
                  cv::Size(template_img.cols * scale, template_img.rows * scale));
        
        // Skip if template is too large
        if (scaled_template.cols >= src.cols || scaled_template.rows >= src.rows) {
            continue;
        }
        
        // Apply template matching
        cv::Mat result;
        cv::matchTemplate(src, scaled_template, result, cv::TM_CCOEFF_NORMED);
        
        // Find best matches
        double min_val, max_val;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
        
        // If good match, add to detections
        if (max_val > 0.7) {
            Detection det;
            det.location = cv::Point(max_loc.x + scaled_template.cols/2, 
                                   max_loc.y + scaled_template.rows/2);
            det.scale = scale;
            det.score = max_val;
            det.scale_index = scale_idx;
            all_detections.push_back(det);
        }
        
        // Find multiple matches above threshold
        cv::Mat thresh_result;
        cv::threshold(result, thresh_result, 0.7, 1.0, cv::THRESH_BINARY);
        
        cv::Mat result_8u;
        thresh_result.convertTo(result_8u, CV_8U, 255);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(result_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        for (const auto& contour : contours) {
            cv::Rect rect = cv::boundingRect(contour);
            cv::Point center(rect.x + rect.width/2, rect.y + rect.height/2);
            
            cv::rectangle(result_display, 
                         cv::Point(center.x - scaled_template.cols/2, center.y - scaled_template.rows/2),
                         cv::Point(center.x + scaled_template.cols/2, center.y + scaled_template.rows/2),
                         scale_colors[scale_idx], 2);
            
            // Add scale info
            std::string scale_text = cv::format("%.2f", scale);
            cv::putText(result_display, scale_text, 
                       cv::Point(center.x - 15, center.y - scaled_template.rows/2 - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, scale_colors[scale_idx], 2);
        }
    }
    
    // Non-maximum suppression to remove overlapping detections
    std::sort(all_detections.begin(), all_detections.end(), 
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    
    cv::Mat nms_display = src.clone();
    std::vector<Detection> final_detections;
    
    for (const auto& det : all_detections) {
        bool is_suppressed = false;
        for (const auto& final_det : final_detections) {
            double distance = cv::norm(det.location - final_det.location);
            if (distance < 50) {  // Suppression threshold
                is_suppressed = true;
                break;
            }
        }
        
        if (!is_suppressed) {
            final_detections.push_back(det);
            
            int template_size = template_img.cols * det.scale;
            cv::rectangle(nms_display, 
                         cv::Point(det.location.x - template_size/2, det.location.y - template_size/2),
                         cv::Point(det.location.x + template_size/2, det.location.y + template_size/2),
                         cv::Scalar(0, 255, 0), 3);
            
            std::string info = cv::format("%.2f(%.2f)", det.scale, det.score);
            cv::putText(nms_display, info, 
                       cv::Point(det.location.x - 25, det.location.y - template_size/2 - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }
    }
    
    // Create display
    cv::Mat display = cv::Mat::zeros(src.rows, src.cols * 2, CV_8UC3);
    result_display.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    nms_display.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    
    cv::putText(display, "All Scales", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "After NMS", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Multi-Scale Template Matching", cv::WINDOW_AUTOSIZE);
    cv::imshow("Multi-Scale Template Matching", display);
    
    std::cout << "Multi-scale template matching:" << std::endl;
    std::cout << "  Scales tested: ";
    for (double scale : scales) std::cout << scale << " ";
    std::cout << std::endl;
    std::cout << "  Total detections: " << all_detections.size() << std::endl;
    std::cout << "  After NMS: " << final_detections.size() << std::endl;
    std::cout << "  Best detections (scale, score):" << std::endl;
    for (const auto& det : final_detections) {
        std::cout << "    Scale: " << det.scale << ", Score: " << det.score << std::endl;
    }
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateTemplateMatchingMethods(const cv::Mat& src) {
    std::cout << "\n=== Template Matching Methods Comparison ===" << std::endl;
    
    cv::Rect template_roi(110, 110, 80, 80);
    cv::Mat template_img = extractTemplate(src, template_roi);
    
    std::vector<int> methods = {
        cv::TM_CCOEFF_NORMED,
        cv::TM_CCORR_NORMED,
        cv::TM_SQDIFF_NORMED
    };
    
    std::vector<std::string> method_names = {
        "TM_CCOEFF_NORMED",
        "TM_CCORR_NORMED", 
        "TM_SQDIFF_NORMED"
    };
    
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    // Original
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    
    for (size_t i = 0; i < methods.size() && i < 3; i++) {
        cv::Mat result;
        cv::matchTemplate(src, template_img, result, methods[i]);
        
        // Normalize and convert to color map
        cv::Mat result_norm;
        cv::normalize(result, result_norm, 0, 255, cv::NORM_MINMAX);
        result_norm.convertTo(result_norm, CV_8U);
        
        cv::Mat result_colored;
        cv::applyColorMap(result_norm, result_colored, cv::COLORMAP_JET);
        cv::resize(result_colored, result_colored, src.size());
        
        // Find best match
        double min_val, max_val;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
        
        cv::Point match_loc = (methods[i] == cv::TM_SQDIFF_NORMED) ? min_loc : max_loc;
        double match_val = (methods[i] == cv::TM_SQDIFF_NORMED) ? min_val : max_val;
        
        // Mark best match
        cv::rectangle(result_colored, match_loc, 
                     cv::Point(match_loc.x + template_img.cols, match_loc.y + template_img.rows),
                     cv::Scalar(255, 255, 255), 3);
        
        // Place in display grid
        int row = (i + 1) / 2;
        int col = (i + 1) % 2;
        result_colored.copyTo(display(cv::Rect(col * src.cols, row * src.rows, src.cols, src.rows)));
        
        // Add method name and score
        cv::putText(display, method_names[i], 
                   cv::Point(col * src.cols + 10, row * src.rows + 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        cv::putText(display, cv::format("Score: %.3f", match_val), 
                   cv::Point(col * src.cols + 10, row * src.rows + 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }
    
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Template Matching Methods", cv::WINDOW_AUTOSIZE);
    cv::imshow("Template Matching Methods", display);
    
    std::cout << "Template matching methods comparison:" << std::endl;
    std::cout << "  TM_CCOEFF_NORMED: Best for normalized correlation" << std::endl;
    std::cout << "  TM_CCORR_NORMED: Simple correlation (bright templates)" << std::endl;
    std::cout << "  TM_SQDIFF_NORMED: Squared difference (lower is better)" << std::endl;
    std::cout << "  Normalized methods are generally preferred" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Template Matching ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createTemplateMatchingTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg");
    if (!real_image.empty()) {
        std::cout << "Using loaded image for some demonstrations." << std::endl;
        
        // Use test image for comprehensive demos, real image for basic
        demonstrateBasicTemplateMatching(test_image);
        demonstrateMultipleTemplateMatching(test_image);
        demonstrateMultiScaleTemplateMatching(test_image);
        demonstrateTemplateMatchingMethods(test_image);
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all template matching techniques
        demonstrateBasicTemplateMatching(test_image);
        demonstrateMultipleTemplateMatching(test_image);
        demonstrateMultiScaleTemplateMatching(test_image);
        demonstrateTemplateMatchingMethods(test_image);
    }
    
    std::cout << "\n✓ Template Matching demonstration complete!" << std::endl;
    std::cout << "Template matching is effective for finding known objects in images." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Template matching finds regions in an image similar to a template
 * 2. Different methods (TM_CCOEFF_NORMED, TM_SQDIFF_NORMED) have different characteristics
 * 3. Normalized methods are generally more robust to illumination changes
 * 4. Multi-scale matching handles size variations in objects
 * 5. Non-maximum suppression helps remove duplicate detections
 * 6. Template matching is sensitive to rotation and deformation
 * 7. Works best with distinctive, high-contrast templates
 * 8. Computational complexity increases with template and image size
 */

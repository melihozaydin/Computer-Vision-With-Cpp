/**
 * 11-Contours.cpp
 * 
 * Finding and analyzing contours in binary images.
 * 
 * Concepts covered:
 * - Contour detection
 * - Contour approximation
 * - Contour properties
 * - Contour hierarchy
 * - Shape matching
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

cv::Mat createContourTestImage() {
    cv::Mat image = cv::Mat::zeros(500, 700, CV_8UC1);
    
    // Various shapes for contour analysis
    cv::rectangle(image, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255), -1);
    cv::circle(image, cv::Point(250, 100), 50, cv::Scalar(255), -1);
    
    // Triangle
    std::vector<cv::Point> triangle = {cv::Point(400, 50), cv::Point(350, 150), cv::Point(450, 150)};
    cv::fillPoly(image, triangle, cv::Scalar(255));
    
    // Complex polygon
    std::vector<cv::Point> polygon = {
        cv::Point(550, 50), cv::Point(650, 80), cv::Point(630, 130),
        cv::Point(600, 140), cv::Point(570, 120), cv::Point(530, 100)
    };
    cv::fillPoly(image, polygon, cv::Scalar(255));
    
    // Nested shapes (for hierarchy testing)
    cv::rectangle(image, cv::Point(100, 250), cv::Point(300, 400), cv::Scalar(255), -1);
    cv::rectangle(image, cv::Point(130, 280), cv::Point(180, 330), cv::Scalar(0), -1);  // Hole
    cv::rectangle(image, cv::Point(220, 280), cv::Point(270, 330), cv::Scalar(0), -1);  // Another hole
    cv::circle(image, cv::Point(200, 350), 15, cv::Scalar(0), -1);  // Circular hole
    
    // Concentric circles
    cv::circle(image, cv::Point(450, 320), 60, cv::Scalar(255), -1);
    cv::circle(image, cv::Point(450, 320), 40, cv::Scalar(0), -1);
    cv::circle(image, cv::Point(450, 320), 20, cv::Scalar(255), -1);
    
    // Ellipse
    cv::ellipse(image, cv::Point(600, 350), cv::Size(60, 40), 30, 0, 360, cv::Scalar(255), -1);
    
    return image;
}

void demonstrateContourDetection(const cv::Mat& binary) {
    std::cout << "\n=== Contour Detection ===" << std::endl;
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    // Find contours with different retrieval modes
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::cout << "External contours found: " << contours.size() << std::endl;
    
    std::vector<std::vector<cv::Point>> contours_all;
    std::vector<cv::Vec4i> hierarchy_all;
    cv::findContours(binary, contours_all, hierarchy_all, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::cout << "All contours (with hierarchy): " << contours_all.size() << std::endl;
    
    // Visualize external contours
    cv::Mat result_external = cv::Mat::zeros(binary.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
        cv::drawContours(result_external, contours, static_cast<int>(i), color, 2);
    }
    
    // Visualize all contours with hierarchy colors
    cv::Mat result_all = cv::Mat::zeros(binary.size(), CV_8UC3);
    for (size_t i = 0; i < contours_all.size(); i++) {
        // Color based on hierarchy level
        int level = 0;
        int idx = static_cast<int>(i);
        while (hierarchy_all[idx][3] != -1) {  // Count parent levels
            idx = hierarchy_all[idx][3];
            level++;
        }
        
        cv::Scalar color;
        switch (level) {
            case 0: color = cv::Scalar(0, 255, 0); break;    // Green for outermost
            case 1: color = cv::Scalar(0, 0, 255); break;    // Red for first level holes
            case 2: color = cv::Scalar(255, 0, 0); break;    // Blue for inner contours
            default: color = cv::Scalar(255, 255, 0); break; // Cyan for deeper levels
        }
        
        cv::drawContours(result_all, contours_all, static_cast<int>(i), color, 2);
        
        // Add contour number
        cv::Moments m = cv::moments(contours_all[i]);
        if (m.m00 != 0) {
            cv::Point center(m.m10 / m.m00, m.m01 / m.m00);
            cv::putText(result_all, std::to_string(i), center,
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    cv::namedWindow("Original Binary", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("External Contours", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("All Contours (Hierarchy)", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original Binary", binary);
    cv::imshow("External Contours", result_external);
    cv::imshow("All Contours (Hierarchy)", result_all);
    
    // Print hierarchy information
    std::cout << "\nHierarchy information (next, previous, first_child, parent):" << std::endl;
    for (size_t i = 0; i < std::min(hierarchy_all.size(), size_t(10)); i++) {
        std::cout << "Contour " << i << ": [" 
                  << hierarchy_all[i][0] << ", " << hierarchy_all[i][1] << ", "
                  << hierarchy_all[i][2] << ", " << hierarchy_all[i][3] << "]" << std::endl;
    }
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateContourApproximation(const cv::Mat& binary) {
    std::cout << "\n=== Contour Approximation ===" << std::endl;
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Mat result = cv::Mat::zeros(binary.size(), CV_8UC3);
    cv::Mat result_approx = cv::Mat::zeros(binary.size(), CV_8UC3);
    
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() < 5) continue;  // Skip small contours
        
        // Original contour
        cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
        
        // Douglas-Peucker approximation with different epsilon values
        std::vector<cv::Point> approx_loose, approx_tight;
        double epsilon_loose = 0.02 * cv::arcLength(contours[i], true);
        double epsilon_tight = 0.005 * cv::arcLength(contours[i], true);
        
        cv::approxPolyDP(contours[i], approx_loose, epsilon_loose, true);
        cv::approxPolyDP(contours[i], approx_tight, epsilon_tight, true);
        
        // Draw approximations
        cv::polylines(result_approx, approx_loose, true, cv::Scalar(0, 0, 255), 2);  // Red for loose
        cv::polylines(result_approx, approx_tight, true, cv::Scalar(255, 0, 0), 1);   // Blue for tight
        
        // Mark vertices of loose approximation
        for (const auto& pt : approx_loose) {
            cv::circle(result_approx, pt, 3, cv::Scalar(0, 255, 255), -1);
        }
        
        // Print approximation info
        std::cout << "Contour " << i << ": Original=" << contours[i].size() 
                  << " points, Loose=" << approx_loose.size() 
                  << " points, Tight=" << approx_tight.size() << " points" << std::endl;
        
        // Convex hull
        std::vector<cv::Point> hull;
        cv::convexHull(contours[i], hull);
        cv::polylines(result, hull, true, cv::Scalar(255, 255, 0), 1);  // Cyan for hull
        
        // Check if contour is convex
        bool is_convex = cv::isContourConvex(approx_loose);
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 != 0) {
            cv::Point center(m.m10 / m.m00, m.m01 / m.m00);
            std::string text = is_convex ? "Convex" : "Concave";
            cv::putText(result, text, cv::Point(center.x - 30, center.y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    cv::namedWindow("Original + Convex Hull", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Approximations", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original + Convex Hull", result);
    cv::imshow("Approximations", result_approx);
    
    std::cout << "Green: Original contours, Cyan: Convex hulls" << std::endl;
    std::cout << "Red: Loose approximation, Blue: Tight approximation" << std::endl;
    std::cout << "Yellow circles: Approximation vertices" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateContourProperties(const cv::Mat& binary) {
    std::cout << "\n=== Contour Properties ===" << std::endl;
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Mat result;
    cv::cvtColor(binary, result, cv::COLOR_GRAY2BGR);
    
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() < 5) continue;
        
        // Basic properties
        double area = cv::contourArea(contours[i]);
        double perimeter = cv::arcLength(contours[i], true);
        
        // Bounding rectangle
        cv::Rect bounding_rect = cv::boundingRect(contours[i]);
        cv::rectangle(result, bounding_rect, cv::Scalar(0, 255, 0), 1);
        
        // Minimum enclosing circle
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contours[i], center, radius);
        cv::circle(result, center, static_cast<int>(radius), cv::Scalar(255, 0, 0), 1);
        
        // Fitted ellipse (if enough points)
        if (contours[i].size() >= 5) {
            cv::RotatedRect fitted_ellipse = cv::fitEllipse(contours[i]);
            cv::ellipse(result, fitted_ellipse, cv::Scalar(0, 0, 255), 1);
        }
        
        // Calculate derived properties
        double extent = area / (bounding_rect.width * bounding_rect.height);
        // double solidity = area / cv::contourArea(contours[i]);  // This would need convex hull
        double aspect_ratio = static_cast<double>(bounding_rect.width) / bounding_rect.height;
        
        // Convex hull for solidity calculation
        std::vector<cv::Point> hull;
        cv::convexHull(contours[i], hull);
        double hull_area = cv::contourArea(hull);
        double actual_solidity = hull_area > 0 ? area / hull_area : 0;
        
        // Shape factor (circularity)
        double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
        
        // Draw contour
        cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(255, 255, 0), 2);
        
        // Add text with properties
        cv::Point text_pos(bounding_rect.x, bounding_rect.y - 10);
        std::string text = "A:" + std::to_string(static_cast<int>(area)) +
                          " C:" + std::to_string(circularity).substr(0, 4);
        cv::putText(result, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        
        // Print detailed properties
        std::cout << "\nContour " << i << " properties:" << std::endl;
        std::cout << "  Area: " << area << std::endl;
        std::cout << "  Perimeter: " << perimeter << std::endl;
        std::cout << "  Extent: " << extent << std::endl;
        std::cout << "  Solidity: " << actual_solidity << std::endl;
        std::cout << "  Aspect Ratio: " << aspect_ratio << std::endl;
        std::cout << "  Circularity: " << circularity << std::endl;
        std::cout << "  Bounding Rect: " << bounding_rect << std::endl;
        std::cout << "  Enclosing Circle: center(" << center.x << "," << center.y 
                  << ") radius=" << radius << std::endl;
    }
    
    cv::namedWindow("Contour Properties", cv::WINDOW_AUTOSIZE);
    cv::imshow("Contour Properties", result);
    
    std::cout << "\nVisualization:" << std::endl;
    std::cout << "Green: Bounding rectangles" << std::endl;
    std::cout << "Blue: Minimum enclosing circles" << std::endl;
    std::cout << "Red: Fitted ellipses" << std::endl;
    std::cout << "Yellow: Contours" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateShapeMatching(const cv::Mat& binary) {
    std::cout << "\n=== Shape Matching ===" << std::endl;
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.size() < 2) {
        std::cout << "Need at least 2 contours for shape matching." << std::endl;
        return;
    }
    
    cv::Mat result;
    cv::cvtColor(binary, result, cv::COLOR_GRAY2BGR);
    
    // Use first contour as reference
    int reference_idx = 0;
    double max_area = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            reference_idx = static_cast<int>(i);
        }
    }
    
    std::cout << "Using contour " << reference_idx << " as reference (area: " << max_area << ")" << std::endl;
    
    // Draw reference contour in special color
    cv::drawContours(result, contours, reference_idx, cv::Scalar(0, 255, 0), 3);
    
    // Compare all other contours to reference
    std::vector<std::pair<int, double>> matches;
    
    for (size_t i = 0; i < contours.size(); i++) {
        if (static_cast<int>(i) == reference_idx) continue;
        if (contours[i].size() < 5) continue;
        
        // Hu moments for shape matching
        cv::Moments m1 = cv::moments(contours[reference_idx]);
        cv::Moments m2 = cv::moments(contours[i]);
        
        double hu1[7], hu2[7];
        cv::HuMoments(m1, hu1);
        cv::HuMoments(m2, hu2);
        
        // Calculate similarity using Hu moments
        double similarity = 0;
        for (int j = 0; j < 7; j++) {
            if (hu1[j] != 0 && hu2[j] != 0) {
                similarity += std::abs(std::log(std::abs(hu1[j])) - std::log(std::abs(hu2[j])));
            }
        }
        
        matches.push_back({static_cast<int>(i), similarity});
        
        // Match shapes using OpenCV's matchShapes
        double match_value = cv::matchShapes(contours[reference_idx], contours[i], cv::CONTOURS_MATCH_I1, 0);
        
        // Color based on similarity (lower is better)
        cv::Scalar color;
        if (match_value < 0.1) color = cv::Scalar(0, 0, 255);      // Red: very similar
        else if (match_value < 0.5) color = cv::Scalar(0, 165, 255); // Orange: similar
        else if (match_value < 1.0) color = cv::Scalar(0, 255, 255);  // Yellow: somewhat similar
        else color = cv::Scalar(255, 0, 0);                           // Blue: not similar
        
        cv::drawContours(result, contours, static_cast<int>(i), color, 2);
        
        // Add similarity text
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 != 0) {
            cv::Point center(m.m10 / m.m00, m.m01 / m.m00);
            std::string text = std::to_string(match_value).substr(0, 4);
            cv::putText(result, text, center, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
        
        std::cout << "Contour " << i << " similarity: " << match_value 
                  << " (Hu moments: " << similarity << ")" << std::endl;
    }
    
    // Sort matches by similarity
    std::sort(matches.begin(), matches.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    cv::namedWindow("Shape Matching", cv::WINDOW_AUTOSIZE);
    cv::imshow("Shape Matching", result);
    
    std::cout << "\nShape matching results (lower values = more similar):" << std::endl;
    std::cout << "Green: Reference contour" << std::endl;
    std::cout << "Red: Very similar (< 0.1)" << std::endl;
    std::cout << "Orange: Similar (0.1-0.5)" << std::endl;
    std::cout << "Yellow: Somewhat similar (0.5-1.0)" << std::endl;
    std::cout << "Blue: Not similar (> 1.0)" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateContourFeatures(const cv::Mat& binary) {
    std::cout << "\n=== Advanced Contour Features ===" << std::endl;
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Mat result;
    cv::cvtColor(binary, result, cv::COLOR_GRAY2BGR);
    
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() < 5) continue;
        
        // Convexity defects
        std::vector<cv::Point> hull_points;
        std::vector<int> hull_indices;
        cv::convexHull(contours[i], hull_points, false);
        cv::convexHull(contours[i], hull_indices, false);
        
        if (hull_indices.size() > 3) {
            std::vector<cv::Vec4i> defects;
            cv::convexityDefects(contours[i], hull_indices, defects);
            
            // Draw convex hull
            cv::polylines(result, hull_points, true, cv::Scalar(0, 255, 0), 1);
            
            // Draw defects
            for (const auto& defect : defects) {
                if (defect[3] > 5000) {  // Filter small defects (depth > 50 pixels)
                    cv::Point start = contours[i][defect[0]];
                    cv::Point end = contours[i][defect[1]];
                    cv::Point defect_point = contours[i][defect[2]];
                    
                    cv::line(result, start, defect_point, cv::Scalar(0, 0, 255), 2);
                    cv::line(result, end, defect_point, cv::Scalar(0, 0, 255), 2);
                    cv::circle(result, defect_point, 3, cv::Scalar(255, 0, 255), -1);
                }
            }
            
            std::cout << "Contour " << i << ": " << defects.size() << " convexity defects" << std::endl;
        }
        
        // Find extreme points
        std::vector<cv::Point> extreme_points(4);
        
        // Leftmost, rightmost, topmost, bottommost points
        auto leftmost = std::min_element(contours[i].begin(), contours[i].end(),
                                        [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
        auto rightmost = std::max_element(contours[i].begin(), contours[i].end(),
                                         [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
        auto topmost = std::min_element(contours[i].begin(), contours[i].end(),
                                       [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
        auto bottommost = std::max_element(contours[i].begin(), contours[i].end(),
                                          [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
        
        // Mark extreme points
        cv::circle(result, *leftmost, 5, cv::Scalar(255, 255, 0), -1);    // Cyan
        cv::circle(result, *rightmost, 5, cv::Scalar(255, 255, 0), -1);   // Cyan
        cv::circle(result, *topmost, 5, cv::Scalar(255, 255, 0), -1);     // Cyan
        cv::circle(result, *bottommost, 5, cv::Scalar(255, 255, 0), -1);  // Cyan
        
        // Draw contour
        cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(255, 255, 255), 1);
    }
    
    cv::namedWindow("Advanced Features", cv::WINDOW_AUTOSIZE);
    cv::imshow("Advanced Features", result);
    
    std::cout << "\nVisualization:" << std::endl;
    std::cout << "White: Contours" << std::endl;
    std::cout << "Green: Convex hulls" << std::endl;
    std::cout << "Red lines: Convexity defects" << std::endl;
    std::cout << "Magenta circles: Defect points" << std::endl;
    std::cout << "Cyan circles: Extreme points" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Contour Analysis ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createContourTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg", cv::IMREAD_GRAYSCALE);
    if (!real_image.empty()) {
        cv::Mat binary_real;
        cv::threshold(real_image, binary_real, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        std::cout << "Using loaded image for some demonstrations." << std::endl;
        
        // Use real image for some demonstrations
        demonstrateContourDetection(test_image);      // Use synthetic for clear hierarchy
        demonstrateContourApproximation(binary_real); // Use real for approximation
        demonstrateContourProperties(test_image);     // Use synthetic for clear properties
        demonstrateShapeMatching(test_image);         // Use synthetic for matching
        demonstrateContourFeatures(binary_real);     // Use real for features
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all contour operations
        demonstrateContourDetection(test_image);
        demonstrateContourApproximation(test_image);
        demonstrateContourProperties(test_image);
        demonstrateShapeMatching(test_image);
        demonstrateContourFeatures(test_image);
    }
    
    std::cout << "\n✓ Contour Analysis demonstration complete!" << std::endl;
    std::cout << "Contours are fundamental for shape analysis and object recognition." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. findContours() extracts object boundaries from binary images
 * 2. Different retrieval modes handle nested contours differently
 * 3. Hierarchy information describes parent-child relationships
 * 4. approxPolyDP() simplifies contours while preserving shape
 * 5. Contour properties enable shape classification and analysis
 * 6. Hu moments provide rotation-invariant shape descriptors
 * 7. Convexity defects indicate shape concavities and fingers
 * 8. Extreme points help with orientation and feature detection
 */

/**
 * 35-Defect_Detection.cpp
 * 
 * Industrial vision and quality inspection.
 * 
 * Concepts covered:
 * - Blob detection
 * - Surface inspection
 * - Dimensional measurement
 * - Pattern recognition
 * - Automated quality control
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <random>

// Simulated product structures
struct Product {
    cv::Rect bounds;
    std::vector<cv::Point> holes;
    std::vector<cv::Rect> components;
    bool has_defects;
};

cv::Mat createPCBTestImage() {
    cv::Mat pcb = cv::Mat::zeros(500, 700, CV_8UC3);
    
    // PCB substrate (green)
    pcb.setTo(cv::Scalar(0, 100, 0));
    
    // Copper traces (golden)
    std::vector<std::vector<cv::Point>> traces;
    
    // Horizontal traces
    for (int y = 50; y < 450; y += 40) {
        std::vector<cv::Point> trace;
        for (int x = 50; x < 650; x += 20) {
            trace.push_back(cv::Point(x, y));
            trace.push_back(cv::Point(x + 15, y));
            trace.push_back(cv::Point(x + 15, y + 3));
            trace.push_back(cv::Point(x, y + 3));
        }
        traces.push_back(trace);
    }
    
    // Vertical traces
    for (int x = 100; x < 600; x += 80) {
        std::vector<cv::Point> trace;
        for (int y = 50; y < 450; y += 20) {
            trace.push_back(cv::Point(x, y));
            trace.push_back(cv::Point(x + 3, y));
            trace.push_back(cv::Point(x + 3, y + 15));
            trace.push_back(cv::Point(x, y + 15));
        }
        traces.push_back(trace);
    }
    
    // Draw traces
    for (const auto& trace : traces) {
        cv::fillPoly(pcb, std::vector<std::vector<cv::Point>>{trace}, cv::Scalar(0, 215, 255));
    }
    
    // Add components (ICs, resistors, capacitors)
    
    // ICs (black rectangles)
    std::vector<cv::Rect> ics = {
        cv::Rect(150, 120, 60, 40),
        cv::Rect(300, 200, 80, 50),
        cv::Rect(450, 150, 70, 45),
        cv::Rect(200, 300, 90, 60)
    };
    
    for (const auto& ic : ics) {
        cv::rectangle(pcb, ic, cv::Scalar(20, 20, 20), -1);
        
        // IC pins
        for (int i = 0; i < 8; i++) {
            cv::Point pin1(ic.x - 5, ic.y + 5 + i * 5);
            cv::Point pin2(ic.x + ic.width + 5, ic.y + 5 + i * 5);
            cv::rectangle(pcb, cv::Rect(pin1.x, pin1.y, 8, 3), cv::Scalar(200, 200, 200), -1);
            cv::rectangle(pcb, cv::Rect(pin2.x - 3, pin2.y, 8, 3), cv::Scalar(200, 200, 200), -1);
        }
    }
    
    // Resistors (small brown rectangles)
    std::vector<cv::Point> resistor_positions = {
        cv::Point(350, 100), cv::Point(400, 280), cv::Point(150, 250),
        cv::Point(500, 300), cv::Point(250, 150), cv::Point(380, 350)
    };
    
    for (const auto& pos : resistor_positions) {
        cv::rectangle(pcb, cv::Rect(pos.x, pos.y, 20, 8), cv::Scalar(0, 69, 139), -1);
        // Color bands
        cv::line(pcb, cv::Point(pos.x + 3, pos.y), cv::Point(pos.x + 3, pos.y + 8), cv::Scalar(0, 0, 255), 2);
        cv::line(pcb, cv::Point(pos.x + 7, pos.y), cv::Point(pos.x + 7, pos.y + 8), cv::Scalar(0, 255, 0), 2);
        cv::line(pcb, cv::Point(pos.x + 11, pos.y), cv::Point(pos.x + 11, pos.y + 8), cv::Scalar(255, 0, 0), 2);
    }
    
    // Capacitors (cylindrical)
    std::vector<cv::Point> cap_positions = {
        cv::Point(120, 350), cv::Point(320, 120), cv::Point(480, 250), cv::Point(180, 180)
    };
    
    for (const auto& pos : cap_positions) {
        cv::circle(pcb, pos, 12, cv::Scalar(0, 0, 139), -1);
        cv::putText(pcb, "+", cv::Point(pos.x - 3, pos.y + 3), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
    }
    
    // Drill holes
    std::vector<cv::Point> holes = {
        cv::Point(80, 80), cv::Point(620, 80), cv::Point(80, 420), cv::Point(620, 420),
        cv::Point(350, 50), cv::Point(350, 450)
    };
    
    for (const auto& hole : holes) {
        cv::circle(pcb, hole, 8, cv::Scalar(0, 0, 0), -1);
    }
    
    return pcb;
}

cv::Mat addDefectsToPCB(const cv::Mat& clean_pcb) {
    cv::Mat defective_pcb = clean_pcb.clone();
    
    // Missing component
    cv::rectangle(defective_pcb, cv::Rect(200, 300, 90, 60), cv::Scalar(0, 100, 0), -1);
    
    // Scratches on traces
    cv::line(defective_pcb, cv::Point(150, 90), cv::Point(250, 110), cv::Scalar(0, 100, 0), 8);
    cv::line(defective_pcb, cv::Point(400, 200), cv::Point(450, 220), cv::Scalar(0, 100, 0), 6);
    
    // Solder bridges (short circuits)
    cv::line(defective_pcb, cv::Point(160, 120), cv::Point(160, 160), cv::Scalar(200, 200, 200), 15);
    cv::line(defective_pcb, cv::Point(470, 150), cv::Point(470, 195), cv::Scalar(200, 200, 200), 12);
    
    // Component misalignment
    cv::rectangle(defective_pcb, cv::Rect(450, 150, 70, 45), cv::Scalar(0, 100, 0), -1);  // Remove original
    cv::rectangle(defective_pcb, cv::Rect(455, 155, 70, 45), cv::Scalar(20, 20, 20), -1);  // Misaligned
    
    // Corrosion spots
    cv::circle(defective_pcb, cv::Point(300, 350), 15, cv::Scalar(0, 50, 100), -1);
    cv::circle(defective_pcb, cv::Point(550, 100), 12, cv::Scalar(0, 50, 100), -1);
    
    // Broken trace
    cv::rectangle(defective_pcb, cv::Rect(480, 289, 20, 6), cv::Scalar(0, 100, 0), -1);
    
    return defective_pcb;
}

void demonstrateBlobDetection(const cv::Mat& image) {
    std::cout << "\n=== Blob Detection for Defects ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // Setup blob detector
    cv::SimpleBlobDetector::Params params;
    
    // Filter by area
    params.filterByArea = true;
    params.minArea = 50;
    params.maxArea = 5000;
    
    // Filter by circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.3;
    
    // Filter by convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.5;
    
    // Filter by inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.3;
    
    // Create detector
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    
    // Detect blobs
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(gray, keypoints);
    
    // Draw detected blobs
    cv::Mat blob_image;
    cv::drawKeypoints(image, keypoints, blob_image, cv::Scalar(0, 0, 255), 
                     cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    // Add detection info
    for (size_t i = 0; i < keypoints.size(); i++) {
        cv::Point2f pt = keypoints[i].pt;
        float size = keypoints[i].size;
        
        cv::putText(blob_image, std::to_string(i), 
                   cv::Point(static_cast<int>(pt.x) - 10, static_cast<int>(pt.y) - 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        
        std::cout << "Blob " << i << ": Position(" << pt.x << "," << pt.y 
                  << ") Size=" << size << std::endl;
    }
    
    cv::namedWindow("Blob Detection", cv::WINDOW_AUTOSIZE);
    cv::imshow("Blob Detection", blob_image);
    
    std::cout << "Blob detection parameters:" << std::endl;
    std::cout << "  - Area filter: " << params.minArea << " - " << params.maxArea << " pixels" << std::endl;
    std::cout << "  - Circularity: " << params.minCircularity << " - 1.0" << std::endl;
    std::cout << "  - Convexity: " << params.minConvexity << " - 1.0" << std::endl;
    std::cout << "  - Total blobs detected: " << keypoints.size() << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateSurfaceInspection(const cv::Mat& clean_image, const cv::Mat& defective_image) {
    std::cout << "\n=== Surface Inspection by Comparison ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat clean_gray, defective_gray;
    cv::cvtColor(clean_image, clean_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(defective_image, defective_gray, cv::COLOR_BGR2GRAY);
    
    // Align images (assume they're already aligned for this demo)
    
    // Compute difference
    cv::Mat diff;
    cv::absdiff(clean_gray, defective_gray, diff);
    
    // Threshold to highlight significant differences
    cv::Mat thresh_diff;
    cv::threshold(diff, thresh_diff, 30, 255, cv::THRESH_BINARY);
    
    // Morphological operations to clean up noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat cleaned_diff;
    cv::morphologyEx(thresh_diff, cleaned_diff, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(cleaned_diff, cleaned_diff, cv::MORPH_OPEN, kernel);
    
    // Find contours of defects
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cleaned_diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Filter contours by area
    std::vector<std::vector<cv::Point>> defect_contours;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > 100) {  // Filter small noise
            defect_contours.push_back(contour);
        }
    }
    
    // Create result visualization
    cv::Mat result_image = defective_image.clone();
    
    // Draw defect regions
    for (size_t i = 0; i < defect_contours.size(); i++) {
        // Draw contour
        cv::drawContours(result_image, defect_contours, static_cast<int>(i), cv::Scalar(0, 0, 255), 3);
        
        // Get bounding box
        cv::Rect bbox = cv::boundingRect(defect_contours[i]);
        cv::rectangle(result_image, bbox, cv::Scalar(255, 0, 0), 2);
        
        // Add defect label
        cv::putText(result_image, "DEFECT " + std::to_string(i + 1), 
                   cv::Point(bbox.x, bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        
        // Calculate defect area
        double defect_area = cv::contourArea(defect_contours[i]);
        std::cout << "Defect " << (i + 1) << ": Area = " << defect_area << " pixels" << std::endl;
    }
    
    // Create comparison display
    cv::Mat top_row, bottom_row, display;
    cv::hconcat(clean_image, defective_image, top_row);
    
    // Convert difference image to color for display
    cv::Mat diff_colored;
    cv::cvtColor(cleaned_diff, diff_colored, cv::COLOR_GRAY2BGR);
    cv::hconcat(diff_colored, result_image, bottom_row);
    cv::vconcat(top_row, bottom_row, display);
    
    // Add labels
    cv::putText(display, "Reference (Clean)", cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Test Sample", cv::Point(clean_image.cols + 10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Difference Map", cv::Point(10, clean_image.rows + 60), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Defects Found", cv::Point(clean_image.cols + 10, clean_image.rows + 60), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::namedWindow("Surface Inspection", cv::WINDOW_AUTOSIZE);
    cv::imshow("Surface Inspection", display);
    
    std::cout << "Surface inspection results:" << std::endl;
    std::cout << "  - Total defects found: " << defect_contours.size() << std::endl;
    std::cout << "  - Threshold used: 30 intensity levels" << std::endl;
    std::cout << "  - Minimum defect area: 100 pixels" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateDimensionalMeasurement(const cv::Mat& image) {
    std::cout << "\n=== Dimensional Measurement ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // Find circular objects (drill holes, components)
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2, 2);
    
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1, 30, 50, 30, 5, 50);
    
    cv::Mat measurement_image = image.clone();
    
    // Known pixel-to-mm conversion (simulated)
    double pixels_per_mm = 2.5;  // Calibration parameter
    
    std::cout << "Circular feature measurements:" << std::endl;
    
    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point center(static_cast<int>(circles[i][0]), static_cast<int>(circles[i][1]));
        int radius = static_cast<int>(circles[i][2]);
        
        // Draw circle
        cv::circle(measurement_image, center, radius, cv::Scalar(255, 0, 0), 2);
        cv::circle(measurement_image, center, 2, cv::Scalar(255, 0, 0), 3);
        
        // Calculate diameter in mm
        double diameter_mm = (2.0 * radius) / pixels_per_mm;
        
        // Add measurement annotation
        std::string measurement = std::to_string(diameter_mm).substr(0, 4) + "mm";
        cv::putText(measurement_image, measurement, 
                   cv::Point(center.x - 20, center.y - radius - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        
        // Draw measurement lines
        cv::line(measurement_image, cv::Point(center.x - radius, center.y), 
                cv::Point(center.x + radius, center.y), cv::Scalar(0, 255, 255), 2);
        
        std::cout << "  Circle " << (i + 1) << ": Diameter = " << diameter_mm << " mm" << std::endl;
    }
    
    // Find rectangular components for length/width measurement
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::cout << "\nRectangular component measurements:" << std::endl;
    
    int rect_count = 0;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > 1000 && area < 10000) {  // Filter by size
            cv::Rect bbox = cv::boundingRect(contour);
            
            // Check if roughly rectangular
            double rect_area = bbox.width * bbox.height;
            if (area / rect_area > 0.7) {  // At least 70% rectangular
                rect_count++;
                
                // Draw bounding box
                cv::rectangle(measurement_image, bbox, cv::Scalar(0, 255, 0), 2);
                
                // Calculate dimensions in mm
                double width_mm = bbox.width / pixels_per_mm;
                double height_mm = bbox.height / pixels_per_mm;
                
                // Add dimension annotations
                std::string width_text = std::to_string(width_mm).substr(0, 4) + "mm";
                std::string height_text = std::to_string(height_mm).substr(0, 4) + "mm";
                
                cv::putText(measurement_image, width_text, 
                           cv::Point(bbox.x, bbox.y - 25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
                cv::putText(measurement_image, height_text, 
                           cv::Point(bbox.x + bbox.width + 5, bbox.y + bbox.height / 2),
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
                
                // Draw dimension lines
                cv::line(measurement_image, cv::Point(bbox.x, bbox.y - 15), 
                        cv::Point(bbox.x + bbox.width, bbox.y - 15), cv::Scalar(0, 255, 0), 1);
                cv::line(measurement_image, cv::Point(bbox.x + bbox.width + 10, bbox.y), 
                        cv::Point(bbox.x + bbox.width + 10, bbox.y + bbox.height), cv::Scalar(0, 255, 0), 1);
                
                std::cout << "  Component " << rect_count << ": " << width_mm << " x " << height_mm << " mm" << std::endl;
            }
        }
    }
    
    cv::namedWindow("Dimensional Measurement", cv::WINDOW_AUTOSIZE);
    cv::imshow("Dimensional Measurement", measurement_image);
    
    std::cout << "Measurement calibration: " << pixels_per_mm << " pixels/mm" << std::endl;
    std::cout << "Circular features: " << circles.size() << std::endl;
    std::cout << "Rectangular components: " << rect_count << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateQualityControl(const cv::Mat& image) {
    std::cout << "\n=== Automated Quality Control ===" << std::endl;
    
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // Quality control criteria
    struct QualityCheck {
        std::string name;
        bool passed;
        std::string details;
    };
    
    std::vector<QualityCheck> checks;
    
    // Check 1: Component count
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    int component_count = 0;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > 1000) {  // Significant components
            component_count++;
        }
    }
    
    bool component_check = (component_count >= 4 && component_count <= 8);
    checks.push_back({"Component Count", component_check, 
                     "Found: " + std::to_string(component_count) + " (Expected: 4-8)"});
    
    // Check 2: Drill hole presence
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2, 2);
    
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1, 30, 50, 30, 5, 15);
    
    bool hole_check = (circles.size() >= 4);
    checks.push_back({"Drill Holes", hole_check, 
                     "Found: " + std::to_string(circles.size()) + " (Expected: ≥4)"});
    
    // Check 3: Surface uniformity (using standard deviation)
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    bool uniformity_check = (stddev[0] < 60);  // Reasonable variation
    checks.push_back({"Surface Uniformity", uniformity_check, 
                     "StdDev: " + std::to_string(stddev[0]).substr(0, 4) + " (Max: 60)"});
    
    // Check 4: Color consistency (for colored components)
    std::vector<cv::Mat> bgr_channels;
    cv::split(image, bgr_channels);
    
    cv::Scalar b_mean, b_stddev;
    cv::meanStdDev(bgr_channels[0], b_mean, b_stddev);
    
    bool color_check = (b_stddev[0] < 50);
    checks.push_back({"Color Consistency", color_check, 
                     "Blue channel StdDev: " + std::to_string(b_stddev[0]).substr(0, 4) + " (Max: 50)"});
    
    // Check 5: Edge quality (sharpness)
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar laplacian_mean, laplacian_stddev;
    cv::meanStdDev(laplacian, laplacian_mean, laplacian_stddev);
    
    bool sharpness_check = (laplacian_stddev[0] > 20);  // Sufficient edge content
    checks.push_back({"Edge Sharpness", sharpness_check, 
                     "Laplacian StdDev: " + std::to_string(laplacian_stddev[0]).substr(0, 4) + " (Min: 20)"});
    
    // Create quality report visualization
    cv::Mat report_image = image.clone();
    
    // Add quality status overlay
    int passed_checks = 0;
    for (const auto& check : checks) {
        if (check.passed) passed_checks++;
    }
    
    bool overall_pass = (passed_checks == static_cast<int>(checks.size()));
    cv::Scalar status_color = overall_pass ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    std::string status_text = overall_pass ? "PASS" : "FAIL";
    
    // Draw status banner
    cv::rectangle(report_image, cv::Point(0, 0), cv::Point(200, 40), status_color, -1);
    cv::putText(report_image, status_text, cv::Point(10, 28), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // Add detailed results
    int y_pos = 60;
    for (const auto& check : checks) {
        cv::Scalar check_color = check.passed ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        std::string check_symbol = check.passed ? "✓" : "✗";
        
        cv::putText(report_image, check_symbol + " " + check.name, 
                   cv::Point(10, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.5, check_color, 1);
        y_pos += 20;
        
        cv::putText(report_image, "  " + check.details, 
                   cv::Point(10, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        y_pos += 25;
    }
    
    cv::namedWindow("Quality Control Report", cv::WINDOW_AUTOSIZE);
    cv::imshow("Quality Control Report", report_image);
    
    // Print detailed report
    std::cout << "=== QUALITY CONTROL REPORT ===" << std::endl;
    std::cout << "Overall Status: " << status_text << std::endl;
    std::cout << "Passed Checks: " << passed_checks << "/" << checks.size() << std::endl;
    std::cout << "\nDetailed Results:" << std::endl;
    
    for (const auto& check : checks) {
        std::cout << "  " << (check.passed ? "PASS" : "FAIL") << " - " 
                  << check.name << ": " << check.details << std::endl;
    }
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Defect Detection Demonstration ===" << std::endl;
    
    // Try to load real industrial images
    cv::Mat real_clean = cv::imread("../data/pcb_clean.jpg");
    cv::Mat real_defective = cv::imread("../data/pcb_defective.jpg");
    
    cv::Mat clean_image, defective_image;
    
    if (!real_clean.empty() && !real_defective.empty()) {
        std::cout << "Using real PCB images." << std::endl;
        clean_image = real_clean;
        defective_image = real_defective;
        
        // Resize if necessary
        if (clean_image.cols > 800 || clean_image.rows > 600) {
            cv::Size new_size(800, 600);
            cv::resize(clean_image, clean_image, new_size);
            cv::resize(defective_image, defective_image, new_size);
        }
    } else {
        std::cout << "Creating synthetic PCB images for demonstration." << std::endl;
        clean_image = createPCBTestImage();
        defective_image = addDefectsToPCB(clean_image);
    }
    
    // Demonstrate industrial vision techniques
    demonstrateBlobDetection(defective_image);
    demonstrateSurfaceInspection(clean_image, defective_image);
    demonstrateDimensionalMeasurement(clean_image);
    demonstrateQualityControl(defective_image);
    
    std::cout << "\n✓ Defect Detection demonstration complete!" << std::endl;
    std::cout << "Industrial vision enables automated quality control and precision measurement." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Blob detection identifies isolated defects and components
 * 2. Surface inspection compares against reference standards
 * 3. Dimensional measurement requires proper calibration
 * 4. Quality control combines multiple inspection criteria
 * 5. Automated systems need robust threshold setting
 * 6. Lighting and imaging setup critically affect results
 * 7. Statistical analysis helps set quality boundaries
 * 8. Real-time processing requires optimized algorithms
 * 9. Machine learning can improve defect classification
 * 10. Regular system calibration ensures measurement accuracy
 */
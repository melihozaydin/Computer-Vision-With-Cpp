/**
 * 36-Advanced_Demos.cpp
 * 
 * Real-time applications and complex demonstrations.
 * 
 * Concepts covered:
 * - Real-time edge detection
 * - Object counting
 * - Motion analysis
 * - Interactive applications
 * - Performance optimization
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

// Global variables for interactive demos
bool capture_running = false;
int edge_threshold1 = 50;
int edge_threshold2 = 150;
int blur_kernel = 5;
bool show_fps = true;

// Performance monitoring class
class PerformanceMonitor {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::vector<double> frame_times;
    size_t max_samples;
    
public:
    PerformanceMonitor(size_t max_samples = 30) : max_samples(max_samples) {
        frame_times.reserve(max_samples);
    }
    
    void startFrame() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void endFrame() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double ms = duration.count() / 1000.0;
        
        frame_times.push_back(ms);
        if (frame_times.size() > max_samples) {
            frame_times.erase(frame_times.begin());
        }
    }
    
    double getAverageFrameTime() const {
        if (frame_times.empty()) return 0.0;
        double sum = 0.0;
        for (double time : frame_times) {
            sum += time;
        }
        return sum / frame_times.size();
    }
    
    double getFPS() const {
        double avg_time = getAverageFrameTime();
        return avg_time > 0 ? 1000.0 / avg_time : 0.0;
    }
    
    void drawStats(cv::Mat& image, cv::Point position = cv::Point(10, 30)) const {
        if (!show_fps) return;
        
        double fps = getFPS();
        double avg_time = getAverageFrameTime();
        
        std::string fps_text = "FPS: " + std::to_string(fps).substr(0, 5);
        std::string time_text = "Frame: " + std::to_string(avg_time).substr(0, 5) + "ms";
        
        cv::putText(image, fps_text, position, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(image, time_text, cv::Point(position.x, position.y + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
};

void demonstrateRealTimeEdgeDetection() {
    std::cout << "\n=== Real-Time Edge Detection ===" << std::endl;
    
    cv::VideoCapture cap;
    
    // Try to open webcam first, then fallback to test images
    if (!cap.open(0)) {
        std::cout << "No webcam found, using static image for simulation." << std::endl;
        
        // Create a test image sequence
        cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        
        // Create animated scene
        for (int frame = 0; frame < 300; frame++) {
            test_image.setTo(cv::Scalar(50, 50, 50));
            
            // Moving circle
            int x = 100 + (frame * 2) % 440;
            cv::circle(test_image, cv::Point(x, 240), 50, cv::Scalar(255, 100, 100), -1);
            
            // Rotating rectangle
            cv::Point2f center(320, 240);
            cv::Size2f size(100, 60);
            float angle = frame * 2.0f;
            cv::RotatedRect rect(center, size, angle);
            
            cv::Point2f vertices[4];
            rect.points(vertices);
            
            std::vector<cv::Point> poly;
            for (int i = 0; i < 4; i++) {
                poly.push_back(cv::Point(static_cast<int>(vertices[i].x), static_cast<int>(vertices[i].y)));
            }
            cv::fillPoly(test_image, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(100, 255, 100));
            
            // Process frame for edge detection
            cv::Mat gray, blurred, edges;
            cv::cvtColor(test_image, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, blurred, cv::Size(blur_kernel, blur_kernel), 0);
            cv::Canny(blurred, edges, edge_threshold1, edge_threshold2);
            
            // Convert edges back to color
            cv::Mat edges_colored;
            cv::cvtColor(edges, edges_colored, cv::COLOR_GRAY2BGR);
            
            // Create side-by-side display
            cv::Mat display;
            cv::hconcat(test_image, edges_colored, display);
            
            // Add frame info
            cv::putText(display, "Frame: " + std::to_string(frame), cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            cv::putText(display, "Canny(" + std::to_string(edge_threshold1) + "," + std::to_string(edge_threshold2) + ")", 
                       cv::Point(650, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            cv::imshow("Real-Time Edge Detection", display);
            
            char key = cv::waitKey(30);
            if (key == 27 || key == 'q') break;  // ESC or 'q' to quit
        }
        
        cv::destroyAllWindows();
        return;
    }
    
    // Webcam is available
    std::cout << "Webcam detected. Starting real-time edge detection..." << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  - 'q' or ESC: Quit" << std::endl;
    std::cout << "  - '+'/'-': Adjust upper threshold" << std::endl;
    std::cout << "  - 'w'/'s': Adjust lower threshold" << std::endl;
    std::cout << "  - 'a'/'d': Adjust blur kernel size" << std::endl;
    std::cout << "  - 'f': Toggle FPS display" << std::endl;
    
    PerformanceMonitor perf_monitor;
    capture_running = true;
    
    cv::namedWindow("Real-Time Edge Detection", cv::WINDOW_AUTOSIZE);
    
    while (capture_running) {
        perf_monitor.startFrame();
        
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cout << "Failed to read frame from webcam." << std::endl;
            break;
        }
        
        // Process frame
        cv::Mat gray, blurred, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // Ensure odd kernel size
        int kernel_size = (blur_kernel % 2 == 0) ? blur_kernel + 1 : blur_kernel;
        cv::GaussianBlur(gray, blurred, cv::Size(kernel_size, kernel_size), 0);
        cv::Canny(blurred, edges, edge_threshold1, edge_threshold2);
        
        // Convert edges to color for display
        cv::Mat edges_colored;
        cv::cvtColor(edges, edges_colored, cv::COLOR_GRAY2BGR);
        
        // Create side-by-side display
        cv::Mat display;
        cv::hconcat(frame, edges_colored, display);
        
        // Add performance stats
        perf_monitor.endFrame();
        perf_monitor.drawStats(display);
        
        // Add parameter info
        cv::putText(display, "Thresholds: " + std::to_string(edge_threshold1) + "," + std::to_string(edge_threshold2), 
                   cv::Point(10, display.rows - 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(display, "Blur: " + std::to_string(kernel_size), 
                   cv::Point(10, display.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Real-Time Edge Detection", display);
        
        // Handle user input
        char key = cv::waitKey(1);
        if (key == 27 || key == 'q') {  // ESC or 'q'
            capture_running = false;
        } else if (key == '+' || key == '=') {
            edge_threshold2 = std::min(255, edge_threshold2 + 10);
        } else if (key == '-') {
            edge_threshold2 = std::max(1, edge_threshold2 - 10);
        } else if (key == 'w') {
            edge_threshold1 = std::min(edge_threshold2 - 1, edge_threshold1 + 5);
        } else if (key == 's') {
            edge_threshold1 = std::max(1, edge_threshold1 - 5);
        } else if (key == 'a') {
            blur_kernel = std::max(1, blur_kernel - 2);
        } else if (key == 'd') {
            blur_kernel = std::min(21, blur_kernel + 2);
        } else if (key == 'f') {
            show_fps = !show_fps;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
}

void demonstrateObjectCounting() {
    std::cout << "\n=== Automated Object Counting ===" << std::endl;
    
    // Create test images with different numbers of objects
    std::vector<cv::Mat> test_images;
    std::vector<int> expected_counts;
    
    for (int test_case = 0; test_case < 4; test_case++) {
        cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
        image.setTo(cv::Scalar(50, 50, 50));  // Dark background
        
        int object_count = 5 + test_case * 3;  // 5, 8, 11, 14 objects
        expected_counts.push_back(object_count);
        
        // Create random objects
        std::srand(test_case * 42);  // Reproducible randomness
        
        for (int i = 0; i < object_count; i++) {
            int x = 50 + std::rand() % 500;
            int y = 50 + std::rand() % 300;
            int radius = 15 + std::rand() % 20;
            
            // Random color
            cv::Scalar color(std::rand() % 200 + 55, std::rand() % 200 + 55, std::rand() % 200 + 55);
            
            // Add some variety: circles and rectangles
            if (i % 3 == 0) {
                cv::rectangle(image, cv::Point(x - radius, y - radius), 
                             cv::Point(x + radius, y + radius), color, -1);
            } else {
                cv::circle(image, cv::Point(x, y), radius, color, -1);
            }
        }
        
        // Add noise
        cv::Mat noise(image.size(), CV_8UC3);
        cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(30));
        cv::add(image, noise, image);
        
        test_images.push_back(image);
    }
    
    // Process each test image
    for (size_t i = 0; i < test_images.size(); i++) {
        cv::Mat image = test_images[i];
        
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
        // Threshold to create binary image
        cv::Mat binary;
        cv::threshold(gray, binary, 80, 255, cv::THRESH_BINARY);
        
        // Morphological operations to clean up
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::Mat cleaned;
        cv::morphologyEx(binary, cleaned, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel);
        
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cleaned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Filter contours by area
        std::vector<std::vector<cv::Point>> valid_contours;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > 200 && area < 5000) {  // Filter by reasonable object size
                valid_contours.push_back(contour);
            }
        }
        
        // Draw results
        cv::Mat result = image.clone();
        
        for (size_t j = 0; j < valid_contours.size(); j++) {
            // Draw contour
            cv::drawContours(result, valid_contours, static_cast<int>(j), cv::Scalar(0, 255, 0), 2);
            
            // Add number label
            cv::Moments moments = cv::moments(valid_contours[j]);
            if (moments.m00 != 0) {
                cv::Point centroid(static_cast<int>(moments.m10 / moments.m00),
                                 static_cast<int>(moments.m01 / moments.m00));
                cv::putText(result, std::to_string(j + 1), centroid,
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
            }
        }
        
        // Add count information
        cv::putText(result, "Detected: " + std::to_string(valid_contours.size()), 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::putText(result, "Expected: " + std::to_string(expected_counts[i]), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        
        // Accuracy assessment
        bool accurate = (static_cast<int>(valid_contours.size()) == expected_counts[i]);
        cv::Scalar accuracy_color = accurate ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        std::string accuracy_text = accurate ? "ACCURATE" : "INACCURATE";
        cv::putText(result, accuracy_text, cv::Point(10, 90), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, accuracy_color, 2);
        
        // Create binary visualization
        cv::Mat binary_colored;
        cv::cvtColor(cleaned, binary_colored, cv::COLOR_GRAY2BGR);
        
        // Side-by-side display
        cv::Mat display;
        cv::hconcat(result, binary_colored, display);
        
        cv::putText(display, "Test Case " + std::to_string(i + 1), 
                   cv::Point(result.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(display, "Binary Mask", 
                   cv::Point(result.cols + 10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::namedWindow("Object Counting", cv::WINDOW_AUTOSIZE);
        cv::imshow("Object Counting", display);
        
        std::cout << "Test " << (i + 1) << ": Expected=" << expected_counts[i] 
                  << ", Detected=" << valid_contours.size() 
                  << ", Accuracy=" << accuracy_text << std::endl;
        
        cv::waitKey(0);
    }
    
    cv::destroyAllWindows();
}

void demonstrateMotionAnalysis() {
    std::cout << "\n=== Motion Analysis Demo ===" << std::endl;
    
    // Create synthetic video sequence with moving objects
    std::vector<cv::Mat> frames;
    int num_frames = 50;
    
    for (int f = 0; f < num_frames; f++) {
        cv::Mat frame = cv::Mat::zeros(400, 600, CV_8UC3);
        frame.setTo(cv::Scalar(30, 30, 30));
        
        // Moving circle (linear motion)
        int circle_x = 50 + (f * 10) % 500;
        cv::circle(frame, cv::Point(circle_x, 150), 30, cv::Scalar(255, 100, 100), -1);
        
        // Moving rectangle (sinusoidal motion)
        int rect_x = 300 + static_cast<int>(150 * std::sin(f * 0.2));
        cv::rectangle(frame, cv::Point(rect_x - 25, 250), cv::Point(rect_x + 25, 300), 
                     cv::Scalar(100, 255, 100), -1);
        
        // Static background elements
        cv::circle(frame, cv::Point(500, 100), 20, cv::Scalar(100, 100, 255), -1);
        cv::rectangle(frame, cv::Point(100, 300), cv::Point(150, 350), cv::Scalar(255, 255, 100), -1);
        
        frames.push_back(frame);
    }
    
    // Motion detection using frame differencing
    std::cout << "Analyzing motion in synthetic video..." << std::endl;
    
    for (size_t i = 1; i < frames.size(); i++) {
        cv::Mat current_frame = frames[i];
        cv::Mat previous_frame = frames[i - 1];
        
        // Convert to grayscale
        cv::Mat current_gray, previous_gray;
        cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(previous_frame, previous_gray, cv::COLOR_BGR2GRAY);
        
        // Frame difference
        cv::Mat diff;
        cv::absdiff(current_gray, previous_gray, diff);
        
        // Threshold to detect significant motion
        cv::Mat motion_mask;
        cv::threshold(diff, motion_mask, 25, 255, cv::THRESH_BINARY);
        
        // Morphological operations to clean up
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(motion_mask, motion_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(motion_mask, motion_mask, cv::MORPH_CLOSE, kernel);
        
        // Find motion regions
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(motion_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Draw motion analysis results
        cv::Mat result = current_frame.clone();
        
        int motion_count = 0;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > 100) {  // Filter small motion
                motion_count++;
                
                // Draw bounding box
                cv::Rect bbox = cv::boundingRect(contour);
                cv::rectangle(result, bbox, cv::Scalar(0, 255, 255), 2);
                
                // Calculate center
                cv::Moments moments = cv::moments(contour);
                if (moments.m00 != 0) {
                    cv::Point center(static_cast<int>(moments.m10 / moments.m00),
                                   static_cast<int>(moments.m01 / moments.m00));
                    cv::circle(result, center, 5, cv::Scalar(0, 0, 255), -1);
                    
                    // Add motion vector (simplified)
                    cv::arrowedLine(result, center, cv::Point(center.x + 20, center.y), 
                                   cv::Scalar(255, 0, 0), 2);
                }
            }
        }
        
        // Add motion statistics
        cv::putText(result, "Frame: " + std::to_string(i), cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(result, "Motion Regions: " + std::to_string(motion_count), cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Create motion visualization
        cv::Mat motion_colored;
        cv::cvtColor(motion_mask, motion_colored, cv::COLOR_GRAY2BGR);
        
        cv::Mat display;
        cv::hconcat(result, motion_colored, display);
        
        cv::putText(display, "Motion Detection", cv::Point(result.cols + 10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::namedWindow("Motion Analysis", cv::WINDOW_AUTOSIZE);
        cv::imshow("Motion Analysis", display);
        
        char key = cv::waitKey(100);  // Slow playback
        if (key == 27 || key == 'q') break;
    }
    
    cv::destroyAllWindows();
}

void demonstrateInteractiveApplication() {
    std::cout << "\n=== Interactive Computer Vision Application ===" << std::endl;
    
    // Create an interactive drawing and analysis application
    cv::Mat canvas = cv::Mat::zeros(600, 800, CV_8UC3);
    cv::Mat overlay = cv::Mat::zeros(600, 800, CV_8UC3);
    
    bool drawing = false;
    bool analyzing = false;
    cv::Point last_point(-1, -1);
    int brush_size = 5;
    cv::Scalar brush_color(255, 255, 255);
    
    // Mouse callback
    auto mouse_callback = [](int event, int x, int y, int, void* userdata) {
        auto* data = static_cast<std::tuple<cv::Mat&, cv::Mat&, bool&, cv::Point&, int&, cv::Scalar&>*>(userdata);
        cv::Mat& canvas = std::get<0>(*data);
        //cv::Mat& overlay = std::get<1>(*data);  // Available for overlays if needed
        bool& drawing = std::get<2>(*data);
        cv::Point& last_point = std::get<3>(*data);
        int& brush_size = std::get<4>(*data);
        cv::Scalar& brush_color = std::get<5>(*data);
        
        if (event == cv::EVENT_LBUTTONDOWN) {
            drawing = true;
            last_point = cv::Point(x, y);
            cv::circle(canvas, cv::Point(x, y), brush_size, brush_color, -1);
        } else if (event == cv::EVENT_MOUSEMOVE && drawing) {
            cv::line(canvas, last_point, cv::Point(x, y), brush_color, brush_size * 2);
            last_point = cv::Point(x, y);
        } else if (event == cv::EVENT_LBUTTONUP) {
            drawing = false;
        }
    };
    
    auto callback_data = std::make_tuple(std::ref(canvas), std::ref(overlay), 
                                       std::ref(drawing), std::ref(last_point), 
                                       std::ref(brush_size), std::ref(brush_color));
    
    cv::namedWindow("Interactive CV Application", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Interactive CV Application", mouse_callback, &callback_data);
    
    std::cout << "Interactive application controls:" << std::endl;
    std::cout << "  - Left click and drag: Draw on canvas" << std::endl;
    std::cout << "  - 'c': Clear canvas" << std::endl;
    std::cout << "  - 'a': Analyze drawing (contours, features)" << std::endl;
    std::cout << "  - 'r/g/b': Change brush color to Red/Green/Blue" << std::endl;
    std::cout << "  - 'w': Change brush color to White" << std::endl;
    std::cout << "  - '+'/'-': Increase/Decrease brush size" << std::endl;
    std::cout << "  - 's': Save current drawing" << std::endl;
    std::cout << "  - 'q': Quit" << std::endl;
    
    while (true) {
        cv::Mat display;
        if (analyzing) {
            cv::addWeighted(canvas, 0.7, overlay, 0.3, 0, display);
        } else {
            display = canvas.clone();
        }
        
        // Add UI elements
        cv::circle(display, cv::Point(750, 50), brush_size + 5, brush_color, 2);
        cv::putText(display, "Brush", cv::Point(720, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        cv::putText(display, "Controls: c-clear, a-analyze, r/g/b/w-color, +/- size", 
                   cv::Point(10, display.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        
        cv::imshow("Interactive CV Application", display);
        
        char key = cv::waitKey(30);
        if (key == 'q' || key == 27) break;
        
        switch (key) {
            case 'c':  // Clear
                canvas = cv::Mat::zeros(600, 800, CV_8UC3);
                overlay = cv::Mat::zeros(600, 800, CV_8UC3);
                analyzing = false;
                break;
                
            case 'a':  // Analyze
                {
                    overlay = cv::Mat::zeros(600, 800, CV_8UC3);
                    
                    // Convert to grayscale for analysis
                    cv::Mat gray;
                    cv::cvtColor(canvas, gray, cv::COLOR_BGR2GRAY);
                    
                    // Find contours
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                    
                    // Analyze each contour
                    for (size_t i = 0; i < contours.size(); i++) {
                        double area = cv::contourArea(contours[i]);
                        if (area > 100) {  // Filter small contours
                            
                            // Draw contour
                            cv::drawContours(overlay, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
                            
                            // Bounding box
                            cv::Rect bbox = cv::boundingRect(contours[i]);
                            cv::rectangle(overlay, bbox, cv::Scalar(255, 0, 0), 1);
                            
                            // Centroid
                            cv::Moments moments = cv::moments(contours[i]);
                            if (moments.m00 != 0) {
                                cv::Point centroid(static_cast<int>(moments.m10 / moments.m00),
                                                 static_cast<int>(moments.m01 / moments.m00));
                                cv::circle(overlay, centroid, 5, cv::Scalar(0, 0, 255), -1);
                                
                                // Add area label
                                cv::putText(overlay, std::to_string(static_cast<int>(area)), 
                                           cv::Point(centroid.x + 10, centroid.y),
                                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
                            }
                            
                            // Convex hull
                            std::vector<cv::Point> hull;
                            cv::convexHull(contours[i], hull);
                            std::vector<std::vector<cv::Point>> hull_vec = {hull};
                            cv::drawContours(overlay, hull_vec, 0, cv::Scalar(0, 255, 255), 1);
                        }
                    }
                    
                    analyzing = true;
                    std::cout << "Analysis complete: Found " << contours.size() << " shapes" << std::endl;
                }
                break;
                
            case 'r':  // Red
                brush_color = cv::Scalar(0, 0, 255);
                break;
            case 'g':  // Green
                brush_color = cv::Scalar(0, 255, 0);
                break;
            case 'b':  // Blue
                brush_color = cv::Scalar(255, 0, 0);
                break;
            case 'w':  // White
                brush_color = cv::Scalar(255, 255, 255);
                break;
                
            case '+':
            case '=':
                brush_size = std::min(20, brush_size + 1);
                break;
            case '-':
                brush_size = std::max(1, brush_size - 1);
                break;
                
            case 's':  // Save
                cv::imwrite("interactive_drawing.png", canvas);
                std::cout << "Drawing saved as 'interactive_drawing.png'" << std::endl;
                break;
        }
    }
    
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Advanced Computer Vision Demonstrations ===" << std::endl;
    std::cout << "This module showcases real-time processing and interactive applications." << std::endl;
    
    // Menu for selecting demonstrations
    while (true) {
        std::cout << "\n=== Demo Selection Menu ===" << std::endl;
        std::cout << "1. Real-Time Edge Detection" << std::endl;
        std::cout << "2. Automated Object Counting" << std::endl;
        std::cout << "3. Motion Analysis" << std::endl;
        std::cout << "4. Interactive CV Application" << std::endl;
        std::cout << "5. Exit" << std::endl;
        std::cout << "Select demo (1-5): ";
        
        int choice;
        std::cin >> choice;
        
        switch (choice) {
            case 1:
                demonstrateRealTimeEdgeDetection();
                break;
            case 2:
                demonstrateObjectCounting();
                break;
            case 3:
                demonstrateMotionAnalysis();
                break;
            case 4:
                demonstrateInteractiveApplication();
                break;
            case 5:
                std::cout << "Exiting advanced demonstrations." << std::endl;
                return 0;
            default:
                std::cout << "Invalid choice. Please select 1-5." << std::endl;
                break;
        }
    }
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Real-time processing requires performance optimization
 * 2. Frame rate monitoring helps identify bottlenecks
 * 3. Interactive applications need responsive user interfaces
 * 4. Object counting combines thresholding and contour analysis
 * 5. Motion detection uses temporal frame differences
 * 6. User input handling enables parameter adjustment
 * 7. Morphological operations clean up noisy detections
 * 8. Multi-threading can improve real-time performance
 * 9. Memory management is crucial for continuous processing
 * 10. Practical applications combine multiple CV techniques
 */
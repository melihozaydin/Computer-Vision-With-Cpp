/**
 * 25-Optical_Flow.cpp
 * 
 * Motion estimation using optical flow algorithms.
 * 
 * Concepts covered:
 * - Lucas-Kanade optical flow
 * - Farneback dense flow
 * - Motion tracking
 * - Flow field visualization
 * - Applications in tracking
 */

#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <vector>
#include <random>

// Create synthetic motion sequence
std::vector<cv::Mat> createMotionSequence() {
    std::vector<cv::Mat> sequence;
    
    for (int frame = 0; frame < 30; frame++) {
        cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
        
        // Moving circle
        int circle_x = 100 + frame * 15;
        int circle_y = 200 + sin(frame * 0.2) * 50;
        cv::circle(image, cv::Point(circle_x, circle_y), 30, cv::Scalar(0, 255, 0), -1);
        
        // Moving rectangle
        int rect_x = 500 - frame * 10;
        int rect_y = 150 + cos(frame * 0.15) * 40;
        cv::rectangle(image, cv::Point(rect_x - 25, rect_y - 20), 
                     cv::Point(rect_x + 25, rect_y + 20), cv::Scalar(255, 0, 0), -1);
        
        // Static elements
        cv::rectangle(image, cv::Point(50, 50), cv::Point(150, 100), cv::Scalar(100, 100, 100), -1);
        cv::circle(image, cv::Point(300, 100), 25, cv::Scalar(150, 150, 150), -1);
        
        // Add some texture for feature tracking
        std::random_device rd;
        std::mt19937 gen(frame);  // Fixed seed per frame for consistency
        std::uniform_int_distribution<> dis_x(0, image.cols - 1);
        std::uniform_int_distribution<> dis_y(0, image.rows - 1);
        std::uniform_int_distribution<> dis_color(100, 255);
        
        for (int i = 0; i < 20; i++) {
            cv::circle(image, cv::Point(dis_x(gen), dis_y(gen)), 2, 
                      cv::Scalar(dis_color(gen), dis_color(gen), dis_color(gen)), -1);
        }
        
        sequence.push_back(image);
    }
    
    return sequence;
}

void demonstrateLucasKanadeFlow(const std::vector<cv::Mat>& sequence) {
    std::cout << "\n=== Lucas-Kanade Optical Flow ===" << std::endl;
    
    if (sequence.size() < 2) {
        std::cout << "Need at least 2 frames for optical flow." << std::endl;
        return;
    }
    
    // Convert first frame to grayscale and detect features
    cv::Mat prev_gray, curr_gray;
    cv::cvtColor(sequence[0], prev_gray, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::Point2f> prev_points, curr_points;
    
    // Detect good features to track
    cv::goodFeaturesToTrack(prev_gray, prev_points, 100, 0.01, 10);
    
    std::cout << "Initial features detected: " << prev_points.size() << std::endl;
    
    // Parameters for Lucas-Kanade
    cv::Size win_size(15, 15);
    int max_level = 2;
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.03);
    
    // Track through sequence
    std::vector<cv::Mat> flow_visualization;
    
    for (size_t i = 1; i < std::min(sequence.size(), size_t(10)); i++) {
        cv::cvtColor(sequence[i], curr_gray, cv::COLOR_BGR2GRAY);
        
        std::vector<uchar> status;
        std::vector<float> error;
        
        // Calculate optical flow
        cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, curr_points, 
                                status, error, win_size, max_level, criteria);
        
        // Visualize flow
        cv::Mat flow_vis = sequence[i].clone();
        
        // Draw tracked points and motion vectors
        int valid_tracks = 0;
        for (size_t j = 0; j < prev_points.size(); j++) {
            if (status[j] && error[j] < 50.0) {
                // Draw motion vector
                cv::arrowedLine(flow_vis, prev_points[j], curr_points[j], 
                               cv::Scalar(0, 255, 255), 2);
                
                // Draw current point
                cv::circle(flow_vis, curr_points[j], 3, cv::Scalar(0, 255, 0), -1);
                
                valid_tracks++;
            }
        }
        
        // Add frame info
        cv::putText(flow_vis, "Frame " + std::to_string(i), 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(flow_vis, "Tracks: " + std::to_string(valid_tracks), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        flow_visualization.push_back(flow_vis);
        
        // Update for next iteration - keep only good points
        std::vector<cv::Point2f> good_prev, good_curr;
        for (size_t j = 0; j < prev_points.size(); j++) {
            if (status[j] && error[j] < 50.0) {
                good_prev.push_back(prev_points[j]);
                good_curr.push_back(curr_points[j]);
            }
        }
        
        prev_points = good_curr;
        prev_gray = curr_gray.clone();
        
        std::cout << "Frame " << i << ": " << valid_tracks << " valid tracks" << std::endl;
    }
    
    // Display flow visualization
    for (size_t i = 0; i < flow_visualization.size(); i++) {
        cv::namedWindow("Lucas-Kanade Flow", cv::WINDOW_AUTOSIZE);
        cv::imshow("Lucas-Kanade Flow", flow_visualization[i]);
        cv::waitKey(500);  // Show each frame for 500ms
    }
    
    std::cout << "Lucas-Kanade characteristics:" << std::endl;
    std::cout << "  - Sparse optical flow (tracks specific features)" << std::endl;
    std::cout << "  - Works well for small motions" << std::endl;
    std::cout << "  - Requires good features to track" << std::endl;
    std::cout << "  - Yellow arrows show motion vectors" << std::endl;
    
    cv::destroyAllWindows();
}

void demonstrateFarnebackFlow(const std::vector<cv::Mat>& sequence) {
    std::cout << "\n=== Farneback Dense Optical Flow ===" << std::endl;
    
    if (sequence.size() < 2) {
        std::cout << "Need at least 2 frames for optical flow." << std::endl;
        return;
    }
    
    cv::Mat prev_gray, curr_gray;
    cv::cvtColor(sequence[0], prev_gray, cv::COLOR_BGR2GRAY);
    
    for (size_t i = 1; i < std::min(sequence.size(), size_t(5)); i++) {
        cv::cvtColor(sequence[i], curr_gray, cv::COLOR_BGR2GRAY);
        
        // Calculate dense optical flow
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        
        // Visualize flow field
        cv::Mat flow_vis = cv::Mat::zeros(flow.size(), CV_8UC3);
        
        // Convert flow to HSV for visualization
        cv::Mat magnitude, angle;
        cv::Mat flow_split[2];
        cv::split(flow, flow_split);
        
        cv::cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
        
        // Create HSV image where hue represents direction and value represents magnitude
        cv::Mat hsv = cv::Mat::zeros(flow.size(), CV_8UC3);
        cv::Mat hsv_split[3];
        
        // Hue channel (direction)
        angle.convertTo(hsv_split[0], CV_8U, 255.0 / 360.0);
        
        // Saturation channel (constant)
        hsv_split[1] = cv::Mat::ones(flow.size(), CV_8U) * 255;
        
        // Value channel (magnitude)
        cv::normalize(magnitude, hsv_split[2], 0, 255, cv::NORM_MINMAX, CV_8U);
        
        cv::merge(hsv_split, 3, hsv);
        cv::cvtColor(hsv, flow_vis, cv::COLOR_HSV2BGR);
        
        // Also create arrow visualization
        cv::Mat arrow_vis = sequence[i].clone();
        int step = 20;  // Sample every 20 pixels
        
        for (int y = step; y < flow.rows; y += step) {
            for (int x = step; x < flow.cols; x += step) {
                cv::Point2f flow_vec = flow.at<cv::Point2f>(y, x);
                
                if (cv::norm(flow_vec) > 1.0) {  // Only draw significant motion
                    cv::Point start(x, y);
                    cv::Point end(x + static_cast<int>(flow_vec.x), y + static_cast<int>(flow_vec.y));
                    
                    cv::arrowedLine(arrow_vis, start, end, cv::Scalar(0, 255, 0), 1);
                }
            }
        }
        
        // Combine visualizations
        cv::Mat combined = cv::Mat::zeros(flow.rows, flow.cols * 3, CV_8UC3);
        
        sequence[i].copyTo(combined(cv::Rect(0, 0, flow.cols, flow.rows)));
        flow_vis.copyTo(combined(cv::Rect(flow.cols, 0, flow.cols, flow.rows)));
        arrow_vis.copyTo(combined(cv::Rect(flow.cols * 2, 0, flow.cols, flow.rows)));
        
        // Add labels
        cv::putText(combined, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(combined, "Flow Field", cv::Point(flow.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(combined, "Flow Vectors", cv::Point(flow.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::namedWindow("Farneback Dense Flow", cv::WINDOW_AUTOSIZE);
        cv::imshow("Farneback Dense Flow", combined);
        cv::waitKey(1000);
        
        prev_gray = curr_gray.clone();
        
        // Calculate flow statistics
        cv::Scalar mean_flow = cv::mean(magnitude);
        double min_mag, max_mag;
        cv::minMaxLoc(magnitude, &min_mag, &max_mag);
        
        std::cout << "Frame " << i << " flow statistics:" << std::endl;
        std::cout << "  - Mean magnitude: " << mean_flow[0] << std::endl;
        std::cout << "  - Max magnitude: " << max_mag << std::endl;
    }
    
    std::cout << "Farneback flow characteristics:" << std::endl;
    std::cout << "  - Dense optical flow (every pixel has motion vector)" << std::endl;
    std::cout << "  - Color coding: Hue = direction, Brightness = magnitude" << std::endl;
    std::cout << "  - Good for understanding overall motion patterns" << std::endl;
    std::cout << "  - More computationally expensive than sparse methods" << std::endl;
    
    cv::destroyAllWindows();
}

void demonstrateMotionDetection(const std::vector<cv::Mat>& sequence) {
    std::cout << "\n=== Motion Detection using Optical Flow ===" << std::endl;
    
    if (sequence.size() < 2) return;
    
    cv::Mat prev_gray, curr_gray;
    cv::cvtColor(sequence[0], prev_gray, cv::COLOR_BGR2GRAY);
    
    // Background subtractor for comparison
    cv::Ptr<cv::BackgroundSubtractor> bg_subtractor = cv::createBackgroundSubtractorMOG2();
    bg_subtractor->apply(sequence[0], cv::Mat());  // Initialize with first frame
    
    for (size_t i = 1; i < std::min(sequence.size(), size_t(10)); i++) {
        cv::cvtColor(sequence[i], curr_gray, cv::COLOR_BGR2GRAY);
        
        // Method 1: Dense optical flow for motion detection
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        
        // Calculate motion magnitude
        cv::Mat flow_split[2];
        cv::split(flow, flow_split);
        cv::Mat magnitude;
        cv::cartToPolar(flow_split[0], flow_split[1], magnitude, cv::Mat());
        
        // Threshold motion magnitude
        cv::Mat motion_mask;
        cv::threshold(magnitude, motion_mask, 2.0, 255, cv::THRESH_BINARY);
        motion_mask.convertTo(motion_mask, CV_8U);
        
        // Method 2: Background subtraction for comparison
        cv::Mat bg_mask;
        bg_subtractor->apply(sequence[i], bg_mask);
        
        // Morphological operations to clean up
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(motion_mask, motion_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(motion_mask, motion_mask, cv::MORPH_CLOSE, kernel);
        
        cv::morphologyEx(bg_mask, bg_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(bg_mask, bg_mask, cv::MORPH_CLOSE, kernel);
        
        // Find contours of moving objects
        std::vector<std::vector<cv::Point>> contours_flow, contours_bg;
        cv::findContours(motion_mask, contours_flow, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(bg_mask, contours_bg, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Visualize results
        cv::Mat result_flow = sequence[i].clone();
        cv::Mat result_bg = sequence[i].clone();
        
        // Draw optical flow based motion
        for (const auto& contour : contours_flow) {
            if (cv::contourArea(contour) > 100) {  // Filter small areas
                cv::Rect bbox = cv::boundingRect(contour);
                cv::rectangle(result_flow, bbox, cv::Scalar(0, 255, 0), 2);
                cv::putText(result_flow, "Moving", cv::Point(bbox.x, bbox.y - 10), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }
        
        // Draw background subtraction based motion
        for (const auto& contour : contours_bg) {
            if (cv::contourArea(contour) > 100) {
                cv::Rect bbox = cv::boundingRect(contour);
                cv::rectangle(result_bg, bbox, cv::Scalar(255, 0, 0), 2);
                cv::putText(result_bg, "Moving", cv::Point(bbox.x, bbox.y - 10), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
        }
        
        // Create comparison display
        cv::Mat display = cv::Mat::zeros(sequence[i].rows * 2, sequence[i].cols * 2, CV_8UC3);
        
        sequence[i].copyTo(display(cv::Rect(0, 0, sequence[i].cols, sequence[i].rows)));
        result_flow.copyTo(display(cv::Rect(sequence[i].cols, 0, sequence[i].cols, sequence[i].rows)));
        result_bg.copyTo(display(cv::Rect(0, sequence[i].rows, sequence[i].cols, sequence[i].rows)));
        
        cv::cvtColor(motion_mask, motion_mask, cv::COLOR_GRAY2BGR);
        motion_mask.copyTo(display(cv::Rect(sequence[i].cols, sequence[i].rows, sequence[i].cols, sequence[i].rows)));
        
        // Add labels
        cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(display, "Optical Flow", cv::Point(sequence[i].cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(display, "Background Sub", cv::Point(10, sequence[i].rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(display, "Motion Mask", cv::Point(sequence[i].cols + 10, sequence[i].rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::namedWindow("Motion Detection Comparison", cv::WINDOW_AUTOSIZE);
        cv::imshow("Motion Detection Comparison", display);
        cv::waitKey(800);
        
        prev_gray = curr_gray.clone();
        
        std::cout << "Frame " << i << ": Flow detections=" << contours_flow.size() 
                  << ", BG detections=" << contours_bg.size() << std::endl;
    }
    
    std::cout << "Motion detection comparison:" << std::endl;
    std::cout << "  - Optical flow: Detects actual motion vectors" << std::endl;
    std::cout << "  - Background subtraction: Detects changes from background model" << std::endl;
    std::cout << "  - Optical flow better for understanding motion direction" << std::endl;
    std::cout << "  - Background subtraction better for static camera scenarios" << std::endl;
    
    cv::destroyAllWindows();
}

void demonstrateFlowQuality(const std::vector<cv::Mat>& sequence) {
    std::cout << "\n=== Optical Flow Quality Assessment ===" << std::endl;
    
    if (sequence.size() < 2) return;
    
    cv::Mat prev_gray, curr_gray;
    cv::cvtColor(sequence[0], prev_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(sequence[1], curr_gray, cv::COLOR_BGR2GRAY);
    
    // Compare different optical flow parameters
    struct FlowParams {
        double pyr_scale;
        int levels;
        int winsize;
        int iterations;
        int poly_n;
        double poly_sigma;
        std::string name;
    };
    
    std::vector<FlowParams> param_sets = {
        {0.5, 3, 15, 3, 5, 1.2, "Default"},
        {0.75, 2, 10, 2, 5, 1.1, "Fast"},
        {0.5, 4, 20, 5, 7, 1.5, "Accurate"},
        {0.8, 1, 8, 1, 3, 1.0, "Minimal"}
    };
    
    cv::Mat display = cv::Mat::zeros(curr_gray.rows * 2, curr_gray.cols * 2, CV_8UC3);
    
    for (size_t i = 0; i < param_sets.size(); i++) {
        const auto& params = param_sets[i];
        
        // Calculate flow with specific parameters
        cv::Mat flow;
        auto start_time = cv::getTickCount();
        
        cv::calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 
                                    params.pyr_scale, params.levels, params.winsize, 
                                    params.iterations, params.poly_n, params.poly_sigma, 0);
        
        auto end_time = cv::getTickCount();
        double elapsed_time = (end_time - start_time) / cv::getTickFrequency() * 1000.0;
        
        // Visualize flow
        cv::Mat flow_split[2];
        cv::split(flow, flow_split);
        cv::Mat magnitude, angle;
        cv::cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
        
        cv::Mat hsv = cv::Mat::zeros(flow.size(), CV_8UC3);
        cv::Mat hsv_split[3];
        
        angle.convertTo(hsv_split[0], CV_8U, 255.0 / 360.0);
        hsv_split[1] = cv::Mat::ones(flow.size(), CV_8U) * 255;
        cv::normalize(magnitude, hsv_split[2], 0, 255, cv::NORM_MINMAX, CV_8U);
        
        cv::merge(hsv_split, 3, hsv);
        cv::Mat flow_vis;
        cv::cvtColor(hsv, flow_vis, cv::COLOR_HSV2BGR);
        
        // Add performance info
        cv::Scalar mean_mag = cv::mean(magnitude);
        cv::putText(flow_vis, params.name, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        cv::putText(flow_vis, std::to_string(static_cast<int>(elapsed_time)) + "ms", 
                   cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(flow_vis, "Mag:" + std::to_string(mean_mag[0]).substr(0, 4), 
                   cv::Point(10, 75), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // Place in grid
        int row = i / 2;
        int col = i % 2;
        flow_vis.copyTo(display(cv::Rect(col * curr_gray.cols, row * curr_gray.rows, curr_gray.cols, curr_gray.rows)));
        
        std::cout << params.name << " parameters - Time: " << elapsed_time << "ms, Mean magnitude: " << mean_mag[0] << std::endl;
    }
    
    cv::namedWindow("Flow Quality Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("Flow Quality Comparison", display);
    
    std::cout << "\nParameter trade-offs:" << std::endl;
    std::cout << "  - Pyramid scale: Lower = more levels, more accurate but slower" << std::endl;
    std::cout << "  - Levels: More levels = handle larger motions" << std::endl;
    std::cout << "  - Window size: Larger = more robust but less precise" << std::endl;
    std::cout << "  - Iterations: More = better convergence but slower" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Optical Flow ===" << std::endl;
    
    // Create synthetic motion sequence
    std::vector<cv::Mat> motion_sequence = createMotionSequence();
    std::cout << "Created motion sequence with " << motion_sequence.size() << " frames" << std::endl;
    
    // Try to load video file for real motion analysis
    cv::VideoCapture cap("../data/motion.mp4");
    if (!cap.isOpened()) {
        std::cout << "No video file found, using synthetic sequence." << std::endl;
    } else {
        std::cout << "Loading frames from video file..." << std::endl;
        motion_sequence.clear();
        
        cv::Mat frame;
        int frame_count = 0;
        while (cap.read(frame) && frame_count < 20) {
            if (frame.cols > 600) {
                cv::resize(frame, frame, cv::Size(600, frame.rows * 600 / frame.cols));
            }
            motion_sequence.push_back(frame.clone());
            frame_count++;
        }
        cap.release();
        std::cout << "Loaded " << motion_sequence.size() << " frames from video." << std::endl;
    }
    
    // Demonstrate optical flow techniques
    demonstrateLucasKanadeFlow(motion_sequence);
    demonstrateFarnebackFlow(motion_sequence);
    demonstrateMotionDetection(motion_sequence);
    demonstrateFlowQuality(motion_sequence);
    
    std::cout << "\n✓ Optical Flow demonstration complete!" << std::endl;
    std::cout << "Optical flow enables motion analysis and object tracking in video sequences." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Lucas-Kanade tracks sparse features between frames
 * 2. Farneback provides dense flow fields for every pixel
 * 3. Optical flow assumes brightness constancy and small motion
 * 4. Pyramid processing handles larger motions across scales
 * 5. Flow visualization uses color coding for direction and magnitude
 * 6. Motion detection can be based on flow magnitude thresholding
 * 7. Parameter tuning balances accuracy vs computational cost
 * 8. Flow quality depends on texture and lighting conditions
 * 9. Applications include motion tracking, video stabilization, object detection
 * 10. Dense flow is computationally expensive but provides complete motion field
 */

/**
 * 23-Cascade_Classifiers.cpp
 * 
 * Object detection using Haar and LBP cascades.
 * 
 * Concepts covered:
 * - Haar cascade classifiers
 * - LBP cascade classifiers
 * - Face detection
 * - Eye detection
 * - Multi-scale detection
 * - Detection parameter tuning
 */

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <string>

// Function to create a test image with faces for demonstration
cv::Mat createTestFaceImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // Draw simple face-like structures for testing when no real image is available
    cv::circle(image, cv::Point(150, 150), 80, cv::Scalar(200, 180, 160), -1);  // Face 1
    cv::circle(image, cv::Point(130, 130), 8, cv::Scalar(0, 0, 0), -1);         // Eye 1
    cv::circle(image, cv::Point(170, 130), 8, cv::Scalar(0, 0, 0), -1);         // Eye 2
    cv::ellipse(image, cv::Point(150, 170), cv::Size(20, 10), 0, 0, 180, cv::Scalar(0, 0, 0), 2); // Mouth
    
    cv::circle(image, cv::Point(450, 200), 60, cv::Scalar(210, 190, 170), -1);  // Face 2
    cv::circle(image, cv::Point(435, 185), 6, cv::Scalar(0, 0, 0), -1);         // Eye 1
    cv::circle(image, cv::Point(465, 185), 6, cv::Scalar(0, 0, 0), -1);         // Eye 2
    cv::ellipse(image, cv::Point(450, 210), cv::Size(15, 8), 0, 0, 180, cv::Scalar(0, 0, 0), 2); // Mouth
    
    return image;
}

void demonstrateFaceDetection(const cv::Mat& src) {
    std::cout << "\n=== Face Detection with Haar Cascades ===" << std::endl;
    
    // Try to load the face cascade
    cv::CascadeClassifier face_cascade;
    std::string face_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
    
    // Alternative paths for different OpenCV installations
    std::vector<std::string> cascade_paths = {
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt.xml",
        "../haarcascades/haarcascade_frontalface_alt.xml"
    };
    
    bool cascade_loaded = false;
    for (const auto& path : cascade_paths) {
        if (face_cascade.load(path)) {
            face_cascade_path = path;
            cascade_loaded = true;
            std::cout << "Loaded face cascade from: " << path << std::endl;
            break;
        }
    }
    
    if (!cascade_loaded) {
        std::cout << "Warning: Could not load face cascade classifier!" << std::endl;
        std::cout << "Please ensure OpenCV haarcascades are installed." << std::endl;
        std::cout << "Typical locations:" << std::endl;
        for (const auto& path : cascade_paths) {
            std::cout << "  - " << path << std::endl;
        }
        return;
    }
    
    // Convert to grayscale for detection
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);  // Improve contrast
    
    // Detect faces with different parameters
    std::vector<cv::Rect> faces;
    
    // Basic detection
    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
    
    // Create display image
    cv::Mat result = src.clone();
    
    // Draw detected faces
    for (size_t i = 0; i < faces.size(); i++) {
        cv::rectangle(result, faces[i], cv::Scalar(0, 255, 0), 2);
        cv::putText(result, "Face " + std::to_string(i + 1), 
                   cv::Point(faces[i].x, faces[i].y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    
    std::cout << "Detected " << faces.size() << " faces" << std::endl;
    
    // Test different scale factors
    cv::Mat comparison = cv::Mat::zeros(src.rows, src.cols * 3, CV_8UC3);
    src.copyTo(comparison(cv::Rect(0, 0, src.cols, src.rows)));
    
    // Detection with more sensitive parameters
    std::vector<cv::Rect> faces_sensitive;
    face_cascade.detectMultiScale(gray, faces_sensitive, 1.05, 2, 0, cv::Size(20, 20));
    
    cv::Mat sensitive_result = src.clone();
    for (const auto& face : faces_sensitive) {
        cv::rectangle(sensitive_result, face, cv::Scalar(255, 0, 0), 2);
    }
    sensitive_result.copyTo(comparison(cv::Rect(src.cols, 0, src.cols, src.rows)));
    
    // Detection with less sensitive parameters
    std::vector<cv::Rect> faces_conservative;
    face_cascade.detectMultiScale(gray, faces_conservative, 1.3, 5, 0, cv::Size(50, 50));
    
    cv::Mat conservative_result = src.clone();
    for (const auto& face : faces_conservative) {
        cv::rectangle(conservative_result, face, cv::Scalar(0, 0, 255), 2);
    }
    conservative_result.copyTo(comparison(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(comparison, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Sensitive", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Conservative", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Face Detection Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("Face Detection Comparison", comparison);
    
    // Display parameter effects
    std::cout << "Detection results with different parameters:" << std::endl;
    std::cout << "  - Standard (scale=1.1, neighbors=3): " << faces.size() << " faces" << std::endl;
    std::cout << "  - Sensitive (scale=1.05, neighbors=2): " << faces_sensitive.size() << " faces" << std::endl;
    std::cout << "  - Conservative (scale=1.3, neighbors=5): " << faces_conservative.size() << " faces" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateEyeDetection(const cv::Mat& src) {
    std::cout << "\n=== Eye Detection ===" << std::endl;
    
    // Load cascades
    cv::CascadeClassifier face_cascade, eye_cascade;
    
    std::vector<std::string> face_paths = {
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt.xml"
    };
    
    std::vector<std::string> eye_paths = {
        "/usr/share/opencv4/haarcascades/haarcascade_eye.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml",
        "haarcascade_eye.xml"
    };
    
    bool face_loaded = false, eye_loaded = false;
    
    for (const auto& path : face_paths) {
        if (face_cascade.load(path)) {
            face_loaded = true;
            break;
        }
    }
    
    for (const auto& path : eye_paths) {
        if (eye_cascade.load(path)) {
            eye_loaded = true;
            break;
        }
    }
    
    if (!face_loaded || !eye_loaded) {
        std::cout << "Could not load required cascade files for eye detection." << std::endl;
        std::cout << "Face cascade loaded: " << (face_loaded ? "Yes" : "No") << std::endl;
        std::cout << "Eye cascade loaded: " << (eye_loaded ? "Yes" : "No") << std::endl;
        return;
    }
    
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    
    // Detect faces first
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
    
    cv::Mat result = src.clone();
    
    // For each face, detect eyes
    for (const auto& face : faces) {
        cv::rectangle(result, face, cv::Scalar(0, 255, 0), 2);
        
        // Extract face region
        cv::Mat face_roi = gray(face);
        
        // Detect eyes in face region
        std::vector<cv::Rect> eyes;
        eye_cascade.detectMultiScale(face_roi, eyes, 1.1, 3, 0, cv::Size(5, 5));
        
        // Draw eyes (adjust coordinates to full image)
        for (const auto& eye : eyes) {
            cv::Rect eye_rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
            cv::rectangle(result, eye_rect, cv::Scalar(255, 0, 0), 2);
            cv::circle(result, cv::Point(eye_rect.x + eye_rect.width/2, eye_rect.y + eye_rect.height/2), 
                      3, cv::Scalar(0, 0, 255), -1);
        }
        
        std::cout << "Face at (" << face.x << ", " << face.y << ") has " << eyes.size() << " eyes detected" << std::endl;
    }
    
    cv::namedWindow("Face and Eye Detection", cv::WINDOW_AUTOSIZE);
    cv::imshow("Face and Eye Detection", result);
    
    std::cout << "Eye detection strategy:" << std::endl;
    std::cout << "  - First detect faces to reduce search space" << std::endl;
    std::cout << "  - Then search for eyes only within face regions" << std::endl;
    std::cout << "  - This improves accuracy and reduces false positives" << std::endl;
    std::cout << "  - Green rectangles: faces, Blue rectangles: eyes, Red dots: eye centers" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateMultiScaleDetection(const cv::Mat& src) {
    std::cout << "\n=== Multi-Scale Detection Analysis ===" << std::endl;
    
    cv::CascadeClassifier face_cascade;
    std::vector<std::string> cascade_paths = {
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt.xml"
    };
    
    bool loaded = false;
    for (const auto& path : cascade_paths) {
        if (face_cascade.load(path)) {
            loaded = true;
            break;
        }
    }
    
    if (!loaded) {
        std::cout << "Could not load face cascade for multi-scale demonstration." << std::endl;
        return;
    }
    
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    
    // Test different scale factors
    std::vector<double> scale_factors = {1.05, 1.1, 1.2, 1.3};
    std::vector<int> min_neighbors = {2, 3, 4, 5};
    
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    for (size_t i = 0; i < scale_factors.size(); i++) {
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, scale_factors[i], min_neighbors[i], 
                                     0, cv::Size(30, 30));
        
        cv::Mat result = src.clone();
        for (const auto& face : faces) {
            cv::rectangle(result, face, cv::Scalar(0, 255, 0), 2);
        }
        
        // Add parameter info
        std::string info = "Scale:" + std::to_string(scale_factors[i]) + " Neighbors:" + std::to_string(min_neighbors[i]);
        cv::putText(result, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(result, "Faces:" + std::to_string(faces.size()), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        // Place in grid
        int row = i / 2;
        int col = i % 2;
        result.copyTo(display(cv::Rect(col * src.cols, row * src.rows, src.cols, src.rows)));
        
        std::cout << "Scale " << scale_factors[i] << ", Neighbors " << min_neighbors[i] 
                  << ": " << faces.size() << " faces detected" << std::endl;
    }
    
    cv::namedWindow("Multi-Scale Detection", cv::WINDOW_AUTOSIZE);
    cv::imshow("Multi-Scale Detection", display);
    
    std::cout << "\nMulti-scale detection parameters:" << std::endl;
    std::cout << "  - Scale factor: How much the image size is reduced at each scale" << std::endl;
    std::cout << "    * Lower values (1.05): More scales, slower but more thorough" << std::endl;
    std::cout << "    * Higher values (1.3): Fewer scales, faster but may miss objects" << std::endl;
    std::cout << "  - Min neighbors: Minimum number of neighbor detections for each candidate" << std::endl;
    std::cout << "    * Lower values: More detections but more false positives" << std::endl;
    std::cout << "    * Higher values: Fewer false positives but may miss true objects" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateDetectionConfidence(const cv::Mat& src) {
    std::cout << "\n=== Detection Confidence and Filtering ===" << std::endl;
    
    cv::CascadeClassifier face_cascade;
    std::vector<std::string> cascade_paths = {
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt.xml"
    };
    
    bool loaded = false;
    for (const auto& path : cascade_paths) {
        if (face_cascade.load(path)) {
            loaded = true;
            break;
        }
    }
    
    if (!loaded) {
        std::cout << "Could not load face cascade." << std::endl;
        return;
    }
    
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    
    // Detect with detailed output
    std::vector<cv::Rect> faces;
    std::vector<int> reject_levels;
    std::vector<double> level_weights;
    
    face_cascade.detectMultiScale(gray, faces, reject_levels, level_weights, 
                                 1.1, 3, 0, cv::Size(30, 30), cv::Size(), true);
    
    cv::Mat result = src.clone();
    
    // Draw faces with confidence-based coloring
    for (size_t i = 0; i < faces.size(); i++) {
        // Use level weights as confidence measure
        double confidence = (i < level_weights.size()) ? level_weights[i] : 1.0;
        
        // Color based on confidence: green for high, yellow for medium, red for low
        cv::Scalar color;
        if (confidence > 2.0) {
            color = cv::Scalar(0, 255, 0);  // Green - high confidence
        } else if (confidence > 1.0) {
            color = cv::Scalar(0, 255, 255);  // Yellow - medium confidence
        } else {
            color = cv::Scalar(0, 0, 255);   // Red - low confidence
        }
        
        cv::rectangle(result, faces[i], color, 2);
        
        // Add confidence text
        std::string conf_text = "C:" + std::to_string(confidence).substr(0, 4);
        cv::putText(result, conf_text, 
                   cv::Point(faces[i].x, faces[i].y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        
        std::cout << "Face " << i + 1 << ": confidence=" << confidence 
                  << ", reject_level=" << (i < reject_levels.size() ? reject_levels[i] : -1) << std::endl;
    }
    
    cv::namedWindow("Detection Confidence", cv::WINDOW_AUTOSIZE);
    cv::imshow("Detection Confidence", result);
    
    std::cout << "\nDetection confidence interpretation:" << std::endl;
    std::cout << "  - Green rectangles: High confidence detections" << std::endl;
    std::cout << "  - Yellow rectangles: Medium confidence detections" << std::endl;
    std::cout << "  - Red rectangles: Low confidence detections" << std::endl;
    std::cout << "  - Level weights indicate detection strength" << std::endl;
    std::cout << "  - Reject levels show at which stage detection was made" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Cascade Classifiers ===" << std::endl;
    
    // Try to load a real image
    cv::Mat image = cv::imread("../data/faces.jpg");
    
    if (image.empty()) {
        // Try alternative paths
        std::vector<std::string> image_paths = {
            "../data/test.jpg",
            "test.jpg",
            "faces.jpg"
        };
        
        for (const auto& path : image_paths) {
            image = cv::imread(path);
            if (!image.empty()) {
                std::cout << "Loaded image from: " << path << std::endl;
                break;
            }
        }
    }
    
    if (image.empty()) {
        std::cout << "No real image found, using synthetic test image." << std::endl;
        image = createTestFaceImage();
    }
    
    // Resize if image is too large
    if (image.cols > 800 || image.rows > 600) {
        cv::resize(image, image, cv::Size(800, 600));
    }
    
    // Demonstrate different cascade techniques
    demonstrateFaceDetection(image);
    demonstrateEyeDetection(image);
    demonstrateMultiScaleDetection(image);
    demonstrateDetectionConfidence(image);
    
    std::cout << "\n✓ Cascade Classifiers demonstration complete!" << std::endl;
    std::cout << "Cascade classifiers provide fast object detection using trained features." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Haar cascades use rectangular features for fast object detection
 * 2. LBP cascades use local binary patterns (more robust to lighting)
 * 3. Multi-scale detection searches at different image scales
 * 4. Scale factor controls search thoroughness vs speed
 * 5. Min neighbors parameter filters false positives
 * 6. Face detection works best on frontal, upright faces
 * 7. Region of interest (ROI) processing improves accuracy
 * 8. Confidence measures help filter detections
 * 9. Histogram equalization improves detection under varying lighting
 * 10. Cascade files must be available in system or specified paths
 */

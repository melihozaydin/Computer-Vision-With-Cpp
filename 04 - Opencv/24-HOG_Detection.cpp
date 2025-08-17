/**
 * 24-HOG_Detection.cpp
 * 
 * Histogram of Oriented Gradients for object detection.
 * 
 * Concepts covered:
 * - HOG feature extraction
 * - SVM classification
 * - Pedestrian detection
 * - Multi-scale detection
 * - HOG parameter tuning
 * - Custom HOG descriptors
 */

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

cv::Mat createPedestrianTestImage() {
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    
    // Create simple pedestrian-like shapes for testing
    // Pedestrian 1 - standing figure
    cv::rectangle(image, cv::Point(100, 150), cv::Point(140, 350), cv::Scalar(100, 100, 150), -1);  // Body
    cv::circle(image, cv::Point(120, 130), 20, cv::Scalar(200, 180, 160), -1);  // Head
    cv::rectangle(image, cv::Point(90, 200), cv::Point(110, 280), cv::Scalar(80, 80, 120), -1);   // Left arm
    cv::rectangle(image, cv::Point(130, 200), cv::Point(150, 280), cv::Scalar(80, 80, 120), -1);  // Right arm
    cv::rectangle(image, cv::Point(105, 350), cv::Point(120, 420), cv::Scalar(60, 60, 100), -1);  // Left leg
    cv::rectangle(image, cv::Point(120, 350), cv::Point(135, 420), cv::Scalar(60, 60, 100), -1);  // Right leg
    
    // Pedestrian 2 - smaller figure
    cv::rectangle(image, cv::Point(300, 200), cv::Point(330, 350), cv::Scalar(120, 120, 180), -1); // Body
    cv::circle(image, cv::Point(315, 185), 15, cv::Scalar(210, 190, 170), -1);  // Head
    cv::rectangle(image, cv::Point(295, 230), cv::Point(310, 300), cv::Scalar(100, 100, 150), -1); // Left arm
    cv::rectangle(image, cv::Point(320, 230), cv::Point(335, 300), cv::Scalar(100, 100, 150), -1); // Right arm
    cv::rectangle(image, cv::Point(305, 350), cv::Point(315, 400), cv::Scalar(80, 80, 130), -1);   // Left leg
    cv::rectangle(image, cv::Point(315, 350), cv::Point(325, 400), cv::Scalar(80, 80, 130), -1);   // Right leg
    
    // Add some background elements
    cv::rectangle(image, cv::Point(450, 100), cv::Point(600, 400), cv::Scalar(80, 120, 80), -1);   // Building
    cv::rectangle(image, cv::Point(0, 400), cv::Point(640, 480), cv::Scalar(60, 60, 60), -1);      // Ground
    
    // Add noise to make it more realistic
    cv::Mat noise(image.size(), CV_8UC3);
    cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(30));
    cv::add(image, noise, image);
    
    return image;
}

void demonstrateHOGFeatures(const cv::Mat& src) {
    std::cout << "\n=== HOG Feature Extraction ===" << std::endl;
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Create HOG descriptor with different parameters
    cv::HOGDescriptor hog;
    
    // Display HOG parameters
    std::cout << "Default HOG parameters:" << std::endl;
    std::cout << "  - Window size: " << hog.winSize.width << "x" << hog.winSize.height << std::endl;
    std::cout << "  - Block size: " << hog.blockSize.width << "x" << hog.blockSize.height << std::endl;
    std::cout << "  - Block stride: " << hog.blockStride.width << "x" << hog.blockStride.height << std::endl;
    std::cout << "  - Cell size: " << hog.cellSize.width << "x" << hog.cellSize.height << std::endl;
    std::cout << "  - Number of bins: " << hog.nbins << std::endl;
    
    // Calculate HOG features for the entire image
    std::vector<float> descriptors;
    std::vector<cv::Point> locations;
    
    // Resize image to fit HOG window size for feature extraction
    cv::Mat resized;
    cv::resize(gray, resized, hog.winSize);
    
    hog.compute(resized, descriptors, cv::Size(8, 8), cv::Size(0, 0), locations);
    
    std::cout << "HOG features computed:" << std::endl;
    std::cout << "  - Descriptor size: " << descriptors.size() << std::endl;
    std::cout << "  - Feature locations: " << locations.size() << std::endl;
    
    // Visualize gradients
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    
    cv::Mat magnitude, angle;
    cv::cartToPolar(grad_x, grad_y, magnitude, angle, true);
    
    // Normalize for visualization
    cv::Mat mag_vis, angle_vis;
    cv::normalize(magnitude, mag_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(angle, angle_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    // Create visualization
    cv::Mat display = cv::Mat::zeros(gray.rows, gray.cols * 3, CV_8UC3);
    
    cv::cvtColor(gray, display(cv::Rect(0, 0, gray.cols, gray.rows)), cv::COLOR_GRAY2BGR);
    cv::cvtColor(mag_vis, display(cv::Rect(gray.cols, 0, gray.cols, gray.rows)), cv::COLOR_GRAY2BGR);
    cv::cvtColor(angle_vis, display(cv::Rect(gray.cols * 2, 0, gray.cols, gray.rows)), cv::COLOR_GRAY2BGR);
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Gradient Magnitude", cv::Point(gray.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Gradient Direction", cv::Point(gray.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("HOG Features", cv::WINDOW_AUTOSIZE);
    cv::imshow("HOG Features", display);
    
    std::cout << "HOG feature extraction process:" << std::endl;
    std::cout << "  1. Compute gradients in X and Y directions" << std::endl;
    std::cout << "  2. Calculate magnitude and orientation" << std::endl;
    std::cout << "  3. Create histogram of orientations in cells" << std::endl;
    std::cout << "  4. Normalize over blocks for illumination invariance" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstratePedestrianDetection(const cv::Mat& src) {
    std::cout << "\n=== Pedestrian Detection with HOG ===" << std::endl;
    
    // Create HOG descriptor for pedestrian detection
    cv::HOGDescriptor hog;
    
    // Set the default people detector (trained SVM)
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    
    std::cout << "Using default people detector SVM model" << std::endl;
    std::cout << "SVM detector size: " << hog.getDefaultPeopleDetector().size() << std::endl;
    
    // Detect pedestrians
    std::vector<cv::Rect> found, found_filtered;
    
    // Basic detection
    hog.detectMultiScale(src, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
    
    std::cout << "Initial detections: " << found.size() << std::endl;
    
    // Filter overlapping detections
    for (size_t i = 0; i < found.size(); i++) {
        cv::Rect r = found[i];
        
        bool inside = false;
        for (size_t j = 0; j < found.size(); j++) {
            if (j != i && (r & found[j]) == r) {
                inside = true;
                break;
            }
        }
        if (!inside) {
            found_filtered.push_back(r);
        }
    }
    
    std::cout << "Filtered detections: " << found_filtered.size() << std::endl;
    
    // Draw detections
    cv::Mat result = src.clone();
    
    // Draw all initial detections in yellow
    for (const auto& detection : found) {
        cv::rectangle(result, detection, cv::Scalar(0, 255, 255), 2);
    }
    
    // Draw filtered detections in green
    for (const auto& detection : found_filtered) {
        cv::rectangle(result, detection, cv::Scalar(0, 255, 0), 3);
        cv::putText(result, "Person", 
                   cv::Point(detection.x, detection.y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    
    cv::namedWindow("Pedestrian Detection", cv::WINDOW_AUTOSIZE);
    cv::imshow("Pedestrian Detection", result);
    
    std::cout << "Detection visualization:" << std::endl;
    std::cout << "  - Yellow rectangles: All initial detections" << std::endl;
    std::cout << "  - Green rectangles: Filtered detections (non-overlapping)" << std::endl;
    std::cout << "  - Detection filtering removes redundant overlapping detections" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateHOGParameterTuning(const cv::Mat& src) {
    std::cout << "\n=== HOG Parameter Tuning ===" << std::endl;
    
    // Test different HOG parameters
    struct HOGParams {
        cv::Size winSize;
        cv::Size blockSize;
        cv::Size blockStride;
        cv::Size cellSize;
        int nbins;
        std::string name;
    };
    
    std::vector<HOGParams> param_sets = {
        {{64, 128}, {16, 16}, {8, 8}, {8, 8}, 9, "Default"},
        {{48, 96}, {12, 12}, {6, 6}, {6, 6}, 9, "Smaller"},
        {{80, 160}, {20, 20}, {10, 10}, {10, 10}, 9, "Larger"},
        {{64, 128}, {16, 16}, {8, 8}, {8, 8}, 18, "More Bins"}
    };
    
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    for (size_t i = 0; i < param_sets.size(); i++) {
        const auto& params = param_sets[i];
        
        // Create HOG with custom parameters
        cv::HOGDescriptor hog(params.winSize, params.blockSize, params.blockStride, 
                             params.cellSize, params.nbins);
        
        // Set SVM detector
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        
        // Detect pedestrians
        std::vector<cv::Rect> detections;
        try {
            hog.detectMultiScale(src, detections, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
        } catch (const cv::Exception& e) {
            std::cout << "Detection failed for " << params.name << ": " << e.what() << std::endl;
            continue;
        }
        
        // Draw results
        cv::Mat result = src.clone();
        for (const auto& detection : detections) {
            cv::rectangle(result, detection, cv::Scalar(0, 255, 0), 2);
        }
        
        // Add parameter info
        cv::putText(result, params.name, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(result, "Detections: " + std::to_string(detections.size()), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        // Place in grid
        int row = i / 2;
        int col = i % 2;
        if (row < 2 && col < 2) {
            result.copyTo(display(cv::Rect(col * src.cols, row * src.rows, src.cols, src.rows)));
        }
        
        std::cout << params.name << " parameters - Detections: " << detections.size() << std::endl;
        std::cout << "  Window: " << params.winSize.width << "x" << params.winSize.height;
        std::cout << ", Block: " << params.blockSize.width << "x" << params.blockSize.height;
        std::cout << ", Cell: " << params.cellSize.width << "x" << params.cellSize.height;
        std::cout << ", Bins: " << params.nbins << std::endl;
    }
    
    cv::namedWindow("HOG Parameter Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("HOG Parameter Comparison", display);
    
    std::cout << "\nParameter effects:" << std::endl;
    std::cout << "  - Window size: Detection template size (must match training)" << std::endl;
    std::cout << "  - Block size: Normalization region size" << std::endl;
    std::cout << "  - Cell size: Histogram computation region" << std::endl;
    std::cout << "  - Number of bins: Orientation discretization" << std::endl;
    std::cout << "  - Smaller parameters = higher resolution, more computation" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateCustomHOGTraining() {
    std::cout << "\n=== Custom HOG Training Concepts ===" << std::endl;
    
    // This demonstrates the concepts of HOG training without actual training
    // (which would require large datasets and significant computation time)
    
    std::cout << "Custom HOG Training Process:" << std::endl;
    std::cout << "1. Data Collection:" << std::endl;
    std::cout << "   - Positive samples: Images containing the target object" << std::endl;
    std::cout << "   - Negative samples: Images without the target object" << std::endl;
    std::cout << "   - Typically need thousands of samples" << std::endl;
    
    std::cout << "\n2. Feature Extraction:" << std::endl;
    std::cout << "   - Extract HOG features from all samples" << std::endl;
    std::cout << "   - Normalize feature vectors" << std::endl;
    std::cout << "   - Create training matrix" << std::endl;
    
    std::cout << "\n3. SVM Training:" << std::endl;
    std::cout << "   - Train Support Vector Machine classifier" << std::endl;
    std::cout << "   - Tune hyperparameters (C, gamma for RBF kernel)" << std::endl;
    std::cout << "   - Cross-validation for parameter selection" << std::endl;
    
    std::cout << "\n4. Hard Negative Mining:" << std::endl;
    std::cout << "   - Run detector on negative images" << std::endl;
    std::cout << "   - Collect false positive detections" << std::endl;
    std::cout << "   - Retrain SVM with these hard negatives" << std::endl;
    std::cout << "   - Iterate to improve performance" << std::endl;
    
    std::cout << "\n5. Evaluation:" << std::endl;
    std::cout << "   - Test on independent validation set" << std::endl;
    std::cout << "   - Measure precision, recall, F1-score" << std::endl;
    std::cout << "   - Analyze false positives and false negatives" << std::endl;
    
    // Example of how to set up training data structure
    std::cout << "\nExample training setup:" << std::endl;
    
    cv::HOGDescriptor hog;
    std::vector<float> features;
    std::vector<int> labels;
    
    std::cout << "  - HOG descriptor configured" << std::endl;
    std::cout << "  - Feature vector size per sample: " << hog.getDescriptorSize() << std::endl;
    std::cout << "  - Training would involve:" << std::endl;
    std::cout << "    * Loading positive/negative samples" << std::endl;
    std::cout << "    * Computing HOG features for each" << std::endl;
    std::cout << "    * Training SVM classifier" << std::endl;
    std::cout << "    * Validating performance" << std::endl;
    
    std::cout << "\nFor actual training, consider using:" << std::endl;
    std::cout << "  - OpenCV's ml module for SVM training" << std::endl;
    std::cout << "  - Large annotated datasets (INRIA Person, etc.)" << std::endl;
    std::cout << "  - Dedicated training frameworks" << std::endl;
}

int main() {
    std::cout << "=== HOG Detection ===" << std::endl;
    
    // Try to load a real image
    cv::Mat image = cv::imread("../data/pedestrians.jpg");
    
    if (image.empty()) {
        // Try alternative paths
        std::vector<std::string> image_paths = {
            "../data/test.jpg",
            "test.jpg",
            "pedestrians.jpg"
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
        image = createPedestrianTestImage();
    }
    
    // Resize if image is too large
    if (image.cols > 800 || image.rows > 600) {
        cv::resize(image, image, cv::Size(800, 600));
    }
    
    // Demonstrate HOG techniques
    demonstrateHOGFeatures(image);
    demonstratePedestrianDetection(image);
    demonstrateHOGParameterTuning(image);
    demonstrateCustomHOGTraining();
    
    std::cout << "\n✓ HOG Detection demonstration complete!" << std::endl;
    std::cout << "HOG provides robust object detection through gradient-based features." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. HOG extracts gradient-based features robust to illumination changes
 * 2. Features are computed in overlapping blocks for normalization
 * 3. SVM classifier trained on HOG features enables object detection
 * 4. Multi-scale detection handles objects at different sizes
 * 5. Non-maximum suppression filters overlapping detections
 * 6. Parameter tuning affects detection accuracy and speed
 * 7. Default people detector works well for pedestrian detection
 * 8. Custom training requires large labeled datasets
 * 9. Hard negative mining improves classifier performance
 * 10. Window size must match training data dimensions
 */

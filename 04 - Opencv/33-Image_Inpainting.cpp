/**
 * 33-Image_Inpainting.cpp
 * 
 * Image restoration and hole filling techniques.
 * 
 * Concepts covered:
 * - Telea inpainting algorithm
 * - Navier-Stokes inpainting
 * - Mask-based inpainting
 * - Texture synthesis
 * - Photo restoration
 */

#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <random>

cv::Mat createTestImageWithDamage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // Create a complex scene to inpaint
    // Background gradient
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int intensity = 100 + (x * 155) / image.cols;
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity - 50, intensity + 50);
        }
    }
    
    // Add geometric shapes
    cv::circle(image, cv::Point(150, 150), 60, cv::Scalar(0, 255, 0), -1);
    cv::rectangle(image, cv::Point(300, 80), cv::Point(450, 220), cv::Scalar(255, 0, 0), -1);
    cv::ellipse(image, cv::Point(450, 300), cv::Size(80, 40), 45, 0, 360, cv::Scalar(0, 0, 255), -1);
    
    // Add texture pattern
    for (int y = 250; y < 350; y++) {
        for (int x = 50; x < 250; x++) {
            if ((x + y) % 20 < 10) {
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 0);
            }
        }
    }
    
    // Add text
    cv::putText(image, "INPAINT", cv::Point(100, 350), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 3);
    
    return image;
}

cv::Mat createDamageMask(const cv::Mat& image) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    
    // Create various types of damage
    
    // Scratches (thin lines)
    cv::line(mask, cv::Point(100, 50), cv::Point(500, 200), cv::Scalar(255), 8);
    cv::line(mask, cv::Point(200, 250), cv::Point(400, 350), cv::Scalar(255), 6);
    cv::line(mask, cv::Point(50, 300), cv::Point(250, 280), cv::Scalar(255), 4);
    
    // Holes (circular regions)
    cv::circle(mask, cv::Point(300, 150), 25, cv::Scalar(255), -1);
    cv::circle(mask, cv::Point(450, 250), 20, cv::Scalar(255), -1);
    cv::circle(mask, cv::Point(150, 250), 15, cv::Scalar(255), -1);
    
    // Irregular damage areas
    std::vector<cv::Point> damage_contour = {
        cv::Point(400, 80), cv::Point(450, 90), cv::Point(480, 120),
        cv::Point(460, 150), cv::Point(430, 140), cv::Point(410, 110)
    };
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{damage_contour}, cv::Scalar(255));
    
    // Text occlusion
    cv::rectangle(mask, cv::Point(120, 320), cv::Point(200, 360), cv::Scalar(255), -1);
    
    // Speckle noise damage
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> x_dist(0, image.cols - 1);
    std::uniform_int_distribution<> y_dist(0, image.rows - 1);
    std::uniform_int_distribution<> size_dist(2, 8);
    
    for (int i = 0; i < 30; i++) {
        cv::circle(mask, cv::Point(x_dist(gen), y_dist(gen)), size_dist(gen), cv::Scalar(255), -1);
    }
    
    return mask;
}

void demonstrateBasicInpainting(const cv::Mat& damaged_image, const cv::Mat& mask) {
    std::cout << "\n=== Basic Inpainting Algorithms ===" << std::endl;
    
    // Telea algorithm
    cv::Mat telea_result;
    cv::inpaint(damaged_image, mask, telea_result, 3, cv::INPAINT_TELEA);
    
    // Navier-Stokes algorithm
    cv::Mat ns_result;
    cv::inpaint(damaged_image, mask, ns_result, 3, cv::INPAINT_NS);
    
    // Create comparison display
    cv::Mat top_row, bottom_row, display;
    
    // Convert mask to color for display consistency
    cv::Mat mask_colored;
    cv::cvtColor(mask, mask_colored, cv::COLOR_GRAY2BGR);
    
    cv::hconcat(damaged_image, telea_result, top_row);
    cv::hconcat(ns_result, mask_colored, bottom_row);
    
    cv::vconcat(top_row, bottom_row, display);
    
    // Add labels
    cv::putText(display, "Damaged", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Telea", cv::Point(damaged_image.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Navier-Stokes", cv::Point(10, damaged_image.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Mask", cv::Point(damaged_image.cols + 10, damaged_image.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::namedWindow("Basic Inpainting", cv::WINDOW_AUTOSIZE);
    cv::imshow("Basic Inpainting", display);
    
    std::cout << "Inpainting algorithm comparison:" << std::endl;
    std::cout << "  - Telea: Fast Marching Method based" << std::endl;
    std::cout << "  - Navier-Stokes: Fluid dynamics based" << std::endl;
    std::cout << "  - Both work well for thin scratches" << std::endl;
    std::cout << "  - NS better for larger regions" << std::endl;
    std::cout << "  - Telea faster, NS more accurate" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateInpaintingRadius(const cv::Mat& damaged_image, const cv::Mat& mask) {
    std::cout << "\n=== Inpainting Radius Effects ===" << std::endl;
    
    std::vector<int> radii = {1, 3, 7, 15};
    cv::Mat display = cv::Mat::zeros(damaged_image.rows * 2, damaged_image.cols * 2, CV_8UC3);
    
    for (size_t i = 0; i < radii.size(); i++) {
        cv::Mat result;
        cv::inpaint(damaged_image, mask, result, radii[i], cv::INPAINT_TELEA);
        
        // Place in grid
        int row = i / 2;
        int col = i % 2;
        cv::Rect roi(col * damaged_image.cols, row * damaged_image.rows, 
                     damaged_image.cols, damaged_image.rows);
        result.copyTo(display(roi));
        
        // Add label
        std::string label = "Radius: " + std::to_string(radii[i]);
        cv::putText(display, label, 
                   cv::Point(col * damaged_image.cols + 10, row * damaged_image.rows + 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    
    cv::namedWindow("Inpainting Radius Effects", cv::WINDOW_AUTOSIZE);
    cv::imshow("Inpainting Radius Effects", display);
    
    std::cout << "Inpainting radius effects:" << std::endl;
    std::cout << "  - Small radius: Local information, preserves detail" << std::endl;
    std::cout << "  - Large radius: Global information, smoother result" << std::endl;
    std::cout << "  - Choose based on damage size and desired quality" << std::endl;
    std::cout << "  - Larger radius = longer computation time" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateInteractiveInpainting(const cv::Mat& original_image) {
    std::cout << "\n=== Interactive Inpainting Demo ===" << std::endl;
    
    cv::Mat current_image = original_image.clone();
    cv::Mat mask = cv::Mat::zeros(original_image.size(), CV_8UC1);
    cv::Mat display;
    
    bool drawing = false;
    int brush_size = 10;
    
    // Mouse callback for interactive mask creation
    auto mouse_callback = [](int event, int x, int y, int, void* userdata) {
        auto* data = static_cast<std::tuple<cv::Mat&, cv::Mat&, bool&, int&>*>(userdata);
        cv::Mat& mask = std::get<0>(*data);
        //cv::Mat& image = std::get<1>(*data);  // Available for display updates if needed
        bool& drawing = std::get<2>(*data);
        int& brush_size = std::get<3>(*data);
        
        if (event == cv::EVENT_LBUTTONDOWN) {
            drawing = true;
            cv::circle(mask, cv::Point(x, y), brush_size, cv::Scalar(255), -1);
        } else if (event == cv::EVENT_MOUSEMOVE && drawing) {
            cv::circle(mask, cv::Point(x, y), brush_size, cv::Scalar(255), -1);
        } else if (event == cv::EVENT_LBUTTONUP) {
            drawing = false;
        }
    };
    
    auto callback_data = std::make_tuple(std::ref(mask), std::ref(current_image), 
                                       std::ref(drawing), std::ref(brush_size));
    
    cv::namedWindow("Interactive Inpainting", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Interactive Inpainting", mouse_callback, &callback_data);
    
    std::cout << "Interactive inpainting controls:" << std::endl;
    std::cout << "  - Left click and drag to mark areas for inpainting" << std::endl;
    std::cout << "  - Press 'i' to perform inpainting" << std::endl;
    std::cout << "  - Press 'r' to reset" << std::endl;
    std::cout << "  - Press '+'/'-' to change brush size" << std::endl;
    std::cout << "  - Press 'q' to quit" << std::endl;
    
    while (true) {
        // Create display with mask overlay
        cv::Mat mask_colored;
        cv::cvtColor(mask, mask_colored, cv::COLOR_GRAY2BGR);
        cv::addWeighted(current_image, 0.7, mask_colored, 0.3, 0, display);
        
        // Add brush size indicator
        cv::putText(display, "Brush: " + std::to_string(brush_size), 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(display, "Press 'i' to inpaint", 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Interactive Inpainting", display);
        
        char key = cv::waitKey(30);
        if (key == 'q' || key == 27) break;  // Quit
        
        if (key == 'i') {  // Inpaint
            if (cv::countNonZero(mask) > 0) {
                cv::Mat inpainted;
                cv::inpaint(current_image, mask, inpainted, 5, cv::INPAINT_TELEA);
                current_image = inpainted;
                mask = cv::Mat::zeros(original_image.size(), CV_8UC1);
                std::cout << "Inpainting applied!" << std::endl;
            }
        } else if (key == 'r') {  // Reset
            current_image = original_image.clone();
            mask = cv::Mat::zeros(original_image.size(), CV_8UC1);
            std::cout << "Reset to original image." << std::endl;
        } else if (key == '+' || key == '=') {  // Increase brush size
            brush_size = std::min(50, brush_size + 2);
        } else if (key == '-') {  // Decrease brush size
            brush_size = std::max(2, brush_size - 2);
        }
    }
    
    cv::destroyAllWindows();
}

void demonstrateTextureInpainting(const cv::Mat&) {
    std::cout << "\n=== Texture-Aware Inpainting ===" << std::endl;
    
    // Create a textured test image
    cv::Mat textured_image = cv::Mat::zeros(300, 400, CV_8UC3);
    
    // Create brick pattern
    for (int y = 0; y < textured_image.rows; y += 40) {
        for (int x = 0; x < textured_image.cols; x += 80) {
            // Brick color variation
            cv::Scalar brick_color(120 + (x + y) % 40, 80 + (x + y) % 30, 60 + (x + y) % 20);
            
            // Draw brick
            cv::rectangle(textured_image, cv::Point(x, y), cv::Point(x + 75, y + 35), brick_color, -1);
            
            // Mortar lines
            cv::rectangle(textured_image, cv::Point(x, y), cv::Point(x + 75, y + 35), cv::Scalar(200, 200, 200), 2);
        }
    }
    
    // Create damage that breaks texture pattern
    cv::Mat texture_mask = cv::Mat::zeros(textured_image.size(), CV_8UC1);
    
    // Large hole that spans multiple bricks
    cv::circle(texture_mask, cv::Point(200, 150), 40, cv::Scalar(255), -1);
    
    // Crack across bricks
    cv::line(texture_mask, cv::Point(50, 50), cv::Point(350, 250), cv::Scalar(255), 12);
    
    // Apply damage to image
    cv::Mat damaged_textured = textured_image.clone();
    damaged_textured.setTo(cv::Scalar(0, 0, 0), texture_mask);
    
    // Inpaint with different methods
    cv::Mat telea_textured, ns_textured;
    cv::inpaint(damaged_textured, texture_mask, telea_textured, 5, cv::INPAINT_TELEA);
    cv::inpaint(damaged_textured, texture_mask, ns_textured, 5, cv::INPAINT_NS);
    
    // Create display
    cv::Mat top_row, bottom_row, display;
    cv::hconcat(textured_image, damaged_textured, top_row);
    cv::hconcat(telea_textured, ns_textured, bottom_row);
    cv::vconcat(top_row, bottom_row, display);
    
    // Add labels
    cv::putText(display, "Original Texture", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Damaged", cv::Point(textured_image.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "Telea Result", cv::Point(10, textured_image.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "NS Result", cv::Point(textured_image.cols + 10, textured_image.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    
    cv::namedWindow("Texture Inpainting", cv::WINDOW_AUTOSIZE);
    cv::imshow("Texture Inpainting", display);
    
    std::cout << "Texture inpainting challenges:" << std::endl;
    std::cout << "  - Regular patterns require structure preservation" << std::endl;
    std::cout << "  - Large holes may break pattern continuity" << std::endl;
    std::cout << "  - NS method better for preserving texture structure" << std::endl;
    std::cout << "  - Advanced methods use texture synthesis" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstratePhotoRestoration() {
    std::cout << "\n=== Photo Restoration Workflow ===" << std::endl;
    
    // Simulate an old damaged photo
    cv::Mat vintage_photo = cv::Mat::zeros(300, 400, CV_8UC3);
    
    // Create a vintage scene
    cv::rectangle(vintage_photo, cv::Point(50, 50), cv::Point(350, 250), cv::Scalar(150, 140, 120), -1);
    cv::rectangle(vintage_photo, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(80, 70, 60), -1);
    cv::circle(vintage_photo, cv::Point(300, 100), 30, cv::Scalar(200, 180, 160), -1);
    cv::putText(vintage_photo, "1945", cv::Point(150, 240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(60, 50, 40), 2);
    
    // Add aging effects
    cv::Mat sepia_photo;
    cv::Mat kernel = (cv::Mat_<float>(4, 4) <<
                      0.272, 0.534, 0.131, 0,
                      0.349, 0.686, 0.168, 0,
                      0.393, 0.769, 0.189, 0,
                      0, 0, 0, 1);
    cv::transform(vintage_photo, sepia_photo, kernel);
    
    // Add damage typical of old photos
    cv::Mat restoration_mask = cv::Mat::zeros(sepia_photo.size(), CV_8UC1);
    
    // Cracks and tears
    cv::line(restoration_mask, cv::Point(0, 80), cv::Point(400, 120), cv::Scalar(255), 3);
    cv::line(restoration_mask, cv::Point(200, 0), cv::Point(220, 300), cv::Scalar(255), 2);
    
    // Stains and spots
    cv::circle(restoration_mask, cv::Point(80, 200), 15, cv::Scalar(255), -1);
    cv::circle(restoration_mask, cv::Point(320, 80), 12, cv::Scalar(255), -1);
    cv::ellipse(restoration_mask, cv::Point(250, 200), cv::Size(25, 15), 30, 0, 360, cv::Scalar(255), -1);
    
    // Missing corner
    std::vector<cv::Point> corner_damage = {
        cv::Point(0, 0), cv::Point(60, 0), cv::Point(0, 60)
    };
    cv::fillPoly(restoration_mask, std::vector<std::vector<cv::Point>>{corner_damage}, cv::Scalar(255));
    
    // Apply damage
    cv::Mat damaged_photo = sepia_photo.clone();
    damaged_photo.setTo(cv::Scalar(0, 0, 0), restoration_mask);
    
    // Restoration steps
    
    // Step 1: Basic inpainting
    cv::Mat basic_restoration;
    cv::inpaint(damaged_photo, restoration_mask, basic_restoration, 7, cv::INPAINT_NS);
    
    // Step 2: Denoising (simulated)
    cv::Mat denoised_restoration;
    cv::bilateralFilter(basic_restoration, denoised_restoration, 9, 75, 75);
    
    // Step 3: Contrast enhancement
    cv::Mat enhanced_restoration;
    denoised_restoration.convertTo(enhanced_restoration, -1, 1.2, 20);  // Increase contrast and brightness
    
    // Create restoration workflow display
    cv::Mat step1, step2, display;
    cv::hconcat(vintage_photo, damaged_photo, step1);
    cv::hconcat(basic_restoration, enhanced_restoration, step2);
    cv::vconcat(step1, step2, display);
    
    // Add workflow labels
    cv::putText(display, "1. Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "2. Damaged", cv::Point(vintage_photo.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "3. Inpainted", cv::Point(10, vintage_photo.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "4. Enhanced", cv::Point(vintage_photo.cols + 10, vintage_photo.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    
    cv::namedWindow("Photo Restoration Workflow", cv::WINDOW_AUTOSIZE);
    cv::imshow("Photo Restoration Workflow", display);
    
    std::cout << "Photo restoration workflow:" << std::endl;
    std::cout << "  1. Damage assessment and mask creation" << std::endl;
    std::cout << "  2. Inpainting for structural restoration" << std::endl;
    std::cout << "  3. Denoising to reduce grain and artifacts" << std::endl;
    std::cout << "  4. Contrast and color enhancement" << std::endl;
    std::cout << "  5. Optional: Color correction and sharpening" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Image Inpainting Demonstration ===" << std::endl;
    
    // Try to load a real damaged image
    cv::Mat real_image = cv::imread("../data/damaged_photo.jpg");
    cv::Mat real_mask = cv::imread("../data/damage_mask.jpg", cv::IMREAD_GRAYSCALE);
    
    cv::Mat test_image;
    cv::Mat test_mask;
    
    if (!real_image.empty() && !real_mask.empty()) {
        std::cout << "Using real damaged image and mask." << std::endl;
        test_image = real_image;
        test_mask = real_mask;
        
        // Resize if necessary
        if (test_image.cols > 800 || test_image.rows > 600) {
            cv::Size new_size(800, 600);
            cv::resize(test_image, test_image, new_size);
            cv::resize(test_mask, test_mask, new_size);
        }
    } else {
        std::cout << "Creating synthetic test image and damage mask." << std::endl;
        test_image = createTestImageWithDamage();
        test_mask = createDamageMask(test_image);
    }
    
    // Apply damage to create damaged version
    cv::Mat damaged_image = test_image.clone();
    damaged_image.setTo(cv::Scalar(0, 0, 0), test_mask);
    
    // Demonstrate inpainting techniques
    demonstrateBasicInpainting(damaged_image, test_mask);
    demonstrateInpaintingRadius(damaged_image, test_mask);
    demonstrateInteractiveInpainting(test_image);
    demonstrateTextureInpainting(test_image);
    demonstratePhotoRestoration();
    
    std::cout << "\n✓ Image Inpainting demonstration complete!" << std::endl;
    std::cout << "Inpainting enables restoration of damaged images and removal of unwanted objects." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Telea algorithm uses Fast Marching Method for inpainting
 * 2. Navier-Stokes method uses fluid dynamics principles
 * 3. Mask defines regions to be inpainted (white = inpaint)
 * 4. Inpainting radius affects quality vs speed trade-off
 * 5. Works best for thin scratches and small holes
 * 6. Texture preservation is challenging for large areas
 * 7. Interactive tools enable precise damage marking
 * 8. Photo restoration combines multiple techniques
 * 9. Results depend on surrounding pixel information
 * 10. Modern methods use deep learning for better texture synthesis
 */

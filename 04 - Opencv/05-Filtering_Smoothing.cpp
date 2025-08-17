/**
 * 05-Filtering_Smoothing.cpp
 * 
 * Image filtering and smoothing operations for noise reduction and preprocessing.
 * Various kernel-based operations for image enhancement.
 * 
 * Concepts covered:
 * - Blur filters (averaging, Gaussian, median)
 * - Bilateral filtering (edge-preserving)
 * - Custom kernel filters
 * - Morphological filtering
 * - Box filter and separable filters
 * - Noise reduction techniques
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat addNoise(const cv::Mat& src, int noise_type = 0) {
    cv::Mat noisy = src.clone();
    cv::Mat noise(src.size(), src.type());
    
    if (noise_type == 0) {  // Gaussian noise
        cv::randn(noise, 0, 25);
        cv::add(noisy, noise, noisy);
    } else if (noise_type == 1) {  // Salt and pepper noise
        cv::randu(noise, 0, 255);
        cv::Mat mask = noise > 240;  // 5% salt noise
        noisy.setTo(255, mask);
        mask = noise < 15;           // 5% pepper noise  
        noisy.setTo(0, mask);
    } else {  // Uniform noise
        cv::randu(noise, -30, 30);
        cv::add(noisy, noise, noisy);
    }
    
    return noisy;
}

cv::Mat createTestImage() {
    cv::Mat image(400, 600, CV_8UC3);
    
    // Create regions with different textures
    cv::rectangle(image, cv::Point(0, 0), cv::Point(200, 200), cv::Scalar(100, 150, 200), -1);
    cv::rectangle(image, cv::Point(200, 0), cv::Point(400, 200), cv::Scalar(200, 100, 150), -1);
    cv::rectangle(image, cv::Point(400, 0), cv::Point(600, 200), cv::Scalar(150, 200, 100), -1);
    
    // Add geometric shapes
    cv::circle(image, cv::Point(100, 300), 60, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(image, cv::Point(250, 250), cv::Point(350, 350), cv::Scalar(0, 0, 0), -1);
    
    // Add text
    cv::putText(image, "Filter Test", cv::Point(400, 300), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    return image;
}

void demonstrateBasicBlurring(const cv::Mat& src) {
    std::cout << "\n=== Basic Blurring Filters ===" << std::endl;
    
    cv::Mat blur_avg, blur_gaussian, blur_median, blur_bilateral;
    
    // Average blur (box filter)
    cv::blur(src, blur_avg, cv::Size(15, 15));
    
    // Gaussian blur
    cv::GaussianBlur(src, blur_gaussian, cv::Size(15, 15), 0, 0);
    
    // Median blur (good for salt-and-pepper noise)
    cv::medianBlur(src, blur_median, 15);
    
    // Bilateral filter (edge-preserving)
    cv::bilateralFilter(src, blur_bilateral, 15, 80, 80);
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Average Blur", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Gaussian Blur", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Median Blur", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Bilateral Filter", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src);
    cv::imshow("Average Blur", blur_avg);
    cv::imshow("Gaussian Blur", blur_gaussian);
    cv::imshow("Median Blur", blur_median);
    cv::imshow("Bilateral Filter", blur_bilateral);
    
    std::cout << "Average: simple mean of neighborhood pixels" << std::endl;
    std::cout << "Gaussian: weighted average with Gaussian weights" << std::endl;
    std::cout << "Median: replaces pixel with median value (removes impulse noise)" << std::endl;
    std::cout << "Bilateral: edge-preserving smoothing" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateNoiseReduction() {
    std::cout << "\n=== Noise Reduction Demo ===" << std::endl;
    
    cv::Mat clean_image = createTestImage();
    cv::Mat gaussian_noisy = addNoise(clean_image, 0);
    cv::Mat salt_pepper_noisy = addNoise(clean_image, 1);
    
    // Different filters for different noise types
    cv::Mat gaussian_filtered, median_filtered, bilateral_filtered, nlm_filtered;
    
    // Gaussian blur for Gaussian noise
    cv::GaussianBlur(gaussian_noisy, gaussian_filtered, cv::Size(5, 5), 0);
    
    // Median filter for salt-and-pepper noise
    cv::medianBlur(salt_pepper_noisy, median_filtered, 5);
    
    // Bilateral filter preserves edges
    cv::bilateralFilter(gaussian_noisy, bilateral_filtered, 9, 75, 75);
    
    // Non-local means denoising (advanced)
    cv::fastNlMeansDenoisingColored(gaussian_noisy, nlm_filtered, 10, 10, 7, 21);
    
    // Display comparison
    cv::namedWindow("Clean Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Gaussian Noise", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Salt-Pepper Noise", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Gaussian Filtered", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Median Filtered", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Bilateral Filtered", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("NLM Denoised", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Clean Original", clean_image);
    cv::imshow("Gaussian Noise", gaussian_noisy);
    cv::imshow("Salt-Pepper Noise", salt_pepper_noisy);
    cv::imshow("Gaussian Filtered", gaussian_filtered);
    cv::imshow("Median Filtered", median_filtered);
    cv::imshow("Bilateral Filtered", bilateral_filtered);
    cv::imshow("NLM Denoised", nlm_filtered);
    
    std::cout << "Choose appropriate filter based on noise type:" << std::endl;
    std::cout << "- Gaussian noise: Gaussian blur or bilateral filter" << std::endl;
    std::cout << "- Salt-pepper noise: Median filter" << std::endl;
    std::cout << "- Mixed noise: Non-local means denoising" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateCustomKernels(const cv::Mat& src) {
    std::cout << "\n=== Custom Kernel Filtering ===" << std::endl;
    
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
    
    // Sharpen kernel
    cv::Mat sharpen_kernel = (cv::Mat_<float>(3, 3) << 
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    
    // Edge detection kernel (simple)
    cv::Mat edge_kernel = (cv::Mat_<float>(3, 3) << 
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1);
    
    // Emboss kernel
    cv::Mat emboss_kernel = (cv::Mat_<float>(3, 3) << 
        -2, -1, 0,
        -1, 1, 1,
        0, 1, 2);
    
    // Gaussian kernel (manual)
    cv::Mat gaussian_kernel = (cv::Mat_<float>(5, 5) << 
        1, 4, 6, 4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4, 6, 4, 1) / 256.0;
    
    // Apply custom filters
    cv::Mat sharpened, edges, embossed, gaussian_manual;
    cv::filter2D(src_gray, sharpened, -1, sharpen_kernel);
    cv::filter2D(src_gray, edges, -1, edge_kernel);
    cv::filter2D(src_gray, embossed, -1, emboss_kernel);
    cv::filter2D(src_gray, gaussian_manual, -1, gaussian_kernel);
    
    // Motion blur kernel
    cv::Mat motion_kernel = cv::getRotationMatrix2D(cv::Point2f(7, 7), 0, 1).rowRange(0, 1);
    motion_kernel = cv::repeat(motion_kernel, 15, 1);
    motion_kernel = motion_kernel / cv::sum(motion_kernel)[0];
    
    cv::Mat motion_blurred;
    cv::filter2D(src_gray, motion_blurred, -1, motion_kernel);
    
    // Display results
    cv::namedWindow("Original Gray", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Sharpened", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Edge Detection", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Embossed", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Manual Gaussian", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Motion Blur", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original Gray", src_gray);
    cv::imshow("Sharpened", sharpened);
    cv::imshow("Edge Detection", edges);
    cv::imshow("Embossed", embossed);
    cv::imshow("Manual Gaussian", gaussian_manual);
    cv::imshow("Motion Blur", motion_blurred);
    
    std::cout << "Custom kernels allow specialized filtering effects." << std::endl;
    std::cout << "Kernel design determines the operation performed." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateAdvancedFiltering(const cv::Mat& src) {
    std::cout << "\n=== Advanced Filtering Techniques ===" << std::endl;
    
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
    
    // Separable filter (optimized Gaussian)
    cv::Mat gaussian_sep;
    cv::sepFilter2D(src_gray, gaussian_sep, -1, 
                    cv::getGaussianKernel(15, 5), 
                    cv::getGaussianKernel(15, 5));
    
    // Box filter with different border types
    cv::Mat box_default, box_reflect, box_replicate;
    cv::boxFilter(src_gray, box_default, -1, cv::Size(15, 15), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
    cv::boxFilter(src_gray, box_reflect, -1, cv::Size(15, 15), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    cv::boxFilter(src_gray, box_replicate, -1, cv::Size(15, 15), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    
    // Pyramid filtering (multi-scale)
    cv::Mat pyramid_down, pyramid_up;
    cv::pyrDown(src_gray, pyramid_down);
    cv::pyrUp(pyramid_down, pyramid_up, src_gray.size());
    
    // Laplacian pyramid (difference)
    cv::Mat laplacian_pyramid;
    cv::subtract(src_gray, pyramid_up, laplacian_pyramid);
    
    // Guided filter approximation using bilateral
    cv::Mat guided_filtered;
    cv::bilateralFilter(src, guided_filtered, 15, 50, 50);
    
    // Anisotropic diffusion approximation
    cv::Mat anisotropic = src.clone();
    for (int i = 0; i < 5; i++) {
        cv::bilateralFilter(anisotropic, anisotropic, 5, 50, 50);
    }
    
    // Display results
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Separable Gaussian", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Box Reflect Border", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Pyramid Up", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Laplacian Pyramid", cv::WINDOW_AUTOSIZE + 100);  // Offset for visibility
    cv::namedWindow("Guided Filter", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Anisotropic Diffusion", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original", src_gray);
    cv::imshow("Separable Gaussian", gaussian_sep);
    cv::imshow("Box Reflect Border", box_reflect);
    cv::imshow("Pyramid Up", pyramid_up);
    cv::imshow("Laplacian Pyramid", laplacian_pyramid + 128);  // Offset for visualization
    cv::imshow("Guided Filter", guided_filtered);
    cv::imshow("Anisotropic Diffusion", anisotropic);
    
    std::cout << "Advanced filtering techniques for specialized applications:" << std::endl;
    std::cout << "- Separable filters: computational efficiency" << std::endl;
    std::cout << "- Pyramid filters: multi-scale analysis" << std::endl;
    std::cout << "- Laplacian pyramid: detail extraction" << std::endl;
    std::cout << "- Guided filter: edge-preserving smoothing" << std::endl;
    std::cout << "- Anisotropic diffusion: selective smoothing" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateFilterComparison(const cv::Mat& src) {
    std::cout << "\n=== Filter Performance Comparison ===" << std::endl;
    
    cv::Mat noisy_image = addNoise(src, 0);  // Add Gaussian noise
    
    // Different filter sizes
    std::vector<int> kernel_sizes = {5, 15, 25, 35};
    
    for (int size : kernel_sizes) {
        cv::Mat gaussian_result, bilateral_result, median_result;
        
        auto start = cv::getTickCount();
        cv::GaussianBlur(noisy_image, gaussian_result, cv::Size(size, size), 0);
        double gaussian_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        start = cv::getTickCount();
        cv::bilateralFilter(noisy_image, bilateral_result, size, size*2, size*2);
        double bilateral_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        start = cv::getTickCount();
        cv::medianBlur(noisy_image, median_result, size);
        double median_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        std::cout << "Kernel size " << size << ":" << std::endl;
        std::cout << "  Gaussian: " << gaussian_time << " ms" << std::endl;
        std::cout << "  Bilateral: " << bilateral_time << " ms" << std::endl;
        std::cout << "  Median: " << median_time << " ms" << std::endl;
        
        if (size == 15) {  // Display middle size for visual comparison
            cv::namedWindow("Noisy", cv::WINDOW_AUTOSIZE);
            cv::namedWindow("Gaussian 15x15", cv::WINDOW_AUTOSIZE);
            cv::namedWindow("Bilateral 15x15", cv::WINDOW_AUTOSIZE);
            cv::namedWindow("Median 15x15", cv::WINDOW_AUTOSIZE);
            
            cv::imshow("Noisy", noisy_image);
            cv::imshow("Gaussian 15x15", gaussian_result);
            cv::imshow("Bilateral 15x15", bilateral_result);
            cv::imshow("Median 15x15", median_result);
        }
    }
    
    std::cout << "\nPerformance varies with kernel size and filter complexity." << std::endl;
    std::cout << "Gaussian is fastest, bilateral is slowest but preserves edges." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== OpenCV Filtering and Smoothing Demo ===" << std::endl;
    
    // Create or load test image
    cv::Mat test_image = createTestImage();
    
    // Try to load a real image
    cv::Mat real_image = cv::imread("../data/test.jpg");
    if (!real_image.empty()) {
        test_image = real_image;
        std::cout << "Using loaded image for demonstrations." << std::endl;
    } else {
        std::cout << "Using synthetic test image." << std::endl;
    }
    
    // Demonstrate all filtering techniques
    demonstrateBasicBlurring(test_image);
    demonstrateNoiseReduction();
    demonstrateCustomKernels(test_image);
    demonstrateAdvancedFiltering(test_image);
    demonstrateFilterComparison(test_image);
    
    std::cout << "\n✓ Filtering and smoothing demonstration complete!" << std::endl;
    std::cout << "Choose filters based on noise type, performance needs, and edge preservation requirements." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Different filters for different noise types and purposes
 * 2. Gaussian blur: general smoothing with natural falloff
 * 3. Median filter: excellent for salt-and-pepper noise
 * 4. Bilateral filter: edge-preserving smoothing
 * 5. Custom kernels: unlimited filtering possibilities
 * 6. Separable filters: computational efficiency for large kernels
 * 7. Non-local means: advanced denoising technique
 * 8. Filter selection depends on application requirements
 */

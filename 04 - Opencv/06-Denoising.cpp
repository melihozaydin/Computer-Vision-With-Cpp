/**
 * 06-Denoising.cpp
 * 
 * Advanced image denoising techniques for noise reduction.
 * 
 * Concepts covered:
 * - Non-local means denoising\n * - Bilateral filtering for edge preservation\n * - Wavelet denoising\n * - BM3D denoising\n * - Noise estimation and parameter tuning
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat addNoise(const cv::Mat& src, int noise_type = 0, double intensity = 25.0) {
    cv::Mat noisy = src.clone();
    cv::Mat noise(src.size(), src.type());
    
    switch(noise_type) {
        case 0: { // Gaussian noise
            cv::randn(noise, 0, intensity);
            cv::add(noisy, noise, noisy);
            break;
        }
        case 1: { // Salt and pepper noise
            cv::randu(noise, 0, 255);
            cv::Mat salt_mask = noise > (255 - intensity);
            cv::Mat pepper_mask = noise < intensity;
            noisy.setTo(255, salt_mask);
            noisy.setTo(0, pepper_mask);
            break;
        }
        case 2: { // Speckle noise
            cv::randu(noise, 0, intensity);
            cv::multiply(noisy, noise, noisy, 1.0/255.0);
            break;
        }
    }
    return noisy;
}

void demonstrateNonLocalMeans(const cv::Mat& noisy) {
    std::cout << "\n=== Non-Local Means Denoising ===" << std::endl;
    
    cv::Mat denoised_color, denoised_gray;
    
    if (noisy.channels() == 3) {
        auto start = cv::getTickCount();
        cv::fastNlMeansDenoisingColored(noisy, denoised_color, 10, 10, 7, 21);
        double time_color = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        cv::Mat gray_noisy, gray_denoised;
        cv::cvtColor(noisy, gray_noisy, cv::COLOR_BGR2GRAY);
        start = cv::getTickCount();
        cv::fastNlMeansDenoising(gray_noisy, gray_denoised, 10, 7, 21);
        double time_gray = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        cv::namedWindow("Noisy Color", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("NLM Color Denoised", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Noisy Gray", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("NLM Gray Denoised", cv::WINDOW_AUTOSIZE);
        
        cv::imshow("Noisy Color", noisy);
        cv::imshow("NLM Color Denoised", denoised_color);
        cv::imshow("Noisy Gray", gray_noisy);
        cv::imshow("NLM Gray Denoised", gray_denoised);
        
        std::cout << "Color NLM time: " << time_color << " ms" << std::endl;
        std::cout << "Gray NLM time: " << time_gray << " ms" << std::endl;
        
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

void demonstrateBilateralFiltering(const cv::Mat& noisy) {
    std::cout << "\n=== Bilateral Filtering ===" << std::endl;
    
    cv::Mat bilateral_filtered;
    cv::bilateralFilter(noisy, bilateral_filtered, 15, 80, 80);
    
    // Compare different parameters
    cv::Mat bilateral_weak, bilateral_strong;
    cv::bilateralFilter(noisy, bilateral_weak, 5, 20, 20);
    cv::bilateralFilter(noisy, bilateral_strong, 25, 150, 150);
    
    cv::namedWindow("Noisy", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Bilateral (15,80,80)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Bilateral Weak", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Bilateral Strong", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Noisy", noisy);
    cv::imshow("Bilateral (15,80,80)", bilateral_filtered);
    cv::imshow("Bilateral Weak", bilateral_weak);
    cv::imshow("Bilateral Strong", bilateral_strong);
    
    std::cout << "Bilateral filtering preserves edges while smoothing noise." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateMethodComparison(const cv::Mat& original, const cv::Mat& noisy) {
    std::cout << "\n=== Denoising Methods Comparison ===" << std::endl;
    
    cv::Mat gaussian_filtered, median_filtered, bilateral_filtered, nlm_filtered;
    
    // Apply different methods
    cv::GaussianBlur(noisy, gaussian_filtered, cv::Size(5, 5), 0);
    cv::medianBlur(noisy, median_filtered, 5);
    cv::bilateralFilter(noisy, bilateral_filtered, 9, 75, 75);
    
    if (noisy.channels() == 3) {
        cv::fastNlMeansDenoisingColored(noisy, nlm_filtered, 10, 10, 7, 21);
    } else {
        cv::fastNlMeansDenoising(noisy, nlm_filtered, 10, 7, 21);
    }
    
    // Calculate PSNR for each method
    auto calculatePSNR = [](const cv::Mat& original, const cv::Mat& processed) {
        cv::Mat diff;
        cv::absdiff(original, processed, diff);
        diff.convertTo(diff, CV_32F);
        diff = diff.mul(diff);
        cv::Scalar mse = cv::mean(diff);
        double psnr = 10.0 * log10((255.0 * 255.0) / mse[0]);
        return psnr;
    };
    
    double psnr_noisy = calculatePSNR(original, noisy);
    double psnr_gaussian = calculatePSNR(original, gaussian_filtered);
    double psnr_median = calculatePSNR(original, median_filtered);
    double psnr_bilateral = calculatePSNR(original, bilateral_filtered);
    double psnr_nlm = calculatePSNR(original, nlm_filtered);
    
    std::cout << "PSNR Comparison:" << std::endl;
    std::cout << "Noisy: " << psnr_noisy << " dB" << std::endl;
    std::cout << "Gaussian: " << psnr_gaussian << " dB" << std::endl;
    std::cout << "Median: " << psnr_median << " dB" << std::endl;
    std::cout << "Bilateral: " << psnr_bilateral << " dB" << std::endl;
    std::cout << "NLM: " << psnr_nlm << " dB" << std::endl;
    
    // Create comparison montage
    cv::Mat top_row, bottom_row, comparison;
    cv::hconcat(std::vector<cv::Mat>{original, noisy, gaussian_filtered}, top_row);
    cv::hconcat(std::vector<cv::Mat>{median_filtered, bilateral_filtered, nlm_filtered}, bottom_row);
    cv::vconcat(std::vector<cv::Mat>{top_row, bottom_row}, comparison);
    
    cv::namedWindow("Denoising Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("Denoising Comparison", comparison);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Image Denoising ===" << std::endl;
    
    // Create or load test image
    cv::Mat original;
    cv::Mat loaded = cv::imread("../data/test.jpg");
    if (!loaded.empty()) {
        original = loaded;
        std::cout << "Using loaded test image." << std::endl;
    } else {
        // Create synthetic test image
        original = cv::Mat(300, 400, CV_8UC3);
        cv::rectangle(original, cv::Point(50, 50), cv::Point(350, 250), cv::Scalar(100, 150, 200), -1);
        cv::circle(original, cv::Point(200, 150), 60, cv::Scalar(255, 255, 255), -1);
        cv::rectangle(original, cv::Point(100, 100), cv::Point(300, 200), cv::Scalar(0, 0, 0), 3);
        std::cout << "Using synthetic test image." << std::endl;
    }
    
    // Add different types of noise
    cv::Mat gaussian_noisy = addNoise(original, 0, 25);  // Gaussian noise
    cv::Mat sp_noisy = addNoise(original, 1, 15);        // Salt & pepper noise
    
    // Demonstrate different denoising methods
    demonstrateNonLocalMeans(gaussian_noisy);
    demonstrateBilateralFiltering(gaussian_noisy);
    demonstrateMethodComparison(original, gaussian_noisy);
    
    std::cout << "\n✓ Image Denoising demonstration complete!" << std::endl;
    std::cout << "Non-local means is most effective for preserving details while removing noise." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Add key concepts here
 * 2. Implementation details
 * 3. Best practices
 * 4. Common pitfalls to avoid
 */

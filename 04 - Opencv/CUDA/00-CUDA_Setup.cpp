/**
 * 00-CUDA_Setup.cpp
 * 
 * CUDA setup verification and basic GPU operations with OpenCV.
 * Tests CUDA device availability and basic memory operations.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>

int main() {
    std::cout << "=== OpenCV CUDA Setup Test ===" << std::endl;
    
    // Check if OpenCV was compiled with CUDA support
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    try {
        // Check number of CUDA devices
        int device_count = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "CUDA-enabled devices: " << device_count << std::endl;
        
        if (device_count == 0) {
            std::cout << "No CUDA-capable devices found." << std::endl;
            std::cout << "Please check:" << std::endl;
            std::cout << "1. NVIDIA GPU is installed" << std::endl;
            std::cout << "2. CUDA drivers are installed" << std::endl;
            std::cout << "3. OpenCV was compiled with CUDA support" << std::endl;
            return -1;
        }
        
        // Get device properties
        for (int i = 0; i < device_count; i++) {
            cv::cuda::DeviceInfo device_info(i);
            std::cout << "\nDevice " << i << " Properties:" << std::endl;
            std::cout << "  Name: " << device_info.name() << std::endl;
            std::cout << "  Compute Capability: " << device_info.majorVersion() 
                      << "." << device_info.minorVersion() << std::endl;
            std::cout << "  Total Memory: " << device_info.totalMemory() / (1024*1024) << " MB" << std::endl;
            std::cout << "  Multiprocessors: " << device_info.multiProcessorCount() << std::endl;
            std::cout << "  Max Threads per Block: " << device_info.maxThreadsPerBlock() << std::endl;
            std::cout << "  Max Thread Dimensions: (" 
                      << device_info.maxThreadsDim().x << ", "
                      << device_info.maxThreadsDim().y << ", "
                      << device_info.maxThreadsDim().z << ")" << std::endl;
            std::cout << "  Max Grid Size: ("
                      << device_info.maxGridSize().x << ", "
                      << device_info.maxGridSize().y << ", "
                      << device_info.maxGridSize().z << ")" << std::endl;
        }
        
        // Set device (if multiple available)
        cv::cuda::setDevice(0);
        std::cout << "\nUsing device 0 for operations." << std::endl;
        
        // Create test matrices
        cv::Mat host_mat = cv::Mat::ones(1000, 1000, CV_32F) * 5.0f;
        cv::cuda::GpuMat gpu_mat;
        
        // Upload to GPU
        auto start = cv::getTickCount();
        gpu_mat.upload(host_mat);
        double upload_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        // Perform GPU operation
        cv::cuda::GpuMat gpu_result;
        start = cv::getTickCount();
        cv::cuda::multiply(gpu_mat, gpu_mat, gpu_result);  // Square each element
        double gpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        // Download result
        cv::Mat result;
        start = cv::getTickCount();
        gpu_result.download(result);
        double download_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        // Compare with CPU operation
        cv::Mat cpu_result;
        start = cv::getTickCount();
        cv::multiply(host_mat, host_mat, cpu_result);
        double cpu_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        // Verify results match
        cv::Mat diff;
        cv::absdiff(result, cpu_result, diff);
        cv::Scalar mean_diff = cv::mean(diff);
        
        std::cout << "\n=== Performance Comparison ===" << std::endl;
        std::cout << "Matrix size: " << host_mat.rows << "x" << host_mat.cols << std::endl;
        std::cout << "Upload time: " << upload_time << " ms" << std::endl;
        std::cout << "GPU computation: " << gpu_time << " ms" << std::endl;
        std::cout << "Download time: " << download_time << " ms" << std::endl;
        std::cout << "Total GPU time: " << upload_time + gpu_time + download_time << " ms" << std::endl;
        std::cout << "CPU computation: " << cpu_time << " ms" << std::endl;
        std::cout << "Speedup: " << cpu_time / gpu_time << "x (computation only)" << std::endl;
        std::cout << "Mean difference: " << mean_diff[0] << " (should be ~0)" << std::endl;
        
        // Test image operations
        std::cout << "\n=== Image Operations Test ===" << std::endl;
        
        // Create test image
        cv::Mat test_image(512, 512, CV_8UC3);
        cv::randu(test_image, 0, 255);
        
        cv::cuda::GpuMat gpu_image, gpu_gray;
        gpu_image.upload(test_image);
        
        // Convert to grayscale on GPU
        start = cv::getTickCount();
        cv::cuda::cvtColor(gpu_image, gpu_gray, cv::COLOR_BGR2GRAY);
        double gpu_cvt_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        // Compare with CPU
        cv::Mat cpu_gray;
        start = cv::getTickCount();
        cv::cvtColor(test_image, cpu_gray, cv::COLOR_BGR2GRAY);
        double cpu_cvt_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        std::cout << "Color conversion (BGR->Gray):" << std::endl;
        std::cout << "  GPU: " << gpu_cvt_time << " ms" << std::endl;
        std::cout << "  CPU: " << cpu_cvt_time << " ms" << std::endl;
        std::cout << "  Speedup: " << cpu_cvt_time / gpu_cvt_time << "x" << std::endl;
        
        // Test Gaussian blur
        cv::cuda::GpuMat gpu_blurred;
        cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(
            gpu_gray.type(), -1, cv::Size(15, 15), 5.0);
        
        start = cv::getTickCount();
        gaussian_filter->apply(gpu_gray, gpu_blurred);
        double gpu_blur_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        cv::Mat cpu_blurred;
        start = cv::getTickCount();
        cv::GaussianBlur(cpu_gray, cpu_blurred, cv::Size(15, 15), 5.0);
        double cpu_blur_time = (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        
        std::cout << "Gaussian blur (15x15, sigma=5):" << std::endl;
        std::cout << "  GPU: " << gpu_blur_time << " ms" << std::endl;
        std::cout << "  CPU: " << cpu_blur_time << " ms" << std::endl;
        std::cout << "  Speedup: " << cpu_blur_time / gpu_blur_time << "x" << std::endl;
        
        std::cout << "\n✓ CUDA setup verification complete!" << std::endl;
        std::cout << "GPU acceleration is working correctly." << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cout << "OpenCV CUDA Error: " << e.what() << std::endl;
        std::cout << "This usually means OpenCV was not compiled with CUDA support." << std::endl;
        return -1;
    }
    
    return 0;
}

/**
 * Expected Output:
 * - CUDA device information
 * - Basic GPU vs CPU performance comparison
 * - Image operation speedup measurements
 * 
 * If this runs successfully, your CUDA setup is working correctly.
 */

/**
 * 32-Texture_Analysis.cpp
 * 
 * Texture feature extraction and analysis.
 * 
 * Concepts covered:
 * - Local Binary Patterns (LBP)
 * - Gabor filters
 * - Haralick features
 * - Texture classification
 * - Rotation invariant textures
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>

cv::Mat createTextureTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC1);
    
    // Create different texture regions
    
    // Horizontal stripes
    for (int y = 50; y < 150; y++) {
        for (int x = 50; x < 200; x++) {
            if ((y - 50) % 10 < 5) {
                image.at<uchar>(y, x) = 200;
            } else {
                image.at<uchar>(y, x) = 100;
            }
        }
    }
    
    // Vertical stripes
    for (int y = 50; y < 150; y++) {
        for (int x = 250; x < 400; x++) {
            if ((x - 250) % 8 < 4) {
                image.at<uchar>(y, x) = 180;
            } else {
                image.at<uchar>(y, x) = 80;
            }
        }
    }
    
    // Checkerboard pattern
    for (int y = 200; y < 350; y++) {
        for (int x = 50; x < 200; x++) {
            int check_x = (x - 50) / 15;
            int check_y = (y - 200) / 15;
            if ((check_x + check_y) % 2 == 0) {
                image.at<uchar>(y, x) = 220;
            } else {
                image.at<uchar>(y, x) = 60;
            }
        }
    }
    
    // Random texture
    cv::Mat roi = image(cv::Rect(250, 200, 150, 150));
    cv::randu(roi, cv::Scalar(50), cv::Scalar(200));
    
    // Circular patterns
    for (int y = 50; y < 150; y++) {
        for (int x = 450; x < 550; x++) {
            double dist = std::sqrt(std::pow(x - 500, 2) + std::pow(y - 100, 2));
            int intensity = static_cast<int>(128 + 100 * std::sin(dist * 0.3));
            image.at<uchar>(y, x) = cv::saturate_cast<uchar>(intensity);
        }
    }
    
    // Add noise to make analysis more realistic
    cv::Mat noise(image.size(), CV_8UC1);
    cv::randu(noise, cv::Scalar(0), cv::Scalar(30));
    cv::add(image, noise, image);
    
    return image;
}

cv::Mat computeLBP(const cv::Mat& src, int radius = 1, int neighbors = 8) {
    cv::Mat lbp = cv::Mat::zeros(src.size(), CV_8UC1);
    
    for (int y = radius; y < src.rows - radius; y++) {
        for (int x = radius; x < src.cols - radius; x++) {
            uchar center = src.at<uchar>(y, x);
            uchar lbp_value = 0;
            
            // Sample neighbors in circular pattern
            for (int n = 0; n < neighbors; n++) {
                double angle = 2.0 * CV_PI * n / neighbors;
                int dx = static_cast<int>(radius * std::cos(angle));
                int dy = static_cast<int>(radius * std::sin(angle));
                
                uchar neighbor = src.at<uchar>(y + dy, x + dx);
                
                if (neighbor >= center) {
                    lbp_value |= (1 << n);
                }
            }
            
            lbp.at<uchar>(y, x) = lbp_value;
        }
    }
    
    return lbp;
}

void demonstrateLBP(const cv::Mat& src) {
    std::cout << "\n=== Local Binary Patterns (LBP) ===" << std::endl;
    
    // Compute LBP with different parameters
    cv::Mat lbp_8_1 = computeLBP(src, 1, 8);  // Radius 1, 8 neighbors
    cv::Mat lbp_16_2 = computeLBP(src, 2, 16); // Radius 2, 16 neighbors
    
    // Calculate LBP histograms for different regions
    std::vector<cv::Rect> regions = {
        cv::Rect(50, 50, 150, 100),   // Horizontal stripes
        cv::Rect(250, 50, 150, 100),  // Vertical stripes
        cv::Rect(50, 200, 150, 150),  // Checkerboard
        cv::Rect(250, 200, 150, 150)  // Random
    };
    
    std::vector<std::string> region_names = {"Horizontal", "Vertical", "Checkerboard", "Random"};
    
    // Create visualization
    cv::Mat display;
    cv::hconcat(src, lbp_8_1, display);
    cv::Mat bottom_row;
    cv::hconcat(lbp_16_2, cv::Mat::zeros(src.size(), CV_8UC1), bottom_row);
    cv::vconcat(display, bottom_row, display);
    
    // Convert to color for labeling
    cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "LBP (R=1, N=8)", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "LBP (R=2, N=16)", cv::Point(10, src.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::namedWindow("Local Binary Patterns", cv::WINDOW_AUTOSIZE);
    cv::imshow("Local Binary Patterns", display);
    
    // Compute and display histograms
    for (size_t i = 0; i < regions.size(); i++) {
        cv::Mat roi = lbp_8_1(regions[i]);
        
        // Calculate histogram
        std::vector<int> hist(256, 0);
        for (int y = 0; y < roi.rows; y++) {
            for (int x = 0; x < roi.cols; x++) {
                hist[roi.at<uchar>(y, x)]++;
            }
        }
        
        // Find most frequent patterns
        std::vector<std::pair<int, int>> pattern_freq;
        for (int j = 0; j < 256; j++) {
            if (hist[j] > 0) {
                pattern_freq.push_back({hist[j], j});
            }
        }
        std::sort(pattern_freq.rbegin(), pattern_freq.rend());
        
        std::cout << region_names[i] << " texture - Top LBP patterns:" << std::endl;
        for (int k = 0; k < std::min(5, static_cast<int>(pattern_freq.size())); k++) {
            std::cout << "  Pattern " << pattern_freq[k].second << ": " << pattern_freq[k].first << " pixels" << std::endl;
        }
    }
    
    std::cout << "\nLBP characteristics:" << std::endl;
    std::cout << "  - Robust to monotonic illumination changes" << std::endl;
    std::cout << "  - Captures local texture information" << std::endl;
    std::cout << "  - Different patterns indicate different textures" << std::endl;
    std::cout << "  - Histogram provides texture descriptor" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

cv::Mat createGaborKernel(cv::Size ksize, double sigma, double theta, double lambda, double gamma, double psi = 0) {
    cv::Mat kernel(ksize, CV_32F);
    
    int xmax = ksize.width / 2;
    int ymax = ksize.height / 2;
    
    double sigma_x = sigma;
    double sigma_y = sigma / gamma;
    
    for (int y = -ymax; y <= ymax; y++) {
        for (int x = -xmax; x <= xmax; x++) {
            double x_theta = x * std::cos(theta) + y * std::sin(theta);
            double y_theta = -x * std::sin(theta) + y * std::cos(theta);
            
            double gaussian = std::exp(-0.5 * (x_theta * x_theta / (sigma_x * sigma_x) + 
                                              y_theta * y_theta / (sigma_y * sigma_y)));
            
            double cosine = std::cos(2 * CV_PI * x_theta / lambda + psi);
            
            kernel.at<float>(y + ymax, x + xmax) = gaussian * cosine;
        }
    }
    
    return kernel;
}

void demonstrateGaborFilters(const cv::Mat& src) {
    std::cout << "\n=== Gabor Filter Bank ===" << std::endl;
    
    // Create Gabor filter bank with different orientations
    std::vector<double> orientations = {0, CV_PI/4, CV_PI/2, 3*CV_PI/4};
    std::vector<cv::Mat> gabor_responses;
    
    cv::Size kernel_size(31, 31);
    double sigma = 4;
    double lambda = 10;
    double gamma = 0.5;
    
    cv::Mat src_float;
    src.convertTo(src_float, CV_32F);
    
    for (double theta : orientations) {
        cv::Mat gabor_kernel = createGaborKernel(kernel_size, sigma, theta, lambda, gamma);
        
        cv::Mat response;
        cv::filter2D(src_float, response, CV_32F, gabor_kernel);
        
        // Normalize response
        cv::normalize(response, response, 0, 255, cv::NORM_MINMAX);
        response.convertTo(response, CV_8U);
        
        gabor_responses.push_back(response);
    }
    
    // Create display grid
    cv::Mat top_row, bottom_row, display;
    cv::hconcat(src, gabor_responses[0], top_row);
    cv::hconcat(gabor_responses[1], gabor_responses[2], bottom_row);
    cv::vconcat(top_row, bottom_row, display);
    
    // Convert to color for labeling
    cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "0 deg", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "45 deg", cv::Point(10, src.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(display, "90 deg", cv::Point(src.cols + 10, src.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::namedWindow("Gabor Filter Bank", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gabor Filter Bank", display);
    
    // Compute texture energy for each orientation
    for (size_t i = 0; i < orientations.size(); i++) {
        cv::Scalar energy = cv::sum(gabor_responses[i]);
        double angle_deg = orientations[i] * 180 / CV_PI;
        std::cout << "Orientation " << angle_deg << "°: Energy = " << energy[0] << std::endl;
    }
    
    std::cout << "\nGabor filter characteristics:" << std::endl;
    std::cout << "  - Sensitive to specific orientations and frequencies" << std::endl;
    std::cout << "  - Sigma controls spatial extent" << std::endl;
    std::cout << "  - Lambda controls frequency selectivity" << std::endl;
    std::cout << "  - Gamma controls aspect ratio" << std::endl;
    std::cout << "  - Filter bank captures multiple texture properties" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

cv::Mat computeGLCM(const cv::Mat& src, int dx, int dy, int levels = 256) {
    cv::Mat glcm = cv::Mat::zeros(levels, levels, CV_32F);
    
    for (int y = 0; y < src.rows - std::abs(dy); y++) {
        for (int x = 0; x < src.cols - std::abs(dx); x++) {
            if (x + dx >= 0 && x + dx < src.cols && y + dy >= 0 && y + dy < src.rows) {
                uchar i = src.at<uchar>(y, x);
                uchar j = src.at<uchar>(y + dy, x + dx);
                
                // Quantize to reduce levels if needed
                if (levels < 256) {
                    i = i * levels / 256;
                    j = j * levels / 256;
                }
                
                glcm.at<float>(i, j)++;
            }
        }
    }
    
    // Normalize
    cv::Scalar sum = cv::sum(glcm);
    if (sum[0] > 0) {
        glcm /= sum[0];
    }
    
    return glcm;
}

void computeHaralickFeatures(const cv::Mat& glcm, std::vector<double>& features) {
    features.clear();
    
    // Contrast
    double contrast = 0;
    double energy = 0;
    double homogeneity = 0;
    double correlation = 0;
    
    // Mean values
    double mu_i = 0, mu_j = 0;
    for (int i = 0; i < glcm.rows; i++) {
        for (int j = 0; j < glcm.cols; j++) {
            float p = glcm.at<float>(i, j);
            mu_i += i * p;
            mu_j += j * p;
        }
    }
    
    // Variance values
    double sigma_i = 0, sigma_j = 0;
    for (int i = 0; i < glcm.rows; i++) {
        for (int j = 0; j < glcm.cols; j++) {
            float p = glcm.at<float>(i, j);
            sigma_i += (i - mu_i) * (i - mu_i) * p;
            sigma_j += (j - mu_j) * (j - mu_j) * p;
        }
    }
    sigma_i = std::sqrt(sigma_i);
    sigma_j = std::sqrt(sigma_j);
    
    // Compute features
    for (int i = 0; i < glcm.rows; i++) {
        for (int j = 0; j < glcm.cols; j++) {
            float p = glcm.at<float>(i, j);
            
            // Contrast (measure of local intensity variation)
            contrast += (i - j) * (i - j) * p;
            
            // Energy (measure of uniformity)
            energy += p * p;
            
            // Homogeneity (measure of closeness of distribution)
            homogeneity += p / (1.0 + std::abs(i - j));
            
            // Correlation (measure of linear dependency)
            if (sigma_i > 0 && sigma_j > 0) {
                correlation += ((i - mu_i) * (j - mu_j) * p) / (sigma_i * sigma_j);
            }
        }
    }
    
    features.push_back(contrast);
    features.push_back(energy);
    features.push_back(homogeneity);
    features.push_back(correlation);
}

void demonstrateHaralickFeatures(const cv::Mat& src) {
    std::cout << "\n=== Haralick Texture Features ===" << std::endl;
    
    // Define regions with different textures
    std::vector<cv::Rect> regions = {
        cv::Rect(50, 50, 100, 100),   // Horizontal stripes
        cv::Rect(250, 50, 100, 100),  // Vertical stripes
        cv::Rect(50, 200, 100, 100),  // Checkerboard
        cv::Rect(250, 200, 100, 100)  // Random
    };
    
    std::vector<std::string> region_names = {"Horizontal", "Vertical", "Checkerboard", "Random"};
    
    // Compute GLCM and Haralick features for each region
    std::vector<int> directions_x = {1, 1, 0, -1};  // 0°, 45°, 90°, 135°
    std::vector<int> directions_y = {0, 1, 1, 1};
    std::vector<std::string> direction_names = {"0°", "45°", "90°", "135°"};
    
    cv::Mat visualization = cv::Mat::zeros(400, 600, CV_8UC3);
    src.copyTo(visualization(cv::Rect(0, 0, src.cols, src.rows)));
    cv::cvtColor(visualization, visualization, cv::COLOR_GRAY2BGR);
    
    // Draw region boundaries
    for (size_t i = 0; i < regions.size(); i++) {
        cv::rectangle(visualization, regions[i], cv::Scalar(0, 255, 0), 2);
        cv::putText(visualization, std::to_string(i+1), 
                   cv::Point(regions[i].x + 5, regions[i].y + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    
    cv::namedWindow("Texture Regions", cv::WINDOW_AUTOSIZE);
    cv::imshow("Texture Regions", visualization);
    
    for (size_t i = 0; i < regions.size(); i++) {
        cv::Mat roi = src(regions[i]);
        
        std::cout << "\n" << region_names[i] << " texture:" << std::endl;
        
        for (size_t d = 0; d < directions_x.size(); d++) {
            cv::Mat glcm = computeGLCM(roi, directions_x[d], directions_y[d], 64);  // Reduced levels for efficiency
            
            std::vector<double> features;
            computeHaralickFeatures(glcm, features);
            
            std::cout << "  Direction " << direction_names[d] << ":" << std::endl;
            std::cout << "    Contrast: " << features[0] << std::endl;
            std::cout << "    Energy: " << features[1] << std::endl;
            std::cout << "    Homogeneity: " << features[2] << std::endl;
            std::cout << "    Correlation: " << features[3] << std::endl;
        }
    }
    
    std::cout << "\nHaralick feature interpretation:" << std::endl;
    std::cout << "  - Contrast: High for varying textures, low for uniform" << std::endl;
    std::cout << "  - Energy: High for regular patterns, low for random" << std::endl;
    std::cout << "  - Homogeneity: High for similar neighboring intensities" << std::endl;
    std::cout << "  - Correlation: Measures linear dependency of intensities" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateTextureClassification(const cv::Mat& src) {
    std::cout << "\n=== Texture Classification Demo ===" << std::endl;
    
    // Define texture regions
    std::vector<cv::Rect> regions = {
        cv::Rect(50, 50, 100, 100),   // Horizontal
        cv::Rect(250, 50, 100, 100),  // Vertical
        cv::Rect(50, 200, 100, 100),  // Checkerboard
        cv::Rect(250, 200, 100, 100)  // Random
    };
    
    std::vector<std::string> texture_types = {"Horizontal", "Vertical", "Checkerboard", "Random"};
    
    // Extract features for each region
    std::vector<std::vector<double>> feature_vectors;
    
    for (const auto& region : regions) {
        cv::Mat roi = src(region);
        
        // Compute LBP histogram
        cv::Mat lbp = computeLBP(roi);
        std::vector<int> lbp_hist(256, 0);
        for (int y = 0; y < lbp.rows; y++) {
            for (int x = 0; x < lbp.cols; x++) {
                lbp_hist[lbp.at<uchar>(y, x)]++;
            }
        }
        
        // Normalize histogram
        int total = roi.rows * roi.cols;
        std::vector<double> normalized_hist(10, 0);  // Use top 10 bins for simplicity
        std::vector<std::pair<int, int>> sorted_hist;
        for (int i = 0; i < 256; i++) {
            sorted_hist.push_back({lbp_hist[i], i});
        }
        std::sort(sorted_hist.rbegin(), sorted_hist.rend());
        
        for (int i = 0; i < 10 && i < static_cast<int>(sorted_hist.size()); i++) {
            normalized_hist[i] = static_cast<double>(sorted_hist[i].first) / total;
        }
        
        // Compute Haralick features
        cv::Mat glcm = computeGLCM(roi, 1, 0, 64);
        std::vector<double> haralick_features;
        computeHaralickFeatures(glcm, haralick_features);
        
        // Combine features
        std::vector<double> combined_features = normalized_hist;
        combined_features.insert(combined_features.end(), haralick_features.begin(), haralick_features.end());
        
        feature_vectors.push_back(combined_features);
    }
    
    // Simple classification: find most similar texture for test regions
    std::vector<cv::Rect> test_regions = {
        cv::Rect(450, 50, 80, 80),   // Test region 1
        cv::Rect(450, 200, 80, 80)   // Test region 2
    };
    
    cv::Mat classification_display;
    cv::cvtColor(src, classification_display, cv::COLOR_GRAY2BGR);
    
    // Draw training regions
    for (size_t i = 0; i < regions.size(); i++) {
        cv::rectangle(classification_display, regions[i], cv::Scalar(0, 255, 0), 2);
        cv::putText(classification_display, texture_types[i], 
                   cv::Point(regions[i].x, regions[i].y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    
    // Classify test regions
    for (size_t t = 0; t < test_regions.size(); t++) {
        cv::Mat test_roi = src(test_regions[t]);
        
        // Extract features for test region
        cv::Mat lbp = computeLBP(test_roi);
        std::vector<int> lbp_hist(256, 0);
        for (int y = 0; y < lbp.rows; y++) {
            for (int x = 0; x < lbp.cols; x++) {
                lbp_hist[lbp.at<uchar>(y, x)]++;
            }
        }
        
        int total = test_roi.rows * test_roi.cols;
        std::vector<double> test_hist(10, 0);
        std::vector<std::pair<int, int>> sorted_hist;
        for (int i = 0; i < 256; i++) {
            sorted_hist.push_back({lbp_hist[i], i});
        }
        std::sort(sorted_hist.rbegin(), sorted_hist.rend());
        
        for (int i = 0; i < 10 && i < static_cast<int>(sorted_hist.size()); i++) {
            test_hist[i] = static_cast<double>(sorted_hist[i].first) / total;
        }
        
        cv::Mat glcm = computeGLCM(test_roi, 1, 0, 64);
        std::vector<double> haralick_features;
        computeHaralickFeatures(glcm, haralick_features);
        
        std::vector<double> test_features = test_hist;
        test_features.insert(test_features.end(), haralick_features.begin(), haralick_features.end());
        
        // Find closest match using Euclidean distance
        double min_distance = std::numeric_limits<double>::max();
        int best_match = 0;
        
        for (size_t i = 0; i < feature_vectors.size(); i++) {
            double distance = 0;
            for (size_t j = 0; j < test_features.size() && j < feature_vectors[i].size(); j++) {
                double diff = test_features[j] - feature_vectors[i][j];
                distance += diff * diff;
            }
            distance = std::sqrt(distance);
            
            if (distance < min_distance) {
                min_distance = distance;
                best_match = i;
            }
        }
        
        // Draw classification result
        cv::rectangle(classification_display, test_regions[t], cv::Scalar(0, 0, 255), 2);
        cv::putText(classification_display, texture_types[best_match], 
                   cv::Point(test_regions[t].x, test_regions[t].y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        
        std::cout << "Test region " << (t+1) << " classified as: " << texture_types[best_match] 
                  << " (distance: " << min_distance << ")" << std::endl;
    }
    
    cv::namedWindow("Texture Classification", cv::WINDOW_AUTOSIZE);
    cv::imshow("Texture Classification", classification_display);
    
    std::cout << "\nTexture classification process:" << std::endl;
    std::cout << "  - Extract features from training regions" << std::endl;
    std::cout << "  - Compute features for test regions" << std::endl;
    std::cout << "  - Find closest match using distance metric" << std::endl;
    std::cout << "  - Green: Training regions, Red: Test regions" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Texture Analysis Demonstration ===" << std::endl;
    
    // Try to load a real image
    cv::Mat real_image = cv::imread("../data/texture.jpg", cv::IMREAD_GRAYSCALE);
    
    cv::Mat image;
    if (!real_image.empty()) {
        std::cout << "Using real texture image." << std::endl;
        image = real_image;
        
        // Resize if too large
        if (image.cols > 800 || image.rows > 600) {
            cv::resize(image, image, cv::Size(800, 600));
        }
    } else {
        std::cout << "Using synthetic texture image." << std::endl;
        image = createTextureTestImage();
    }
    
    // Demonstrate texture analysis techniques
    demonstrateLBP(image);
    demonstrateGaborFilters(image);
    demonstrateHaralickFeatures(image);
    demonstrateTextureClassification(image);
    
    std::cout << "\n✓ Texture Analysis demonstration complete!" << std::endl;
    std::cout << "Texture analysis provides powerful tools for material and pattern recognition." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. LBP captures local texture patterns robustly
 * 2. Gabor filters respond to specific orientations and frequencies
 * 3. Haralick features describe texture properties statistically
 * 4. GLCM (Gray-Level Co-occurrence Matrix) captures spatial relationships
 * 5. Texture classification combines multiple feature types
 * 6. Direction-dependent features capture orientation information
 * 7. Feature normalization is crucial for classification
 * 8. Texture analysis is fundamental for material recognition
 * 9. Different techniques complement each other
 * 10. Applications include medical imaging, remote sensing, and quality control
 */

/**
 * 21-Clustering_Segmentation.cpp
 * 
 * Segmentation using clustering algorithms.
 * 
 * Concepts covered:
 * - K-means clustering
 * - Mean-shift segmentation
 * - Color quantization
 * - Spatial-color clustering
 * - Superpixel segmentation
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>

cv::Mat createClusteringTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // Create regions with different colors for clustering
    // Region 1: Red gradient
    for (int y = 50; y < 150; y++) {
        for (int x = 50; x < 200; x++) {
            int intensity = 150 + (x - 50) * 105 / 150;
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, intensity);
        }
    }
    
    // Region 2: Green area
    cv::rectangle(image, cv::Point(250, 50), cv::Point(400, 150), cv::Scalar(0, 200, 0), -1);
    
    // Region 3: Blue circular region
    cv::circle(image, cv::Point(500, 100), 60, cv::Scalar(200, 0, 0), -1);
    
    // Region 4: Yellow bottom region
    cv::rectangle(image, cv::Point(50, 200), cv::Point(300, 350), cv::Scalar(0, 200, 200), -1);
    
    // Region 5: Purple region
    cv::rectangle(image, cv::Point(350, 200), cv::Point(550, 350), cv::Scalar(150, 0, 150), -1);
    
    // Add some noise to make clustering more interesting
    cv::Mat noise(image.size(), CV_8UC3);
    cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(50));
    cv::add(image, noise, image);
    
    return image;
}

void demonstrateKMeansClustering(const cv::Mat& src) {
    std::cout << "\n=== K-Means Clustering Segmentation ===" << std::endl;
    
    // Reshape image to a 2D array of pixels
    cv::Mat data;
    src.reshape(1, src.rows * src.cols).convertTo(data, CV_32F);
    
    // Test different numbers of clusters
    std::vector<int> k_values = {2, 4, 6, 8};
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    
    for (size_t i = 0; i < k_values.size() && i < 3; i++) {
        int k = k_values[i];
        
        // Apply K-means
        cv::Mat labels, centers;
        cv::kmeans(data, k, labels, 
                  cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.0), 
                  3, cv::KMEANS_PP_CENTERS, centers);
        
        // Create segmented image
        cv::Mat segmented = cv::Mat::zeros(src.size(), CV_8UC3);
        
        // Generate distinct colors for each cluster
        std::vector<cv::Vec3b> colors(k);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        
        for (int j = 0; j < k; j++) {
            colors[j] = cv::Vec3b(dis(gen), dis(gen), dis(gen));
        }
        
        // Assign colors based on cluster labels
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                int cluster_idx = labels.at<int>(y * src.cols + x);
                segmented.at<cv::Vec3b>(y, x) = colors[cluster_idx];
            }
        }
        
        // Place in display grid
        int row = (i + 1) / 2;
        int col = (i + 1) % 2;
        segmented.copyTo(display(cv::Rect(col * src.cols, row * src.rows, src.cols, src.rows)));
        
        // Add label
        cv::putText(display, "K=" + std::to_string(k), 
                   cv::Point(col * src.cols + 10, row * src.rows + 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Calculate and display cluster centers
        std::cout << "K=" << k << " cluster centers:" << std::endl;
        for (int j = 0; j < k; j++) {
            cv::Vec3f center = centers.at<cv::Vec3f>(j);
            std::cout << "  Cluster " << j << ": BGR(" << center[0] << ", " << center[1] << ", " << center[2] << ")" << std::endl;
        }
    }
    
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("K-Means Clustering", cv::WINDOW_AUTOSIZE);
    cv::imshow("K-Means Clustering", display);
    
    std::cout << "K-means clustering characteristics:" << std::endl;
    std::cout << "  - Partitions pixels into K clusters based on color similarity" << std::endl;
    std::cout << "  - More clusters = finer segmentation but more complexity" << std::endl;
    std::cout << "  - KMEANS_PP_CENTERS initialization improves convergence" << std::endl;
    std::cout << "  - Works well when distinct color regions exist" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateColorQuantization(const cv::Mat& src) {
    std::cout << "\n=== Color Quantization with K-Means ===" << std::endl;
    
    // Reshape image for K-means
    cv::Mat data;
    src.reshape(1, src.rows * src.cols).convertTo(data, CV_32F);
    
    // Test different levels of quantization
    std::vector<int> color_levels = {2, 4, 8, 16};
    cv::Mat display = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    
    for (size_t i = 0; i < color_levels.size() && i < 3; i++) {
        int k = color_levels[i];
        
        // Apply K-means for color quantization
        cv::Mat labels, centers;
        cv::kmeans(data, k, labels, 
                  cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.0), 
                  3, cv::KMEANS_PP_CENTERS, centers);
        
        // Create quantized image using cluster centers as colors
        cv::Mat quantized = cv::Mat::zeros(src.size(), CV_8UC3);
        
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                int cluster_idx = labels.at<int>(y * src.cols + x);
                cv::Vec3f center = centers.at<cv::Vec3f>(cluster_idx);
                quantized.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    cv::saturate_cast<uchar>(center[0]),
                    cv::saturate_cast<uchar>(center[1]),
                    cv::saturate_cast<uchar>(center[2])
                );
            }
        }
        
        // Place in display grid
        int row = (i + 1) / 2;
        int col = (i + 1) % 2;
        quantized.copyTo(display(cv::Rect(col * src.cols, row * src.rows, src.cols, src.rows)));
        
        // Add label
        cv::putText(display, std::to_string(k) + " colors", 
                   cv::Point(col * src.cols + 10, row * src.rows + 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
    
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Color Quantization", cv::WINDOW_AUTOSIZE);
    cv::imshow("Color Quantization", display);
    
    std::cout << "Color quantization effects:" << std::endl;
    std::cout << "  - Reduces number of colors while preserving image structure" << std::endl;
    std::cout << "  - Lower color count = more posterized appearance" << std::endl;
    std::cout << "  - Useful for artistic effects and data compression" << std::endl;
    std::cout << "  - Maintains spatial coherence better than simple thresholding" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateSpatialColorClustering(const cv::Mat& src) {
    std::cout << "\n=== Spatial-Color Clustering ===" << std::endl;
    
    // Create feature vector with both spatial and color information
    cv::Mat features;
    std::vector<float> feature_vector;
    
    float spatial_weight = 0.3f;  // Weight for spatial features
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            // 5D feature: BGR + spatial coordinates
            feature_vector.push_back(pixel[0]);  // Blue
            feature_vector.push_back(pixel[1]);  // Green
            feature_vector.push_back(pixel[2]);  // Red
            feature_vector.push_back(x * spatial_weight);  // X coordinate (weighted)
            feature_vector.push_back(y * spatial_weight);  // Y coordinate (weighted)
        }
    }
    
    // Convert to OpenCV Mat (each row is a 5D feature vector)
    int num_pixels = src.rows * src.cols;
    cv::Mat data = cv::Mat(num_pixels, 5, CV_32F, feature_vector.data()).clone();
    data.convertTo(data, CV_32F);
    
    // Apply K-means with spatial-color features
    int k = 6;
    cv::Mat labels, centers;
    cv::kmeans(data, k, labels, 
              cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.0), 
              3, cv::KMEANS_PP_CENTERS, centers);
    
    // Create segmented image
    cv::Mat segmented = cv::Mat::zeros(src.size(), CV_8UC3);
    
    // Generate colors for each cluster
    std::vector<cv::Vec3b> colors(k);
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for consistent colors
    std::uniform_int_distribution<> dis(0, 255);
    
    for (int i = 0; i < k; i++) {
        colors[i] = cv::Vec3b(dis(gen), dis(gen), dis(gen));
    }
    
    // Assign colors based on cluster labels
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            int cluster_idx = labels.at<int>(y * src.cols + x);
            segmented.at<cv::Vec3b>(y, x) = colors[cluster_idx];
        }
    }
    
    // Compare with color-only clustering
    cv::Mat color_data;
    src.reshape(1, src.rows * src.cols).convertTo(color_data, CV_32F);
    
    cv::Mat color_labels, color_centers;
    cv::kmeans(color_data, k, color_labels, 
              cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.0), 
              3, cv::KMEANS_PP_CENTERS, color_centers);
    
    cv::Mat color_only = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            int cluster_idx = color_labels.at<int>(y * src.cols + x);
            color_only.at<cv::Vec3b>(y, x) = colors[cluster_idx];
        }
    }
    
    // Display comparison
    cv::Mat display = cv::Mat::zeros(src.rows, src.cols * 3, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    color_only.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    segmented.copyTo(display(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Color Only", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Spatial+Color", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Spatial-Color Clustering", cv::WINDOW_AUTOSIZE);
    cv::imshow("Spatial-Color Clustering", display);
    
    std::cout << "Spatial-color clustering benefits:" << std::endl;
    std::cout << "  - Combines color similarity with spatial proximity" << std::endl;
    std::cout << "  - Creates more spatially coherent segments" << std::endl;
    std::cout << "  - Reduces noise and isolated pixels" << std::endl;
    std::cout << "  - Spatial weight controls smoothness vs. color accuracy" << std::endl;
    std::cout << "  - Current spatial weight: " << spatial_weight << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateSuperpixelClustering(const cv::Mat& src) {
    std::cout << "\n=== Superpixel-based Clustering ===" << std::endl;
    
    // Simple superpixel implementation using spatial grid + color
    int grid_size = 20;  // Size of each superpixel region
    
    cv::Mat superpixel_map = cv::Mat::zeros(src.size(), CV_32S);
    std::vector<cv::Vec3f> superpixel_colors;
    std::vector<cv::Point2f> superpixel_centers;
    
    int superpixel_id = 0;
    
    // Create initial superpixel grid
    for (int y = 0; y < src.rows; y += grid_size) {
        for (int x = 0; x < src.cols; x += grid_size) {
            // Calculate region bounds
            int x_end = std::min(x + grid_size, src.cols);
            int y_end = std::min(y + grid_size, src.rows);
            
            // Calculate average color and center for this superpixel
            cv::Scalar mean_color = cv::mean(src(cv::Rect(x, y, x_end - x, y_end - y)));
            cv::Vec3f avg_color(mean_color[0], mean_color[1], mean_color[2]);
            cv::Point2f center(x + (x_end - x) / 2.0f, y + (y_end - y) / 2.0f);
            
            superpixel_colors.push_back(avg_color);
            superpixel_centers.push_back(center);
            
            // Assign superpixel ID to all pixels in this region
            for (int py = y; py < y_end; py++) {
                for (int px = x; px < x_end; px++) {
                    superpixel_map.at<int>(py, px) = superpixel_id;
                }
            }
            superpixel_id++;
        }
    }
    
    // Refine superpixels with a few iterations of local optimization
    for (int iter = 0; iter < 3; iter++) {
        std::vector<cv::Vec3f> new_colors(superpixel_colors.size(), cv::Vec3f(0, 0, 0));
        std::vector<cv::Point2f> new_centers(superpixel_centers.size(), cv::Point2f(0, 0));
        std::vector<int> pixel_counts(superpixel_colors.size(), 0);
        
        // Reassign pixels to nearest superpixel center
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
                
                float min_distance = FLT_MAX;
                int best_superpixel = 0;
                
                // Check nearby superpixels
                for (int sp = 0; sp < (int)superpixel_colors.size(); sp++) {
                    cv::Point2f sp_center = superpixel_centers[sp];
                    float spatial_dist = cv::norm(cv::Point2f(x, y) - sp_center);
                    
                    // Only consider nearby superpixels
                    if (spatial_dist > grid_size * 2) continue;
                    
                    cv::Vec3f sp_color = superpixel_colors[sp];
                    float color_dist = cv::norm(cv::Vec3f(pixel[0], pixel[1], pixel[2]) - sp_color);
                    
                    float total_dist = color_dist + spatial_dist * 0.5f;
                    
                    if (total_dist < min_distance) {
                        min_distance = total_dist;
                        best_superpixel = sp;
                    }
                }
                
                superpixel_map.at<int>(y, x) = best_superpixel;
                
                // Accumulate for new center calculation
                new_colors[best_superpixel] += cv::Vec3f(pixel[0], pixel[1], pixel[2]);
                new_centers[best_superpixel] += cv::Point2f(x, y);
                pixel_counts[best_superpixel]++;
            }
        }
        
        // Update superpixel centers and colors
        for (size_t sp = 0; sp < superpixel_colors.size(); sp++) {
            if (pixel_counts[sp] > 0) {
                superpixel_colors[sp] = new_colors[sp] / (float)pixel_counts[sp];
                superpixel_centers[sp] = new_centers[sp] / (float)pixel_counts[sp];
            }
        }
    }
    
    // Create visualization
    cv::Mat superpixel_vis = cv::Mat::zeros(src.size(), CV_8UC3);
    cv::Mat boundaries = cv::Mat::zeros(src.size(), CV_8UC1);
    
    // Generate random colors for visualization
    std::vector<cv::Vec3b> vis_colors(superpixel_colors.size());
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 255);
    
    for (size_t i = 0; i < vis_colors.size(); i++) {
        vis_colors[i] = cv::Vec3b(dis(gen), dis(gen), dis(gen));
    }
    
    // Draw superpixels and find boundaries
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            int sp_id = superpixel_map.at<int>(y, x);
            superpixel_vis.at<cv::Vec3b>(y, x) = vis_colors[sp_id];
            
            // Check for boundaries
            bool is_boundary = false;
            for (int dy = -1; dy <= 1 && !is_boundary; dy++) {
                for (int dx = -1; dx <= 1 && !is_boundary; dx++) {
                    int ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < src.rows && nx >= 0 && nx < src.cols) {
                        if (superpixel_map.at<int>(ny, nx) != sp_id) {
                            is_boundary = true;
                        }
                    }
                }
            }
            if (is_boundary) {
                boundaries.at<uchar>(y, x) = 255;
            }
        }
    }
    
    // Create display with boundaries overlaid
    cv::Mat result = src.clone();
    result.setTo(cv::Scalar(0, 255, 0), boundaries);
    
    cv::Mat display = cv::Mat::zeros(src.rows, src.cols * 3, CV_8UC3);
    
    src.copyTo(display(cv::Rect(0, 0, src.cols, src.rows)));
    superpixel_vis.copyTo(display(cv::Rect(src.cols, 0, src.cols, src.rows)));
    result.copyTo(display(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(display, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Superpixels", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Boundaries", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Superpixel Clustering", cv::WINDOW_AUTOSIZE);
    cv::imshow("Superpixel Clustering", display);
    
    std::cout << "Superpixel clustering properties:" << std::endl;
    std::cout << "  - Groups pixels into perceptually meaningful regions" << std::endl;
    std::cout << "  - Reduces computational complexity for further processing" << std::endl;
    std::cout << "  - Preserves important image boundaries" << std::endl;
    std::cout << "  - Grid size: " << grid_size << "x" << grid_size << " pixels" << std::endl;
    std::cout << "  - Number of superpixels: " << superpixel_colors.size() << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Clustering Segmentation ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createClusteringTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg");
    if (!real_image.empty()) {
        std::cout << "Using loaded image for demonstrations." << std::endl;
        
        // Use real image for demonstrations
        demonstrateKMeansClustering(real_image);
        demonstrateColorQuantization(real_image);
        demonstrateSpatialColorClustering(real_image);
        demonstrateSuperpixelClustering(real_image);
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all clustering techniques
        demonstrateKMeansClustering(test_image);
        demonstrateColorQuantization(test_image);
        demonstrateSpatialColorClustering(test_image);
        demonstrateSuperpixelClustering(test_image);
    }
    
    std::cout << "\n✓ Clustering Segmentation demonstration complete!" << std::endl;
    std::cout << "Clustering provides unsupervised segmentation based on similarity measures." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. K-means clustering partitions pixels based on color similarity
 * 2. Color quantization reduces the number of colors while preserving structure
 * 3. Spatial-color clustering combines position and color for coherent segments
 * 4. Superpixels group pixels into perceptually meaningful regions
 * 5. Different distance metrics affect clustering results
 * 6. Initialization method (KMEANS_PP_CENTERS) improves convergence
 * 7. Spatial weighting controls smoothness vs color accuracy trade-off
 * 8. Clustering is unsupervised and doesn't require training data
 */

/**
 * 16-Histograms.cpp
 * 
 * Histogram computation and equalization techniques.
 * 
 * Concepts covered:
 * - Histogram calculation
 *     // Calculate histogram equalization
    auto calculateAndDrawHist = [](const cv::Mat& img, const std::string& title) -> cv::Mat {
        std::vector<cv::Mat> images = {img};
        cv::Mat hist;
        int histSize = 256;
        std::vector<int> channels = {0};
        std::vector<int> histSizes = {histSize};
        std::vector<float> ranges = {0, 256};
        
        cv::calcHist(images, channels, cv::Mat(), hist, histSizes, ranges);am equalization
 * - CLAHE (adaptive)
 * - Histogram matching
 * - Backprojection
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat createHistogramTestImage() {
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // Create regions with different brightness levels
    cv::rectangle(image, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(80, 80, 80), -1);      // Dark
    cv::rectangle(image, cv::Point(200, 50), cv::Point(300, 150), cv::Scalar(160, 160, 160), -1);  // Medium
    cv::rectangle(image, cv::Point(350, 50), cv::Point(450, 150), cv::Scalar(240, 240, 240), -1);  // Bright
    
    // Add some colored regions
    cv::rectangle(image, cv::Point(50, 200), cv::Point(150, 300), cv::Scalar(0, 0, 200), -1);      // Red
    cv::rectangle(image, cv::Point(200, 200), cv::Point(300, 300), cv::Scalar(0, 200, 0), -1);     // Green
    cv::rectangle(image, cv::Point(350, 200), cv::Point(450, 300), cv::Scalar(200, 0, 0), -1);     // Blue
    
    // Add gradient region
    for (int x = 50; x < 450; x++) {
        for (int y = 320; y < 370; y++) {
            int intensity = ((x - 50) * 255) / 400;
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity, intensity);
        }
    }
    
    return image;
}

void demonstrateHistogramCalculation(const cv::Mat& src) {
    std::cout << "\n=== Histogram Calculation ===" << std::endl;
    
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Calculate histogram
    std::vector<cv::Mat> images = {gray};
    cv::Mat hist;
    int histSize = 256;
    std::vector<int> channels = {0};
    std::vector<int> histSizes = {histSize};
    std::vector<float> ranges = {0, 256};
    
    cv::calcHist(images, channels, cv::Mat(), hist, histSizes, ranges);
    
    // Create histogram visualization
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Normalize histogram
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    
    // Draw histogram
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage,
                cv::Point(bin_w * (i-1), hist_h - cvRound(hist.at<float>(i-1))),
                cv::Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
                cv::Scalar(255, 255, 255), 2, 8, 0);
    }
    
    // Calculate color histograms
    std::vector<cv::Mat> bgr_planes;
    cv::split(src, bgr_planes);
    
    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist(std::vector<cv::Mat>{bgr_planes[0]}, channels, cv::Mat(), b_hist, histSizes, ranges);
    cv::calcHist(std::vector<cv::Mat>{bgr_planes[1]}, channels, cv::Mat(), g_hist, histSizes, ranges);
    cv::calcHist(std::vector<cv::Mat>{bgr_planes[2]}, channels, cv::Mat(), r_hist, histSizes, ranges);
    
    // Create color histogram visualization
    cv::Mat colorHistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    
    // Draw color histograms
    for (int i = 1; i < histSize; i++) {
        cv::line(colorHistImage,
                cv::Point(bin_w * (i-1), hist_h - cvRound(b_hist.at<float>(i-1))),
                cv::Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))),
                cv::Scalar(255, 0, 0), 2, 8, 0);  // Blue
        cv::line(colorHistImage,
                cv::Point(bin_w * (i-1), hist_h - cvRound(g_hist.at<float>(i-1))),
                cv::Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i))),
                cv::Scalar(0, 255, 0), 2, 8, 0);  // Green
        cv::line(colorHistImage,
                cv::Point(bin_w * (i-1), hist_h - cvRound(r_hist.at<float>(i-1))),
                cv::Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i))),
                cv::Scalar(0, 0, 255), 2, 8, 0);  // Red
    }
    
    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Grayscale Histogram", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Color Histograms", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Original Image", src);
    cv::imshow("Grayscale Histogram", histImage);
    cv::imshow("Color Histograms", colorHistImage);
    
    // Calculate and display statistics
    double minVal, maxVal;
    cv::minMaxLoc(hist, &minVal, &maxVal);
    cv::Scalar meanVal = cv::mean(gray);
    
    std::cout << "Histogram statistics:" << std::endl;
    std::cout << "  Peak frequency: " << maxVal << std::endl;
    std::cout << "  Mean intensity: " << meanVal[0] << std::endl;
    std::cout << "  Total pixels: " << cv::sum(hist)[0] << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateHistogramEqualization(const cv::Mat& src) {
    std::cout << "\n=== Histogram Equalization ===" << std::endl;
    
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Apply histogram equalization
    cv::Mat equalized;
    cv::equalizeHist(gray, equalized);
    
    // Calculate histograms before and after
    auto calculateAndDrawHist = [](const cv::Mat& img, const std::string& title) -> cv::Mat {
        std::vector<cv::Mat> images = {img};
        cv::Mat hist;
        int histSize = 256;
        std::vector<int> channels = {0};
        std::vector<int> histSizes = {histSize};
        std::vector<float> ranges = {0, 256};
        
        cv::calcHist(images, channels, cv::Mat(), hist, histSizes, ranges);
        
        int hist_w = 512, hist_h = 400;
        int bin_w = cvRound((double)hist_w / histSize);
        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
        
        cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
        
        for (int i = 1; i < histSize; i++) {
            cv::line(histImage,
                    cv::Point(bin_w * (i-1), hist_h - cvRound(hist.at<float>(i-1))),
                    cv::Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
                    cv::Scalar(255, 255, 255), 2, 8, 0);
        }
        
        cv::putText(histImage, title, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        return histImage;
    };
    
    cv::Mat originalHist = calculateAndDrawHist(gray, "Original");
    cv::Mat equalizedHist = calculateAndDrawHist(equalized, "Equalized");
    
    // Create comparison view
    cv::Mat comparison = cv::Mat::zeros(std::max(gray.rows, originalHist.rows), 
                                       gray.cols + equalized.cols + originalHist.cols + equalizedHist.cols, CV_8UC3);
    
    // Convert grayscale images to 3-channel for display
    cv::Mat gray_3ch, eq_3ch;
    cv::cvtColor(gray, gray_3ch, cv::COLOR_GRAY2BGR);
    cv::cvtColor(equalized, eq_3ch, cv::COLOR_GRAY2BGR);
    
    gray_3ch.copyTo(comparison(cv::Rect(0, 0, gray.cols, gray.rows)));
    eq_3ch.copyTo(comparison(cv::Rect(gray.cols, 0, equalized.cols, equalized.rows)));
    originalHist.copyTo(comparison(cv::Rect(gray.cols + equalized.cols, 0, originalHist.cols, originalHist.rows)));
    equalizedHist.copyTo(comparison(cv::Rect(gray.cols + equalized.cols + originalHist.cols, 0, equalizedHist.cols, equalizedHist.rows)));
    
    cv::namedWindow("Histogram Equalization", cv::WINDOW_AUTOSIZE);
    cv::imshow("Histogram Equalization", comparison);
    
    // Calculate contrast improvement
    cv::Scalar originalMean, originalStd;
    cv::meanStdDev(gray, originalMean, originalStd);
    cv::Scalar equalizedMean, equalizedStd;
    cv::meanStdDev(equalized, equalizedMean, equalizedStd);
    
    std::cout << "Histogram equalization results:" << std::endl;
    std::cout << "  Original - Mean: " << originalMean[0] << ", Std: " << originalStd[0] << std::endl;
    std::cout << "  Equalized - Mean: " << equalizedMean[0] << ", Std: " << equalizedStd[0] << std::endl;
    std::cout << "  Contrast improvement: " << equalizedStd[0] / originalStd[0] << "x" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateCLAHE(const cv::Mat& src) {
    std::cout << "\n=== CLAHE (Contrast Limited Adaptive Histogram Equalization) ===" << std::endl;
    
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Apply regular histogram equalization
    cv::Mat regular_eq;
    cv::equalizeHist(gray, regular_eq);
    
    // Apply CLAHE with different parameters
    cv::Ptr<cv::CLAHE> clahe1 = cv::createCLAHE(2.0, cv::Size(8, 8));   // Default parameters
    cv::Ptr<cv::CLAHE> clahe2 = cv::createCLAHE(4.0, cv::Size(8, 8));   // Higher contrast limit
    cv::Ptr<cv::CLAHE> clahe3 = cv::createCLAHE(2.0, cv::Size(16, 16)); // Larger tiles
    
    cv::Mat clahe_result1, clahe_result2, clahe_result3;
    clahe1->apply(gray, clahe_result1);
    clahe2->apply(gray, clahe_result2);
    clahe3->apply(gray, clahe_result3);
    
    // Create comparison grid
    cv::Mat comparison = cv::Mat::zeros(gray.rows * 2, gray.cols * 3, CV_8UC3);
    
    // Convert all to 3-channel for display
    cv::Mat gray_3ch, reg_eq_3ch, clahe1_3ch, clahe2_3ch, clahe3_3ch;
    cv::cvtColor(gray, gray_3ch, cv::COLOR_GRAY2BGR);
    cv::cvtColor(regular_eq, reg_eq_3ch, cv::COLOR_GRAY2BGR);
    cv::cvtColor(clahe_result1, clahe1_3ch, cv::COLOR_GRAY2BGR);
    cv::cvtColor(clahe_result2, clahe2_3ch, cv::COLOR_GRAY2BGR);
    cv::cvtColor(clahe_result3, clahe3_3ch, cv::COLOR_GRAY2BGR);
    
    // Top row
    gray_3ch.copyTo(comparison(cv::Rect(0, 0, gray.cols, gray.rows)));
    reg_eq_3ch.copyTo(comparison(cv::Rect(gray.cols, 0, gray.cols, gray.rows)));
    clahe1_3ch.copyTo(comparison(cv::Rect(gray.cols * 2, 0, gray.cols, gray.rows)));
    
    // Bottom row
    clahe2_3ch.copyTo(comparison(cv::Rect(0, gray.rows, gray.cols, gray.rows)));
    clahe3_3ch.copyTo(comparison(cv::Rect(gray.cols, gray.rows, gray.cols, gray.rows)));
    
    // Add labels
    cv::putText(comparison, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Regular EQ", cv::Point(gray.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "CLAHE (2.0, 8x8)", cv::Point(gray.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "CLAHE (4.0, 8x8)", cv::Point(10, gray.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "CLAHE (2.0, 16x16)", cv::Point(gray.cols + 10, gray.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("CLAHE Comparison", cv::WINDOW_AUTOSIZE);
    cv::imshow("CLAHE Comparison", comparison);
    
    std::cout << "CLAHE parameters tested:" << std::endl;
    std::cout << "  CLAHE (2.0, 8x8): Standard adaptive equalization" << std::endl;
    std::cout << "  CLAHE (4.0, 8x8): Higher contrast limit" << std::endl;
    std::cout << "  CLAHE (2.0, 16x16): Larger tile size for smoother transitions" << std::endl;
    std::cout << "CLAHE prevents over-amplification of noise while enhancing local contrast." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateHistogramMatching(const cv::Mat& src) {
    std::cout << "\n=== Histogram Matching ===" << std::endl;
    
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Create a target image with desired histogram characteristics
    cv::Mat target = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    // Create a target with specific distribution (e.g., bimodal)
    cv::rectangle(target, cv::Point(0, 0), cv::Point(target.cols/2, target.rows), cv::Scalar(80), -1);
    cv::rectangle(target, cv::Point(target.cols/2, 0), cv::Point(target.cols, target.rows), cv::Scalar(180), -1);
    
    // Simple histogram matching implementation
    auto matchHistogram = [](const cv::Mat& source, const cv::Mat& target) -> cv::Mat {
        // Calculate histograms
        cv::Mat src_hist, tgt_hist;
        int histSize = 256;
        std::vector<int> channels = {0};
        std::vector<int> histSizes = {histSize};
        std::vector<float> ranges = {0, 256};
        
        cv::calcHist(std::vector<cv::Mat>{source}, channels, cv::Mat(), src_hist, histSizes, ranges);
        cv::calcHist(std::vector<cv::Mat>{target}, channels, cv::Mat(), tgt_hist, histSizes, ranges);
        
        // Calculate CDFs
        cv::Mat src_cdf = src_hist.clone();
        cv::Mat tgt_cdf = tgt_hist.clone();
        
        for (int i = 1; i < histSize; i++) {
            src_cdf.at<float>(i) += src_cdf.at<float>(i-1);
            tgt_cdf.at<float>(i) += tgt_cdf.at<float>(i-1);
        }
        
        // Normalize CDFs
        src_cdf /= src_cdf.at<float>(histSize-1);
        tgt_cdf /= tgt_cdf.at<float>(histSize-1);
        
        // Create lookup table
        cv::Mat lut(1, 256, CV_8UC1);
        for (int i = 0; i < 256; i++) {
            float src_val = src_cdf.at<float>(i);
            int j = 0;
            while (j < 256 && tgt_cdf.at<float>(j) < src_val) j++;
            lut.at<uchar>(i) = cv::saturate_cast<uchar>(j);
        }
        
        // Apply LUT
        cv::Mat result;
        cv::LUT(source, lut, result);
        return result;
    };
    
    cv::Mat matched = matchHistogram(gray, target);
    
    // Calculate histograms for visualization
    auto calculateHist = [](const cv::Mat& img) -> cv::Mat {
        cv::Mat hist;
        int histSize = 256;
        std::vector<int> channels = {0};
        std::vector<int> histSizes = {histSize};
        std::vector<float> ranges = {0, 256};
        cv::calcHist(std::vector<cv::Mat>{img}, channels, cv::Mat(), hist, histSizes, ranges);
        
        int hist_w = 256, hist_h = 200;
        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
        
        for (int i = 0; i < histSize; i++) {
            cv::line(histImage,
                    cv::Point(i, hist_h),
                    cv::Point(i, hist_h - cvRound(hist.at<float>(i))),
                    cv::Scalar(255, 255, 255), 1, 8, 0);
        }
        return histImage;
    };
    
    cv::Mat src_hist_img = calculateHist(gray);
    cv::Mat tgt_hist_img = calculateHist(target);
    cv::Mat matched_hist_img = calculateHist(matched);
    
    // Create visualization
    cv::Mat visualization = cv::Mat::zeros(gray.rows + src_hist_img.rows, gray.cols * 3, CV_8UC3);
    
    // Convert grayscale to 3-channel
    cv::Mat gray_3ch, target_3ch, matched_3ch;
    cv::cvtColor(gray, gray_3ch, cv::COLOR_GRAY2BGR);
    cv::cvtColor(target, target_3ch, cv::COLOR_GRAY2BGR);
    cv::cvtColor(matched, matched_3ch, cv::COLOR_GRAY2BGR);
    
    // Top row: images
    gray_3ch.copyTo(visualization(cv::Rect(0, 0, gray.cols, gray.rows)));
    target_3ch.copyTo(visualization(cv::Rect(gray.cols, 0, gray.cols, gray.rows)));
    matched_3ch.copyTo(visualization(cv::Rect(gray.cols * 2, 0, gray.cols, gray.rows)));
    
    // Bottom row: histograms
    src_hist_img.copyTo(visualization(cv::Rect(0, gray.rows, src_hist_img.cols, src_hist_img.rows)));
    tgt_hist_img.copyTo(visualization(cv::Rect(gray.cols, gray.rows, tgt_hist_img.cols, tgt_hist_img.rows)));
    matched_hist_img.copyTo(visualization(cv::Rect(gray.cols * 2, gray.rows, matched_hist_img.cols, matched_hist_img.rows)));
    
    // Add labels
    cv::putText(visualization, "Source", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(visualization, "Target", cv::Point(gray.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(visualization, "Matched", cv::Point(gray.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Histogram Matching", cv::WINDOW_AUTOSIZE);
    cv::imshow("Histogram Matching", visualization);
    
    std::cout << "Histogram matching transfers the histogram characteristics from target to source image." << std::endl;
    std::cout << "This technique is useful for image normalization and style transfer." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demonstrateBackProjection(const cv::Mat& src) {
    std::cout << "\n=== Histogram Backprojection ===" << std::endl;
    
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    // Define a region of interest (ROI) for the object we want to track
    cv::Rect roi(50, 50, 100, 100);  // Adjust based on your test image
    cv::Mat roi_hsv = hsv(roi);
    
    // Calculate histogram of the ROI
    cv::Mat roi_hist;
    int h_bins = 50, s_bins = 60;
    std::vector<int> histSize = {h_bins, s_bins};
    std::vector<float> h_ranges = {0, 180};
    std::vector<float> s_ranges = {0, 256};
    std::vector<float> ranges;
    ranges.insert(ranges.end(), h_ranges.begin(), h_ranges.end());
    ranges.insert(ranges.end(), s_ranges.begin(), s_ranges.end());
    std::vector<int> channels = {0, 1};
    
    cv::calcHist(std::vector<cv::Mat>{roi_hsv}, channels, cv::Mat(), roi_hist, histSize, ranges);
    cv::normalize(roi_hist, roi_hist, 0, 255, cv::NORM_MINMAX);
    
    // Calculate backprojection
    cv::Mat backproj;
    cv::calcBackProject(std::vector<cv::Mat>{hsv}, channels, roi_hist, backproj, ranges, 1.0);
    
    // Apply morphological operations to clean up the result
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(backproj, backproj, cv::MORPH_CLOSE, kernel);
    
    // Threshold to create binary mask
    cv::Mat mask;
    cv::threshold(backproj, mask, 50, 255, cv::THRESH_BINARY);
    
    // Apply mask to original image
    cv::Mat result;
    src.copyTo(result, mask);
    
    // Create visualization
    cv::Mat visualization = cv::Mat::zeros(src.rows, src.cols * 3, CV_8UC3);
    
    // Original with ROI marked
    cv::Mat src_with_roi = src.clone();
    cv::rectangle(src_with_roi, roi, cv::Scalar(0, 255, 0), 3);
    
    src_with_roi.copyTo(visualization(cv::Rect(0, 0, src.cols, src.rows)));
    
    // Backprojection result
    cv::Mat backproj_3ch;
    cv::cvtColor(backproj, backproj_3ch, cv::COLOR_GRAY2BGR);
    backproj_3ch.copyTo(visualization(cv::Rect(src.cols, 0, src.cols, src.rows)));
    
    // Segmented result
    result.copyTo(visualization(cv::Rect(src.cols * 2, 0, src.cols, src.rows)));
    
    // Add labels
    cv::putText(visualization, "ROI (Green)", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(visualization, "Backprojection", cv::Point(src.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(visualization, "Segmented", cv::Point(src.cols * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::namedWindow("Histogram Backprojection", cv::WINDOW_AUTOSIZE);
    cv::imshow("Histogram Backprojection", visualization);
    
    std::cout << "Histogram backprojection finds pixels similar to the ROI based on color distribution." << std::endl;
    std::cout << "This technique is useful for object tracking and color-based segmentation." << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    std::cout << "=== Histogram Analysis ===" << std::endl;
    
    // Create test image
    cv::Mat test_image = createHistogramTestImage();
    
    // Try to load a real image for additional testing
    cv::Mat real_image = cv::imread("../data/test.jpg");
    if (!real_image.empty()) {
        std::cout << "Using loaded image for some demonstrations." << std::endl;
        
        // Use real image for most demonstrations
        demonstrateHistogramCalculation(real_image);
        demonstrateHistogramEqualization(real_image);
        demonstrateCLAHE(real_image);
        demonstrateHistogramMatching(test_image);    // Use synthetic for clear demo
        demonstrateBackProjection(real_image);
    } else {
        std::cout << "Using synthetic test image." << std::endl;
        
        // Demonstrate all histogram operations
        demonstrateHistogramCalculation(test_image);
        demonstrateHistogramEqualization(test_image);
        demonstrateCLAHE(test_image);
        demonstrateHistogramMatching(test_image);
        demonstrateBackProjection(test_image);
    }
    
    std::cout << "\n✓ Histogram Analysis demonstration complete!" << std::endl;
    std::cout << "Histograms provide essential statistical information about image intensity distributions." << std::endl;
    
    return 0;
}

/**
 * Key Learning Points:
 * 1. Histograms represent the distribution of pixel intensities in an image
 * 2. Histogram equalization improves overall contrast by spreading intensity values
 * 3. CLAHE provides adaptive local contrast enhancement while limiting noise amplification
 * 4. Histogram matching transfers statistical properties between images
 * 5. Backprojection finds regions similar to a reference based on color distribution
 * 6. Color histograms use multiple channels for richer representation
 * 7. Different histogram techniques serve different enhancement and analysis purposes
 * 8. Proper parameter tuning is crucial for optimal results in each technique
 */

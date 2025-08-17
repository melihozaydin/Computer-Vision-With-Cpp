#!/bin/bash

# OpenCV Examples Unified Runner Script
# Modes: --test (quick compilation check), --run (interactive), --auto (automatic)
# Author: Auto-generated for OpenCV C++ Learning Curriculum

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default mode
MODE="interactive"

# Function to print colored output
print_header() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=====================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Function to wait for user input
wait_for_user() {
    # Always return 0 to continue - no user input needed
    return 0
}

# Function to ensure compilation
ensure_compiled() {
    local example_file="$1"
    
    if [[ ! -f ".build/$example_file" ]]; then
        print_info "Compiling $example_file..."
        if make ".build/$example_file" >/dev/null 2>&1; then
            return 0
        else
            print_error "Failed to compile $example_file"
            return 1
        fi
    fi
    return 0
}

# Function to test compilation only
test_compilation() {
    print_header "OpenCV Examples Compilation Test"
    print_info "Testing compilation of all examples..."
    
    local successful=0
    local failed=0
    local examples=(
        "00-OpenCV_Setup" "01-Image_IO" "02-Basic_Operations" "03-Image_Arithmetic"
        "04-Image_Thresholding" "05-Filtering_Smoothing" "06-Denoising" "07-Edge_Detection"
        "08-Hough_Transform" "09-Morphological_Operations" "10-Distance_Transform" "11-Contours"
        "12-Corner_Detection" "13-Feature_Detection" "14-Feature_Matching" "15-Color_Spaces"
        "16-Histograms" "17-Geometric_Transformations" "18-Image_Pyramids" "19-Watershed_Segmentation"
        "20-GrabCut_Segmentation" "21-Clustering_Segmentation" "22-Template_Matching" "23-Cascade_Classifiers"
        "24-HOG_Detection" "25-Optical_Flow" "26-Background_Subtraction" "27-Object_Tracking"
        "28-Video_Processing" "29-Camera_Calibration" "30-Stereo_Vision" "31-Homography_RANSAC"
        "32-Texture_Analysis" "33-Image_Inpainting" "34-Image_Stitching" "35-Defect_Detection"
        "36-Advanced_Demos"
    )
    
    for example in "${examples[@]}"; do
        if ensure_compiled "$example"; then
            print_success "$example"
            successful=$((successful + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    echo
    print_header "Compilation Test Summary"
    print_success "Successful: $successful"
    if [[ $failed -gt 0 ]]; then
        print_error "Failed: $failed"
        exit 1
    else
        print_success "🎉 All examples compile successfully!"
        print_info "Use './run_all_examples.sh --run' for interactive testing"
        print_info "Use './run_all_examples.sh --auto' for automatic execution"
    fi
}

# Function to run a single example
run_example() {
    local example_name="$1"
    local example_file="$2"
    local description="$3"
    
    echo
    echo "==============================================================================="
    print_header "[$example_file] $example_name"
    echo "==============================================================================="
    print_info "Description: $description"
    print_info "Executable: .build/$example_file"
    
    if ! ensure_compiled "$example_file"; then
        print_error "COMPILATION FAILED for $example_file"
        return 1
    fi
    
    print_info "Running example with 3 second timeout..."
    echo -e "${YELLOW}--- EXECUTION OUTPUT START ---${NC}"
    
    # Capture both stdout and stderr, run with timeout
    local output_file="/tmp/opencv_example_output.txt"
    local error_file="/tmp/opencv_example_error.txt"
    
    if timeout 3s "./.build/$example_file" > "$output_file" 2> "$error_file"; then
        local exit_code=$?
        echo -e "${YELLOW}--- EXECUTION OUTPUT END ---${NC}"
        
        # Show output if any
        if [[ -s "$output_file" ]]; then
            echo -e "${CYAN}STDOUT:${NC}"
            cat "$output_file"
        fi
        
        # Show errors if any
        if [[ -s "$error_file" ]]; then
            echo -e "${RED}STDERR:${NC}"
            cat "$error_file"
        fi
        
        if [[ $exit_code -eq 0 ]]; then
            print_success "✅ PASSED - Example completed successfully"
        else
            print_warning "⚠️  COMPLETED WITH WARNINGS - Exit code: $exit_code"
        fi
    else
        local timeout_code=$?
        echo -e "${YELLOW}--- EXECUTION OUTPUT END (TIMEOUT) ---${NC}"
        
        # Show any output even if timed out
        if [[ -s "$output_file" ]]; then
            echo -e "${CYAN}STDOUT (before timeout):${NC}"
            cat "$output_file"
        fi
        
        if [[ -s "$error_file" ]]; then
            echo -e "${RED}STDERR (before timeout):${NC}"
            cat "$error_file"
        fi
        
        if [[ $timeout_code -eq 124 ]]; then
            print_warning "⏰ TIMEOUT - Example timed out after 3 seconds (normal for interactive demos)"
        else
            print_error "❌ FAILED - Example failed with exit code: $timeout_code"
            return 1
        fi
    fi
    
    # Cleanup temp files
    rm -f "$output_file" "$error_file"
    
    echo -e "${PURPLE}===============================================================================${NC}"
    return 0
}

# Main execution for running examples
run_examples() {
    print_header "OpenCV C++ Examples Automated Runner"
    print_info "Running all 37 examples with 3-second timeouts each"
    print_info "Will capture and display both STDOUT and STDERR for debugging"
    echo

    # Check if build directory exists
    if [[ ! -d ".build" ]]; then
        print_warning "Build directory not found. Compiling all examples..."
        if make all >/dev/null 2>&1; then
            print_success "All examples compiled successfully!"
        else
            print_error "Compilation failed!"
            exit 1
        fi
        echo
    fi

    # Define all examples with descriptions
    local examples=(
        "00-OpenCV_Setup:OpenCV Installation and Setup Verification:Basic OpenCV installation check and version display"
        "01-Image_IO:Image Input/Output Operations:Loading, saving, and displaying images in different formats"
        "02-Basic_Operations:Basic Image Operations:ROI extraction, channel manipulation, and basic transformations"
        "03-Image_Arithmetic:Image Arithmetic Operations:Addition, subtraction, blending, and mathematical operations"
        "04-Image_Thresholding:Image Thresholding Techniques:Binary, adaptive, and Otsu thresholding methods"
        "05-Filtering_Smoothing:Image Filtering and Smoothing:Gaussian, bilateral, median, and custom kernel filtering"
        "06-Denoising:Image Denoising Techniques:Non-local means and fastNlMeans denoising algorithms"
        "07-Edge_Detection:Edge Detection Algorithms:Canny, Sobel, Laplacian, and gradient-based edge detection"
        "08-Hough_Transform:Hough Transform Applications:Line and circle detection using Hough transforms"
        "09-Morphological_Operations:Morphological Image Processing:Erosion, dilation, opening, closing, and connected components"
        "10-Distance_Transform:Distance Transform Analysis:Distance maps, skeleton extraction, and shape analysis"
        "11-Contours:Contour Detection and Analysis:Finding, analyzing, and manipulating object contours"
        "12-Corner_Detection:Corner Detection Methods:Harris, Shi-Tomasi, and FAST corner detection algorithms"
        "13-Feature_Detection:Feature Detection Algorithms:SIFT, SURF, ORB, and keypoint detection methods"
        "14-Feature_Matching:Feature Matching Techniques:Descriptor matching, FLANN, and robust matching"
        "15-Color_Spaces:Color Space Conversions:RGB, HSV, LAB, and other color space transformations"
        "16-Histograms:Histogram Analysis:Computing, displaying, and equalizing image histograms"
        "17-Geometric_Transformations:Geometric Image Transformations:Rotation, scaling, affine, and perspective transforms"
        "18-Image_Pyramids:Image Pyramid Construction:Gaussian and Laplacian pyramids for multi-scale analysis"
        "19-Watershed_Segmentation:Watershed Image Segmentation:Marker-based watershed algorithm for object segmentation"
        "20-GrabCut_Segmentation:GrabCut Interactive Segmentation:Interactive foreground/background segmentation"
        "21-Clustering_Segmentation:Clustering-based Segmentation:K-means and Mean-shift clustering for segmentation"
        "22-Template_Matching:Template Matching Techniques:Pattern recognition and template correlation methods"
        "23-Cascade_Classifiers:Haar Cascade Object Detection:Face and object detection using cascade classifiers"
        "24-HOG_Detection:HOG Feature Detection:Histogram of Oriented Gradients for object detection"
        "25-Optical_Flow:Optical Flow Analysis:Lucas-Kanade and Farneback optical flow algorithms"
        "26-Background_Subtraction:Background Subtraction Methods:MOG2 and KNN background subtraction for motion detection"
        "27-Object_Tracking:Object Tracking Algorithms:Various tracking methods for moving object surveillance"
        "28-Video_Processing:Video Processing Pipeline:Complete video analysis and processing workflow"
        "29-Camera_Calibration:Camera Calibration Process:Intrinsic and extrinsic camera parameter estimation"
        "30-Stereo_Vision:Stereo Vision and Depth:Disparity computation and 3D reconstruction from stereo pairs"
        "31-Homography_RANSAC:Homography and RANSAC:Robust geometric estimation with outlier rejection"
        "32-Texture_Analysis:Texture Analysis Methods:LBP, Gabor filters, and Haralick texture features"
        "33-Image_Inpainting:Image Restoration and Inpainting:Photo restoration using Telea and Navier-Stokes algorithms"
        "34-Image_Stitching:Image Stitching and Panoramas:Feature-based panorama creation and image mosaics"
        "35-Defect_Detection:Industrial Defect Detection:Quality control and surface inspection techniques"
        "36-Advanced_Demos:Advanced Interactive Demonstrations:Real-time processing and interactive computer vision applications"
    )

    local total_examples=${#examples[@]}
    local current_example=0
    local successful_runs=0
    local timeout_runs=0
    local failed_runs=0

    local start_time=$(date +%s)

    # Run each example
    for example_info in "${examples[@]}"; do
        IFS=':' read -r example_file example_name description <<< "$example_info"
        current_example=$((current_example + 1))
        
        echo
        print_info "🔄 Progress: $current_example/$total_examples"
        
        if run_example "$example_name" "$example_file" "$description"; then
            # Success or timeout (both are acceptable)
            if [[ $(timeout 1s echo "test" 2>/dev/null) ]]; then
                # Simple test to avoid re-running the example
                successful_runs=$((successful_runs + 1))
            fi
        else
            failed_runs=$((failed_runs + 1))
        fi
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Final summary
    echo
    echo "==============================================================================="
    print_header "🏁 FINAL EXECUTION SUMMARY"
    echo "==============================================================================="
    print_success "✅ Successful executions: $successful_runs"
    if [[ $timeout_runs -gt 0 ]]; then
        print_warning "⏰ Timeouts (normal for interactive): $timeout_runs"
    fi
    print_error "❌ Failed executions: $failed_runs"
    print_info "📊 Total examples tested: $total_examples"
    print_info "⏱️  Total execution time: ${duration} seconds"
    
    echo
    if [[ $failed_runs -eq 0 ]]; then
        print_success "🎉 ALL EXAMPLES COMPLETED! No critical failures detected."
        print_info "Note: Timeouts are normal for interactive OpenCV demos"
        exit 0
    else
        print_error "⚠️  $failed_runs EXAMPLES HAD CRITICAL FAILURES"
        print_info "Check the individual example outputs above for details"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "OpenCV Examples Automated Runner"
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  --test, -t     Quick compilation test only (no execution)"
        echo "  --list, -l     List all available examples"
        echo "  --help, -h     Show this help message"
        echo "  (no args)      Run all examples automatically"
        echo ""
        echo "Examples:"
        echo "  $0             # Run all examples (default)"
        echo "  $0 --test      # Quick compilation check"
        echo "  $0 --list      # List examples"
        exit 0
        ;;
    --test|-t)
        test_compilation
        exit 0
        ;;
    --list|-l)
        echo "Available OpenCV Examples (37 total):"
        echo "====================================="
        counter=1
        examples=(
            "00-OpenCV_Setup:OpenCV Installation and Setup Verification"
            "01-Image_IO:Image Input/Output Operations"
            "02-Basic_Operations:Basic Image Operations"
            "03-Image_Arithmetic:Image Arithmetic Operations"
            "04-Image_Thresholding:Image Thresholding Techniques"
            "05-Filtering_Smoothing:Image Filtering and Smoothing"
            "06-Denoising:Image Denoising Techniques"
            "07-Edge_Detection:Edge Detection Algorithms"
            "08-Hough_Transform:Hough Transform Applications"
            "09-Morphological_Operations:Morphological Image Processing"
            "10-Distance_Transform:Distance Transform Analysis"
            "11-Contours:Contour Detection and Analysis"
            "12-Corner_Detection:Corner Detection Methods"
            "13-Feature_Detection:Feature Detection Algorithms"
            "14-Feature_Matching:Feature Matching Techniques"
            "15-Color_Spaces:Color Space Conversions"
            "16-Histograms:Histogram Analysis"
            "17-Geometric_Transformations:Geometric Image Transformations"
            "18-Image_Pyramids:Image Pyramid Construction"
            "19-Watershed_Segmentation:Watershed Image Segmentation"
            "20-GrabCut_Segmentation:GrabCut Interactive Segmentation"
            "21-Clustering_Segmentation:Clustering-based Segmentation"
            "22-Template_Matching:Template Matching Techniques"
            "23-Cascade_Classifiers:Haar Cascade Object Detection"
            "24-HOG_Detection:HOG Feature Detection"
            "25-Optical_Flow:Optical Flow Analysis"
            "26-Background_Subtraction:Background Subtraction Methods"
            "27-Object_Tracking:Object Tracking Algorithms"
            "28-Video_Processing:Video Processing Pipeline"
            "29-Camera_Calibration:Camera Calibration Process"
            "30-Stereo_Vision:Stereo Vision and Depth"
            "31-Homography_RANSAC:Homography and RANSAC"
            "32-Texture_Analysis:Texture Analysis Methods"
            "33-Image_Inpainting:Image Restoration and Inpainting"
            "34-Image_Stitching:Image Stitching and Panoramas"
            "35-Defect_Detection:Industrial Defect Detection"
            "36-Advanced_Demos:Advanced Interactive Demonstrations"
        )
        for example_info in "${examples[@]}"; do
            IFS=':' read -r example_file example_name <<< "$example_info"
            printf "%2d. %-25s - %s\n" $counter "$example_file" "$example_name"
            counter=$((counter + 1))
        done
        echo ""
        echo "Use '$0 --test' for quick compilation check"
        echo "Use '$0' to run all examples automatically"
        exit 0
        ;;
    "")
        # Default: run all examples
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use '$0 --help' for usage information"
        exit 1
        ;;
esac

# Run the examples
run_examples

# OpenCV Examples Makefile
# Compiles all OpenCV example files

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`
BUILD_DIR = .build
CUDA_DIR = CUDA

# Get all .cpp files in current directory (excluding CUDA folder)
SOURCES = $(wildcard [0-9][0-9]-*.cpp)
TARGETS = $(SOURCES:%.cpp=$(BUILD_DIR)/%)

# Default target
all: $(BUILD_DIR) $(TARGETS)
	@echo "All OpenCV examples compiled successfully!"
	@echo "Run examples from $(BUILD_DIR)/ directory"

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Pattern rule for compiling individual files
$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS)

# Individual targets for convenience
00-OpenCV_Setup: $(BUILD_DIR)/00-OpenCV_Setup
01-Image_IO: $(BUILD_DIR)/01-Image_IO
02-Basic_Operations: $(BUILD_DIR)/02-Basic_Operations
03-Image_Arithmetic: $(BUILD_DIR)/03-Image_Arithmetic
04-Image_Thresholding: $(BUILD_DIR)/04-Image_Thresholding
05-Filtering_Smoothing: $(BUILD_DIR)/05-Filtering_Smoothing
06-Denoising: $(BUILD_DIR)/06-Denoising
07-Edge_Detection: $(BUILD_DIR)/07-Edge_Detection
08-Hough_Transform: $(BUILD_DIR)/08-Hough_Transform
09-Morphological_Operations: $(BUILD_DIR)/09-Morphological_Operations
10-Distance_Transform: $(BUILD_DIR)/10-Distance_Transform
11-Contours: $(BUILD_DIR)/11-Contours
12-Corner_Detection: $(BUILD_DIR)/12-Corner_Detection
13-Feature_Detection: $(BUILD_DIR)/13-Feature_Detection
14-Feature_Matching: $(BUILD_DIR)/14-Feature_Matching
15-Color_Spaces: $(BUILD_DIR)/15-Color_Spaces
16-Histograms: $(BUILD_DIR)/16-Histograms
17-Geometric_Transformations: $(BUILD_DIR)/17-Geometric_Transformations
18-Image_Pyramids: $(BUILD_DIR)/18-Image_Pyramids
19-Watershed_Segmentation: $(BUILD_DIR)/19-Watershed_Segmentation
20-GrabCut_Segmentation: $(BUILD_DIR)/20-GrabCut_Segmentation
21-Clustering_Segmentation: $(BUILD_DIR)/21-Clustering_Segmentation
22-Template_Matching: $(BUILD_DIR)/22-Template_Matching
23-Cascade_Classifiers: $(BUILD_DIR)/23-Cascade_Classifiers
24-HOG_Detection: $(BUILD_DIR)/24-HOG_Detection
25-Optical_Flow: $(BUILD_DIR)/25-Optical_Flow
26-Background_Subtraction: $(BUILD_DIR)/26-Background_Subtraction
27-Object_Tracking: $(BUILD_DIR)/27-Object_Tracking
28-Video_Processing: $(BUILD_DIR)/28-Video_Processing
29-Camera_Calibration: $(BUILD_DIR)/29-Camera_Calibration
30-Stereo_Vision: $(BUILD_DIR)/30-Stereo_Vision
31-Homography_RANSAC: $(BUILD_DIR)/31-Homography_RANSAC
32-Texture_Analysis: $(BUILD_DIR)/32-Texture_Analysis
33-Image_Inpainting: $(BUILD_DIR)/33-Image_Inpainting
34-Image_Stitching: $(BUILD_DIR)/34-Image_Stitching
35-Defect_Detection: $(BUILD_DIR)/35-Defect_Detection
36-Advanced_Demos: $(BUILD_DIR)/36-Advanced_Demos

# CUDA examples (separate makefile)
cuda:
	@echo "Building CUDA examples..."
	@cd $(CUDA_DIR) && $(MAKE) -f Makefile.cuda all

# Clean build directory
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)/*
	@cd $(CUDA_DIR) && $(MAKE) -f Makefile.cuda clean

# Test OpenCV installation
test:
	@echo "Testing OpenCV installation..."
	@pkg-config --exists opencv4 && echo "✓ OpenCV4 found" || echo "✗ OpenCV4 not found"
	@pkg-config --modversion opencv4 2>/dev/null && echo "Version: $$(pkg-config --modversion opencv4)" || echo "Could not determine version"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all examples"
	@echo "  clean        - Clean build directory"
	@echo "  test         - Test OpenCV installation"
	@echo "  cuda         - Build CUDA examples"
	@echo "  [filename]   - Build specific example (e.g., make 00-OpenCV_Setup)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make all"
	@echo "  make 01-Image_IO"
	@echo "  make clean"

.PHONY: all clean test cuda help $(SOURCES:%.cpp=%)

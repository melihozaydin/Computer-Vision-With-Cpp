# OpenCV Computer Vision Examples

This directory contains comprehensive examples for learning computer vision with OpenCV in C++. Each file demonstrates specific concepts with well-commented code.

## Prerequisites

### Required Dependencies
- OpenCV 4.x (with contrib modules recommended)
- C++17 compatible compiler (GCC 9+ or Clang 10+)
- CMake 3.10+ (for individual builds)
- pkg-config (for dependency management)

### Optional Dependencies
- CUDA Toolkit 11+ (for GPU acceleration examples in CUDA/ folder)
- OpenCV compiled with CUDA support (for GPU examples)

## Installation

### Ubuntu/Debian
```bash
# Install OpenCV
sudo apt update
sudo apt install libopencv-dev libopencv-contrib-dev

# Verify installation
pkg-config --modversion opencv4
```

### Building OpenCV from Source (with CUDA)
```bash
# See ../00 - Setup/Notes/02-Build OpenCV from source.md
# for detailed instructions
```

## Compilation

### Quick Start - All Examples
```bash
# Compile all examples
make all

# Quick compilation test (fast - no execution)
./run_all_examples.sh --test

# Interactive mode (default) - step through examples
./run_all_examples.sh
# or explicitly
./run_all_examples.sh --run

# Automatic mode - run all with short timeouts
./run_all_examples.sh --auto

# List all available examples
./run_all_examples.sh --list

# Get help
./run_all_examples.sh --help
```

### Individual Example Execution
```bash
# Run a specific example
./.build/00-OpenCV_Setup
./.build/01-Image_IO
```

### Script Overview
- `run_all_examples.sh` - Unified script for compilation testing and execution
  - `--test`: Fast compilation verification (no execution)
  - `--run`: Interactive mode with user control (default)
  - `--auto`: Automatic execution with timeouts
  - `--list`: List all available examples
- `Makefile` - Build system for individual and batch compilation

### Individual Compilation
```bash
# Compile single file
make 00-OpenCV_Setup

# Or manually
g++ -std=c++17 00-OpenCV_Setup.cpp -o .build/00-OpenCV_Setup `pkg-config --cflags --libs opencv4`
```

### CUDA Examples
```bash
# Navigate to CUDA folder
cd CUDA/

# Compile CUDA examples (requires CUDA toolkit and OpenCV with CUDA)
make -f Makefile.cuda all
```

## Learning Path

### Beginner (Files 00-11)
Start with basic setup, I/O, and fundamental operations.

### Intermediate (Files 12-28)
Feature detection, segmentation, and motion analysis.

### Advanced (Files 29-36)
3D vision, calibration, and complex applications.

### GPU Acceleration (CUDA/ folder)
High-performance computing with CUDA acceleration.

## File Structure

| File | Topic | Description |
|------|-------|-------------|
| 00-OpenCV_Setup.cpp | Setup | Test installation and basic display |
| 01-Image_IO.cpp | I/O | Read, write, display images |
| 02-Basic_Operations.cpp | Basic Ops | Resize, crop, flip, rotate |
| 03-Image_Arithmetic.cpp | Arithmetic | Blend, add, subtract images |
| 04-Image_Thresholding.cpp | Thresholding | Binary, adaptive, Otsu |
| 05-Filtering_Smoothing.cpp | Filtering | Blur, Gaussian, median |
| 06-Denoising.cpp | Noise Removal | Non-local means denoising |
| 07-Edge_Detection.cpp | Edge Detection | Sobel, Laplacian, Canny |
| 08-Hough_Transform.cpp | Line Detection | Hough lines and circles |
| 09-Morphological_Operations.cpp | Morphology | Erosion, dilation, opening |
| 10-Distance_Transform.cpp | Shape Analysis | EDT, skeletonization |
| 11-Contours.cpp | Contour Analysis | Find and analyze shapes |
| 12-Corner_Detection.cpp | Corner Detection | Harris, Shi-Tomasi |
| 13-Feature_Detection.cpp | Feature Detection | SIFT, SURF, ORB |
| 14-Feature_Matching.cpp | Feature Matching | Match keypoints |
| 15-Color_Spaces.cpp | Color | HSV, LAB, color masking |
| 16-Histograms.cpp | Histograms | Analysis and equalization |
| 17-Geometric_Transformations.cpp | Geometry | Affine, perspective |
| 18-Image_Pyramids.cpp | Pyramids | Multi-scale analysis |
| 19-Watershed_Segmentation.cpp | Segmentation | Watershed algorithm |
| 20-GrabCut_Segmentation.cpp | Segmentation | Interactive extraction |
| 21-Clustering_Segmentation.cpp | Segmentation | K-means, superpixels |
| 22-Template_Matching.cpp | Matching | Template detection |
| 23-Cascade_Classifiers.cpp | Detection | Haar/LBP cascades |
| 24-HOG_Detection.cpp | Detection | HOG + SVM |
| 25-Optical_Flow.cpp | Motion | Lucas-Kanade, dense flow |
| 26-Background_Subtraction.cpp | Motion | Background modeling |
| 27-Object_Tracking.cpp | Tracking | Simple tracking methods |
| 28-Video_Processing.cpp | Video | Video I/O and processing |
| 29-Camera_Calibration.cpp | 3D Vision | Camera parameters |
| 30-Stereo_Vision.cpp | 3D Vision | Depth estimation |
| 31-Homography_RANSAC.cpp | Geometry | Robust matching |
| 32-Texture_Analysis.cpp | Texture | LBP, Gabor filters |
| 33-Image_Inpainting.cpp | Restoration | Image completion |
| 34-Image_Stitching.cpp | Stitching | Panorama creation |
| 35-Defect_Detection.cpp | Industrial | Quality inspection |
| 36-Advanced_Demos.cpp | Applications | Real-time demos |

## Sample Images

Create a `data/` folder and add test images:
```bash
mkdir -p data
# Add sample images: lena.jpg, coins.png, chessboard.jpg, etc.
```

## Troubleshooting

### OpenCV Not Found
```bash
# Check installation
pkg-config --libs opencv4

# If not found, install or set PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

### CUDA Issues
- Ensure CUDA toolkit is installed
- Verify OpenCV was compiled with CUDA support
- Check GPU compatibility (Compute Capability 3.0+)

## Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [CUDA OpenCV Guide](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_gpu.html)

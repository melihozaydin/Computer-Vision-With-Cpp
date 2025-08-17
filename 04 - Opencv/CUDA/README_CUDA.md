# OpenCV CUDA Examples

This directory contains OpenCV examples that demonstrate GPU acceleration using CUDA.

## Prerequisites

### Required
- CUDA Toolkit 11.0+ 
- OpenCV compiled with CUDA support
- NVIDIA GPU with Compute Capability 3.0+
- C++17 compatible compiler with CUDA support

### Verification
```bash
# Check CUDA installation
nvcc --version

# Check GPU
nvidia-smi

# Check OpenCV CUDA support
python3 -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

## Compilation

```bash
# Compile all CUDA examples
make -f Makefile.cuda all

# Compile individual example
make -f Makefile.cuda 00-CUDA_Setup

# Or manually
nvcc -std=c++17 00-CUDA_Setup.cpp -o .build/00-CUDA_Setup `pkg-config --cflags --libs opencv4`
```

## Performance Notes

- GPU acceleration is most beneficial for:
  - Large images (>1MP)
  - Repetitive operations
  - Parallel algorithms
  - Real-time processing

- CPU may be faster for:
  - Small images
  - Single operations
  - Complex sequential algorithms

## File Overview

| File | Description |
|------|-------------|
| 00-CUDA_Setup.cpp | CUDA device detection and basic operations |
| 01-CUDA_Basic_Operations.cpp | Memory transfer and basic arithmetic |
| 02-CUDA_Filtering.cpp | GPU-accelerated image filtering |
| 03-CUDA_Morphology.cpp | Morphological operations on GPU |
| 04-CUDA_Edge_Detection.cpp | Edge detection using GPU |
| 05-CUDA_Feature_Detection.cpp | Feature detection algorithms |
| 06-CUDA_Optical_Flow.cpp | Optical flow computation |
| 07-CUDA_Background_Subtraction.cpp | Background subtraction |
| 08-CUDA_Stereo_Matching.cpp | Stereo vision algorithms |
| 09-CUDA_Memory_Management.cpp | Efficient GPU memory usage |
| 10-CUDA_Performance_Comparison.cpp | CPU vs GPU benchmarks |

## Troubleshooting

### OpenCV without CUDA support
```bash
# Rebuild OpenCV with CUDA
cd opencv-build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN="6.1 7.5 8.6" \
      -D WITH_CUBLAS=ON \
      -D WITH_CUFFT=ON \
      -D WITH_NONFREE=ON \
      ..
make -j$(nproc)
sudo make install
```

### Memory Issues
- Reduce image size for testing
- Use cv::cuda::Stream for memory management
- Check available GPU memory with nvidia-smi

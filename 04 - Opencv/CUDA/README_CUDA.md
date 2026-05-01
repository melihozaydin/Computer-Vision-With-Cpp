# OpenCV CUDA Examples

This directory contains OpenCV C++ examples that use `cv::cuda` for GPU acceleration.

> Important: these examples require **both** CUDA toolkit (`nvcc`) and an OpenCV build with CUDA support.

See also:
- `00 - Setup/00-BuildSystem/Getting_Started_OpenCV.md`

## Quick Start (WSL/Ubuntu)

From this folder (`04 - Opencv/CUDA`):

```bash
# 1) Verify GPU is visible in WSL/Ubuntu
nvidia-smi

# 2) Verify CUDA toolkit compiler exists
nvcc --version

# 3) Verify OpenCV pkg-config is available
pkg-config --modversion opencv4

# 4) Build all examples
make -f Makefile.cuda all

# 5) Run one example
./.build/00-CUDA_Setup
```

If step (2) fails with `nvcc: command not found`, install CUDA toolkit in your Linux environment first.

## Quick Start (Docker, recommended when host OpenCV is CPU-only)

If your host/WSL OpenCV does not provide CUDA modules, run these examples in a CUDA-enabled container.

Container workflow:
- Mount this repo into `/workspace`
- Build examples with CMake using `OpenCV_DIR=/usr/local/lib/cmake/opencv4`
- Run binaries from `.container_build/build/bin`

If you use this repository's container tooling, check:
- `00 - Setup/00-BuildSystem/02-Containerized/Containerized_Dev.md`
- `00 - Setup/00-BuildSystem/02-Containerized/Opencv_CUDA_Docker/deploy.bat`

## Prerequisites

### Required
- NVIDIA GPU with supported driver
- CUDA Toolkit (11+ recommended) available in PATH (`nvcc`)
- OpenCV 4.x compiled with CUDA support
- `pkg-config` entry for `opencv4`
- C++17-capable CUDA toolchain

### Optional but recommended
- OpenCV contrib modules
- WSL2 + NVIDIA driver for CUDA on WSL (Windows users)

## Verify Environment

```bash
# CUDA compiler
nvcc --version

# GPU visibility
nvidia-smi

# OpenCV version used by this project
pkg-config --modversion opencv4

# Optional Python check (if python-opencv is installed)
python3 -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

## Build

```bash
# Build all CUDA examples
make -f Makefile.cuda all

# Build one target
make -f Makefile.cuda 00-CUDA_Setup

# Clean build outputs
make -f Makefile.cuda clean
```

The build system uses `Makefile.cuda` and writes binaries to `./.build/`.

## Run

```bash
# Example: CUDA setup / device check
./.build/00-CUDA_Setup

# Example: basic operations
./.build/01-CUDA_Basic_Operations
```

## Common Issues

### 1) `nvcc: command not found`
CUDA toolkit is not installed (or not in PATH) inside your Linux environment.

### 2) OpenCV found, but CUDA APIs fail at runtime
Your OpenCV may be a CPU-only build. Rebuild OpenCV with CUDA enabled.

### 3) `pkg-config` cannot find `opencv4`
Set `PKG_CONFIG_PATH` to the OpenCV `.pc` location.

Example:

```bash
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

### 4) No GPU detected in WSL
Confirm:
- Windows NVIDIA driver supports WSL CUDA
- WSL2 is used
- `nvidia-smi` works in WSL

## Rebuilding OpenCV with CUDA (reference)

```bash
cd opencv-build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_CUFFT=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_NONFREE=ON \
      ..
make -j$(nproc)
sudo make install
```

## File Overview

| File | Description |
|------|-------------|
| `00-CUDA_Setup.cpp` | CUDA device detection and setup checks |
| `01-CUDA_Basic_Operations.cpp` | Memory transfer and basic arithmetic |
| `02-CUDA_Filtering.cpp` | GPU-accelerated image filtering |
| `03-CUDA_Morphology.cpp` | Morphological operations on GPU |
| `04-CUDA_Edge_Detection.cpp` | Edge detection on GPU |
| `05-CUDA_Feature_Detection.cpp` | CUDA feature detection pipeline |
| `06-CUDA_Optical_Flow.cpp` | Optical flow computation |
| `07-CUDA_Background_Subtraction.cpp` | Background subtraction |
| `08-CUDA_Stereo_Matching.cpp` | Stereo matching |
| `09-CUDA_Memory_Management.cpp` | Efficient GPU memory usage |
| `10-CUDA_Performance_Comparison.cpp` | CPU vs GPU benchmark examples |

## Performance Notes

GPU is typically faster for:
- Large images and batches
- Repeated, data-parallel operations
- Real-time pipelines

CPU can be faster for:
- Small images
- One-off operations with high transfer overhead


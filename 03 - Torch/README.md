# Torch C++ Examples (LibTorch)

This folder contains practical C++ examples for image processing and deep learning using LibTorch (the C++ API for PyTorch) and OpenCV.


## 1. Install LibTorch (PyTorch C++ API)

### Recommended: Download Official Release (cxx11 ABI)

Download the latest LibTorch C++ distribution (with CUDA 12.9 support, cxx11 ABI) from the official PyTorch website:

https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip


Extract it to a location of your choice, for example:

For CPU:
(https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip)
For Cuda 12.9:
(https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip)
```bash
wget https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip
unzip libtorch-shared-with-deps-2.8.0+cu129.zip
```

This will create a `libtorch/` directory in your current folder.

#### Move libtorch to Makefile defined location (/~)

To make compilation easier, you can move the extracted `libtorch` to `/usr/include` (you will need sudo):

```bash
sudo mv /usr/include/libtorch ~/libtorch
```

The provided `Makefile` is already configured for this setup.

If you use a different location, update the `CXXFLAGS` and `LDFLAGS` in `Makefile` accordingly.

If you want the latest version or other CUDA versions, see the [official PyTorch C++ instructions](https://pytorch.org/cppdocs/installing.html).

## 2. Install OpenCV
```bash
sudo apt install libopencv-dev libopencv-highgui-dev
```

## 3. Build All Examples

Use the provided Makefile:

```bash
make
```



## 4. Run an Example

```bash
.build/00-Torch_Basics
```


## 5. Notes

- **Model Architecture Sync (LibTorch .pt):**
	- If you use `torch::save`/`torch::load` (LibTorch .pt files), the C++ model architecture (layer types, sizes, etc.) must match *exactly* in all C++ files that load the weights. If you change the model in one file (e.g., add batch norm, dropout, or more layers), you must update all other C++ files (such as demos) to match.
	- If the architectures do not match, loading will fail or give incorrect results.

- **TorchScript Export (Recommended for Deployment):**
	- TorchScript models exported from Python (`torch.jit.script` or `torch.jit.trace`) include both the architecture and weights. These can be loaded in C++ with `torch::jit::load` without redefining the architecture.
	- See below for instructions on exporting and using TorchScript models.

- **Device Auto-Detection:**
	- All C++ examples will automatically use CUDA if available, or fall back to CPU. The device used is printed at startup.

- **Model File Names:**
	- The desktop demo expects `mnist_cnn_best.pt` for LibTorch models, or `mnist_cnn.pt` for TorchScript models. Make sure you provide the correct file for your workflow.

- The file `20-Torch_Use_TorchVision.cpp` requires TorchVision C++ (not available via apt). By default, it is excluded from the build.
- To build TorchVision C++ from source:
	1. Clone the repo:
			```bash
			git clone --recursive https://github.com/pytorch/vision.git
			cd vision
			```
	2. Set environment variables to match your LibTorch install:
			```bash
			export Torch_DIR=$HOME/libtorch/share/cmake/Torch
			export CMAKE_PREFIX_PATH=$HOME/libtorch
			```
	3. Build with CMake:
			```bash
			mkdir build && cd build
			cmake -DCMAKE_BUILD_TYPE=Release ..
			make -j$(nproc)
			sudo make install
			```
	4. Add the install/include and install/lib paths to your Makefile's CXXFLAGS and LDFLAGS.
	5. Remove the filter-out for `20-Torch_Use_TorchVision.cpp` in the Makefile to enable building this example.
- For CUDA support, install the CUDA-enabled version of LibTorch from the official website.
- Example image files (e.g., `lenna.png`) should be placed in the `../images/` folder relative to this directory.


## Optional: TorchScript Model Export (Python)



### Why export TorchScript from Python?
TorchScript models exported from Python include both the model architecture and weights, so you do not need to redefine the model in C++. This is the most robust way to share models between Python and C++ and is recommended for deployment and demos.

### Python requirements
You need Python 3 and the following packages:
- torch (PyTorch)

Install with:
```bash
cd ~
sudo apt install python3-pip
sudo apt install python3.12-venv
python3 -m venv ~/python-env
echo 'source ~/python-env/bin/activate' >> ~/.bashrc
pip install torch torchvision
```

### How to export a TorchScript MNIST model
1. Train your MNIST model in Python and save the weights (e.g., `mnist_cnn_weights.pt`).
2. Use the provided script to export TorchScript:

```bash
python3 export_torchscript.py mnist_cnn_weights.pt mnist_cnn.pt
```

This will create `mnist_cnn.pt`, which can be loaded by the C++ desktop and server demos.

#### Example Python script (export_torchscript.py)
See `03 - Torch/export_torchscript.py` in this repo for a ready-to-use script.


### Usage in C++
All C++ MNIST inference demos (`32-MNIST_Desktop_Demo.cpp`, `31-TorchScript_ServerAPI.cpp`) expect `mnist_cnn.pt` (TorchScript) or `mnist_cnn_best.pt` (LibTorch) in the working directory, depending on your workflow. See above for architecture sync requirements.



## Optional: Crow HTTP Server Library (for TorchScript API example)

The example `31-TorchScript_ServerAPI.cpp` requires the [Crow](https://github.com/CrowCpp/crow) C++ HTTP server library. You can use either the header-only version or install the .deb package:

### Option 1: Header-only (recommended for most users)
1. Download the header:
	```bash
	wget https://raw.githubusercontent.com/CrowCpp/crow/master/include/crow_all.h -O crow_all.h
	```
2. In `31-TorchScript_ServerAPI.cpp`, change the include to:
	```cpp
	#include "crow_all.h"
	```
3. Add `-I.` (or the path to Crow) to your compile command or Makefile.

### Option 2: Install as a .deb package (Debian/Ubuntu)
1. Download the latest Crow .deb from the [releases page](https://github.com/CrowCpp/Crow/releases) or use:
	```bash
	wget https://github.com/CrowCpp/Crow/releases/download/v1.2.1.2/Crow-1.2.1-Linux.deb
	sudo apt install ./Crow-1.2.1-Linux.deb
	```
2. After install, Crow headers will be available system-wide.

Crow is only needed for the server API example. All other examples do not require it.



---

## References
- [PyTorch C++ API Documentation](https://pytorch.org/cppdocs/)
- [LibTorch Download](https://pytorch.org/get-started/locally/#start-locally)
- [OpenCV Documentation](https://docs.opencv.org/)

# Computer Vision with C++

C++ examples for computer vision, deep learning, GPU programming, and 3D point cloud processing.
The repo is organized as small numbered lesson folders with runnable examples.

---

## Quick Start

Each subject folder has a `run_all_examples.sh` script.
Open WSL, `cd` into the folder, and run the script shown below.

| Folder | What it covers | Recommended command |
|--------|---------------|---------------------|
| `01-CPP/` | C++ basics, OOP, pointers, STL | `./run_all_examples.sh` |
| `02 - Eigen/` | Matrix math, linear algebra | `./run_all_examples.sh` |
| `03 - Torch/` | LibTorch / PyTorch C++ | `./run_all_examples.sh --docker` |
| `04 - Opencv/` | CPU image processing | `./run_all_examples.sh` |
| `04 - Opencv/CUDA/` | CUDA-accelerated OpenCV | `./run_all_cuda_examples.sh` |
| `05 - CUDA/` | Pure CUDA kernels and training primitives | `./run_all_examples.sh` |
| `06 - PCL/` | 3D point cloud processing with PCL | `./run_all_examples.sh` |

All scripts support `--help`.
For `04 - Opencv/CUDA`, a PowerShell wrapper is also available: `./run_all_cuda_examples.ps1` from Windows PowerShell.

Need setup help first? See [`00 - Setup/00-BuildSystem/Getting_Started.md`](00%20-%20Setup/00-BuildSystem/Getting_Started.md).

---

## Prerequisites by folder

| Folder | Minimum requirement |
|--------|---------------------|
| `01-CPP` | `sudo apt install build-essential` |
| `02 - Eigen` | `sudo apt install build-essential libeigen3-dev` |
| `03 - Torch` | Docker (recommended) or local LibTorch + OpenCV |
| `04 - Opencv` | `sudo apt install libopencv-dev libopencv-contrib-dev` |
| `04 - Opencv/CUDA` | Docker + NVIDIA runtime (recommended) or local CUDA + CUDA-enabled OpenCV |
| `05 - CUDA` | Docker + NVIDIA runtime (recommended) or local CUDA Toolkit with `nvcc` |
| `06 - PCL` | `sudo apt install build-essential pkg-config libpcl-dev` |

---

## Folder Structure

```text
01-CPP/                   C++ fundamentals
02 - Eigen/               Eigen linear algebra
03 - Torch/               LibTorch: training, inference, TorchScript
04 - Opencv/              OpenCV image processing
  CUDA/                   OpenCV CUDA examples
05 - CUDA/                Pure CUDA kernels and training demos
06 - PCL/                 3D point cloud processing with PCL
00 - Setup/               Setup guides and Docker tooling
  00-BuildSystem/
    Getting_Started.md    start here for environment setup
    02-Containerized/     Dockerfiles and container workflows
data/                     sample datasets
Resources/                extra scripts and installers
```

---

## Learning Resources

- [LearnCpp.com](https://www.learncpp.com/)
- [PyTorch C++ API docs](https://pytorch.org/cppdocs/)
- [OpenCV docs](https://docs.opencv.org/)
- [PCL docs](https://pointclouds.org/documentation/)
- [CUDA docs](https://docs.nvidia.com/cuda/)

---

## Author

**Melih Sami Özaydın**

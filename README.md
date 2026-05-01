
# Computer Vision with C++

C++ examples for computer vision, deep learning, and GPU programming — covering C++ fundamentals, Eigen, LibTorch (PyTorch C++), OpenCV, and CUDA.

---

## Quick Start

Each folder has a `run_all_examples.sh` script. Open WSL, `cd` to the folder, and run it.

| Folder | What it covers | How to run |
|--------|---------------|------------|
| `01-CPP/` | C++ basics, OOP, pointers, STL | `./run_all_examples.sh` |
| `02 - Eigen/` | Matrix math, linear algebra | `./run_all_examples.sh` |
| `03 - Torch/` | LibTorch / PyTorch C++ deep learning | `./run_all_examples.sh --docker` |
| `04 - Opencv/` | Image processing, feature detection, segmentation | `./run_all_examples.sh` |
| `04 - Opencv/CUDA/` | GPU-accelerated OpenCV | `./run_all_cuda_examples.sh` |

All scripts support `--help`. **Windows users:** run via WSL2. For `04 - Opencv/CUDA` a PowerShell wrapper is also provided: `.\run_all_cuda_examples.ps1`.

Need setup help first? See [`00 - Setup/00-BuildSystem/Getting_Started.md`](00%20-%20Setup/00-BuildSystem/Getting_Started.md).

---

## Prerequisites by folder

| Folder | Minimum requirement |
|--------|---------------------|
| `01-CPP` | `sudo apt install build-essential` |
| `02 - Eigen` | `sudo apt install build-essential libeigen3-dev` |
| `03 - Torch` | Docker (recommended) or `~/libtorch` from [pytorch.org](https://pytorch.org/cppdocs/installing.html) |
| `04 - Opencv` | `sudo apt install libopencv-dev` |
| `04 - Opencv/CUDA` | Docker + NVIDIA runtime (recommended) or local CUDA + CUDA-enabled OpenCV |

---

## Folder Structure

```
01-CPP/                   C++ fundamentals
02 - Eigen/               Eigen linear algebra
03 - Torch/               LibTorch: CNN training, inference, TorchScript
04 - Opencv/              OpenCV image processing (37 examples)
	CUDA/                   GPU-accelerated OpenCV (11 examples)
00 - Setup/               Setup guides and Docker tooling
	00-BuildSystem/
		Getting_Started.md    start here for environment setup
		02-Containerized/     Dockerfiles and container workflows
data/                     Sample datasets
Resources/                Build scripts and installers
```

---

## Learning Resources

- [Computer Vision and OpenCV Tutorial in C++ (YouTube)](https://www.youtube.com/playlist?list=PLkmvobsnE0GHMmTF7GTzJnCISue1L9fJn)
- [LearnCpp.com](https://www.learncpp.com/)
- [PyTorch C++ API docs](https://pytorch.org/cppdocs/)
- [CUDA Course (GitHub)](https://github.com/Infatoshi/cuda-course)

---

## Author

**Melih Sami Özaydın**

---

For more details, see the README files in each subfolder.

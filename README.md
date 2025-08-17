
# Computer Vision with C++: Tutorials & Practical Examples

Comprehensive C++ tutorials and practical code for computer vision, image processing, and deep learning. This project covers:

- **C++ basics and advanced topics**
- **OpenCV** for image processing
- **Eigen** for linear algebra
- **LibTorch (PyTorch C++ API)** for deep learning
- **Docker-based environments** for reproducible builds

---

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Folder Summaries](#folder-summaries)
- [Learning Resources](#learning-resources)
- [Author](#author)

---

## Project Structure

```
01-CPP/           # C++ basics and advanced topics
02 - Eigen/       # Eigen linear algebra examples
03 - Torch/       # Deep learning with LibTorch
04 - Opencv/      # OpenCV image processing
00 - Setup/       # Docker, CMake, and environment setup
Resources/        # Extra resources and scripts
data/, images/    # Datasets and images
```

---

## Getting Started

### Prerequisites
- C++17 or newer compiler
- CMake, Make, or Ninja
- [OpenCV](https://opencv.org/), [Eigen](https://eigen.tuxfamily.org/), [LibTorch](https://pytorch.org/cppdocs/installing.html)
- (Optional) Docker for containerized builds

### Build & Run
See each subfolder's README for details. Example for a C++ file:

```bash
cd 01-CPP
make  # or use CMake as described in the folder
./01-Hello
```

For Docker-based builds, see `00 - Setup/Opencv_CUDA_Docker/` and related folders.

---

## Folder Summaries

- **01-CPP/**: C++ basics, OOP, pointers, STL, and more. Great for beginners and as a reference.
- **02 - Eigen/**: Matrix operations, arithmetic, and image transforms using Eigen.
- **03 - Torch/**: Deep learning and image processing with LibTorch. See [03 - Torch/README.md](03%20-%20Torch/README.md).
- **04 - Opencv/**: OpenCV image processing examples.
- **00 - Setup/**: Environment setup, Dockerfiles, and build notes.
- **Resources/**: Additional scripts, datasets, and supporting files.

---

## Learning Resources

- [Computer Vision and OpenCV Tutorial in C++ (YouTube)](https://www.youtube.com/playlist?list=PLkmvobsnE0GHMmTF7GTzJnCISue1L9fJn)
- [LearnCpp.com](https://www.learncpp.com/)
- [CUDA Course (GitHub)](https://github.com/Infatoshi/cuda-course)
- [MNIST CUDA (GitHub)](https://github.com/Infatoshi/mnist-cuda)
- [C++ pass arrays to functions (YouTube)](https://www.youtube.com/watch?v=VQSroKMqISE)
- [C++ overloaded functions explained (YouTube)](https://www.youtube.com/watch?v=LZd5LhfnYsk)
- [C++ array iteration for beginners (YouTube)](https://www.youtube.com/watch?v=a4P4ial8OgQ)
- [C++ Vectors and Dynamic Arrays (YouTube)](https://www.youtube.com/watch?v=OGQQK-hmOpE)

---

## Author

**Melih Sami Özaydın**

---

For more details, see the README files in each subfolder.

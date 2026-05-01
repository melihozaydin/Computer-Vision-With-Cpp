# Getting Started

Quick-start guide for every subject folder in this repo.
All examples target a **WSL2 + Ubuntu** workflow on Windows.
When local toolchains are heavy or fragile, a Docker mode is provided.

---

## 01-CPP — Basic C++

```bash
sudo apt install build-essential
cd "01-CPP"
./run_all_examples.sh
```

Pure C++; no Docker needed.

---

## 02 - Eigen — Linear Algebra

```bash
sudo apt install build-essential libeigen3-dev
cd "02 - Eigen"
./run_all_examples.sh
```

Eigen is header-only, so local setup is lightweight.

---

## 03 - Torch — LibTorch / PyTorch C++

### Recommended: Docker

```bash
cd "03 - Torch"
./run_all_examples.sh --docker
```

Docker mode installs an ABI-compatible `libtorch-dev` + `libopencv-dev` toolchain inside the container.
This is the most reliable path currently verified in this repo.

### Local

```bash
sudo apt install libopencv-dev
cd "03 - Torch"
./run_all_examples.sh --local --libtorch ~/libtorch
```

Use local mode only if you already have a working LibTorch install.

---

## 04 - Opencv — Computer Vision (CPU)

```bash
sudo apt install libopencv-dev libopencv-contrib-dev
cd "04 - Opencv"
./run_all_examples.sh
```

Optional Docker check path:

```bash
./run_all_examples.sh --docker --test
```

---

## 04 - Opencv/CUDA — GPU-Accelerated OpenCV

### Recommended: Docker

```bash
cd "04 - Opencv/CUDA"
./run_all_cuda_examples.sh
```

Windows PowerShell wrapper:

```powershell
.\run_all_cuda_examples.ps1
```

Use local mode only if WSL already has `nvcc` and a CUDA-enabled OpenCV build.

---

## 05 - CUDA — Pure CUDA Kernels and Training Primitives

### Recommended: Docker

```bash
cd "05 - CUDA"
./run_all_examples.sh
```

This folder contains pure `.cu` examples and uses `nvidia/cuda:12.3.2-devel-ubuntu22.04` by default.

### Local

```bash
cd "05 - CUDA"
./run_all_examples.sh --local
```

Requires a local CUDA Toolkit with `nvcc`.

---

## 06 - PCL — Point Cloud Library

### Local

```bash
sudo apt install build-essential pkg-config libpcl-dev
cd "06 - PCL"
./run_all_examples.sh
```

### Optional Docker

```bash
cd "06 - PCL"
./run_all_examples.sh --docker
```

Docker mode installs `libpcl-dev` in `ubuntu:22.04`.

---

## Quick Decision Table

| Subject | Recommended start | Docker available? |
|---------|-------------------|-------------------|
| `01-CPP` | `./run_all_examples.sh` | No |
| `02 - Eigen` | `./run_all_examples.sh` | No |
| `03 - Torch` | `./run_all_examples.sh --docker` | Yes |
| `04 - Opencv` | `./run_all_examples.sh` | Yes |
| `04 - Opencv/CUDA` | `./run_all_cuda_examples.sh` | Yes — default |
| `05 - CUDA` | `./run_all_examples.sh` | Yes — default |
| `06 - PCL` | `./run_all_examples.sh` | Yes |

---

## Common Pitfalls

1. **Windows asks which app should open a binary**  
   You are trying to run a Linux ELF binary from Windows directly. Use WSL.

2. **`nvcc` is missing**  
   Use Docker mode for CUDA folders, or install the CUDA Toolkit locally.

3. **Torch C++ build fails with ABI/linker errors**  
   Use `03 - Torch` Docker mode; it is the verified path.

4. **PCL headers or pkg-config modules are missing**  
   Install `libpcl-dev`, or use the PCL Docker mode.

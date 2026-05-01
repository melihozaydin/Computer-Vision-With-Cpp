# Getting Started

Quick-start guide for every subject folder in this repo.
All examples target a **WSL2 + Ubuntu** environment on Windows.
Run scripts are provided in each folder; Docker modes avoid needing to install heavy dependencies locally.

---

## 01-CPP — Basic C++

**Dependencies:** `g++` (standard in any Ubuntu/Debian WSL).

```bash
# One-time: install compiler if missing
sudo apt install build-essential

# Build + run all examples
cd "01-CPP"
./run_all_examples.sh          # local WSL (default)
./run_all_examples.sh --help   # see all options
```

No Docker mode needed — pure C++, no external libraries.

---

## 02 - Eigen — Linear Algebra

**Dependencies:** Eigen is header-only, just needs the dev package.

```bash
# One-time
sudo apt install build-essential libeigen3-dev

# Build + run all examples
cd "02 - Eigen"
./run_all_examples.sh          # local WSL (default)
./run_all_examples.sh --help
```

No Docker mode needed — header-only library.

---

## 03 - Torch — LibTorch / PyTorch C++

**Dependencies:** LibTorch (large, ~2 GB) + OpenCV.

```bash
# Option A — Local WSL (requires ~/libtorch installed)
cd "03 - Torch"
./run_all_examples.sh                    # local, uses ~/libtorch
./run_all_examples.sh --libtorch /path   # override libtorch path

# Option B — Docker (no local LibTorch install needed)
./run_all_examples.sh --docker                         # pytorch/pytorch:latest
./run_all_examples.sh --docker --image pytorch/pytorch:latest
./run_all_examples.sh --help
```

**Local setup:** Download LibTorch from https://pytorch.org/cppdocs/installing.html and extract to `~/libtorch`.
See `03 - Torch/README.md` for full instructions.

**Docker:** Uses `pytorch/pytorch:latest` which ships LibTorch at its Python package path.
OpenCV is installed via `apt` inside the container on first run (~1 min).
GPU is used automatically if the NVIDIA Docker runtime is available.

---

## 04 - Opencv — Computer Vision (CPU)

**Dependencies:** OpenCV 4.x.

```bash
# One-time
sudo apt install libopencv-dev libopencv-contrib-dev

# Build + run all 37 examples
cd "04 - Opencv"
./run_all_examples.sh                  # local WSL (default)
./run_all_examples.sh --docker         # inside Docker, no host OpenCV needed
./run_all_examples.sh --test           # compile-check only
./run_all_examples.sh --help
```

Most examples use `cv::imshow` or camera input; they will time out in headless mode — that is expected.

---

## 04 - Opencv/CUDA — GPU-Accelerated OpenCV

**Dependencies:** CUDA toolkit + OpenCV with CUDA support.

```bash
# Recommended: Docker (host WSL may lack nvcc or CUDA-enabled OpenCV)
cd "04 - Opencv/CUDA"
./run_all_cuda_examples.sh             # Docker mode (default)
./run_all_cuda_examples.sh --local     # local WSL CUDA toolchain

# Windows PowerShell wrapper
.\run_all_cuda_examples.ps1
.\run_all_cuda_examples.ps1 -Help
```

**Why Docker is the default here:** Host WSL often has GPU visibility (`nvidia-smi`) but lacks `nvcc`
or a CUDA-enabled OpenCV build. The Docker path uses `thecanadianroot/opencv-cuda:latest`.

Verified: 11/11 CUDA examples pass inside Docker.
See `04 - Opencv/CUDA/README_CUDA.md` for full documentation.

---

## Quick Decision Table

| Subject | Recommended start | Docker available? |
|---------|-------------------|-------------------|
| `01-CPP` | `./run_all_examples.sh` | No (unnecessary) |
| `02 - Eigen` | `./run_all_examples.sh` | No (unnecessary) |
| `03 - Torch` | `./run_all_examples.sh --docker` | Yes (`pytorch/pytorch`) |
| `04 - Opencv` | `./run_all_examples.sh` | Yes (same CUDA image) |
| `04 - Opencv/CUDA` | `./run_all_cuda_examples.sh` | Yes — default |

---

## Common Pitfalls

1. **"Choose an app" dialog on Windows** — You are running a Linux ELF binary from PowerShell. Always run via WSL.
2. **GPU visible but CUDA build fails** — `nvidia-smi` works but `nvcc` is missing or OpenCV is CPU-only. Use Docker.
3. **`pkg-config opencv4` works but CUDA APIs fail** — Installed OpenCV is a CPU-only build. Use Docker for CUDA examples.
4. **LibTorch build fails** — Check `~/libtorch` exists and the Makefile path matches. Use `--docker` to skip local setup.


This guide aligns the project setup with the actual run methods currently working in this repository.

## Recommended Paths

### Method A — OpenCV examples (CPU path, WSL)
Use this for `04 - Opencv` non-CUDA examples.

1. Enter WSL and go to the OpenCV folder.
2. Run existing binaries under `04 - Opencv/.build` (Linux ELF binaries).
3. If they are missing, rebuild in WSL with the folder `Makefile`.

Notes:
- Running these ELF binaries directly from Windows PowerShell can trigger “choose an app” prompts.
- Run them via WSL to avoid that.

---

### Method B — CUDA examples (recommended now: Docker + NVIDIA runtime)
Use this for `04 - Opencv/CUDA`.

Why this path:
- Host WSL may have GPU visibility (`nvidia-smi`) but still lack `nvcc` or CUDA-enabled OpenCV headers/libs.
- Container path gives a reproducible CUDA-enabled OpenCV toolchain.

#### One-time image pull
Use a CUDA+OpenCV image (example used successfully in this repo):
- `thecanadianroot/opencv-cuda:latest`

#### Build + run inside container
- Mount repo to `/workspace`
- Work from `/workspace/04 - Opencv/CUDA`
- Configure with CMake and `OpenCV_DIR=/usr/local/lib/cmake/opencv4`
- Build binaries into `.container_build/build/bin`
- Run with timeouts for automated verification

Current verified result:
- 11/11 CUDA examples built and ran in container (`ok=11, timeout=0, fail=0`).

---

### Method C — Native CUDA in WSL (advanced)
Use this only if you want host-native CUDA builds without containers.

Required:
- `nvcc` available in WSL
- OpenCV built with CUDA support in WSL
- CUDA headers like `opencv2/cudaimgproc.hpp` available from active OpenCV install

If `nvcc` is missing or CUDA OpenCV modules are unavailable, use Method B.

## Quick Decision Checklist

- Need fastest reliable CUDA run now? → **Method B**
- Need non-CUDA OpenCV lessons? → **Method A**
- Need fully native CUDA toolchain in WSL? → **Method C**

## Common Pitfalls

1. **GPU visible but CUDA build fails**
   - `nvidia-smi` works, but `nvcc` is missing, or OpenCV is CPU-only.

2. **`pkg-config opencv4` works but CUDA APIs fail**
   - Active OpenCV package may not include CUDA modules.

3. **Windows app picker appears for binaries**
   - You are trying to run Linux ELF binaries directly in Windows shell.

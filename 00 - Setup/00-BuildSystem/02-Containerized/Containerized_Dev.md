# Containerized Development Environment (OpenCV / CUDA)

This document describes the container workflow that matches this repo's current OpenCV CUDA execution path.

For an overview of all run methods (WSL CPU, Docker CUDA, native WSL CUDA), see:

- `00 - Setup/00-BuildSystem/Getting_Started_OpenCV.md`

## Prerequisites

- Docker Desktop (or Docker Engine)
- NVIDIA Container Toolkit / Docker GPU runtime available
- NVIDIA driver compatible with container CUDA runtime

## Recommended CUDA Workflow

### Option 1: Use an existing CUDA+OpenCV image

Use a known image that has CUDA-enabled OpenCV preinstalled, then mount this repository as `/workspace`.

Inside container:
- Build from `04 - Opencv/CUDA`
- Configure with `OpenCV_DIR=/usr/local/lib/cmake/opencv4`
- Run binaries from `.container_build/build/bin`

### Option 2: Build your own image from this repo

Folder:
- `00 - Setup/00-BuildSystem/02-Containerized/Opencv_CUDA_Docker/`

Build image from its `Dockerfile`, then run a container with:
- `--gpus all`
- repo volume mounted to `/workspace`
- working directory set to `/workspace/04 - Opencv/CUDA`

## Development Workflow

- Edit code on host in VS Code.
- Build/run inside container shell.
- Keep outputs in repo-local build folders (e.g., `.container_build/`) for reproducibility.

## Notes

- If CUDA examples fail on host WSL due to missing `nvcc` or CPU-only OpenCV, use container path.
- If `pkg-config opencv4` is unavailable in container, prefer CMake + `OpenCV_DIR`.
- Linux binaries should be run in Linux/container environments, not directly from Windows shell.

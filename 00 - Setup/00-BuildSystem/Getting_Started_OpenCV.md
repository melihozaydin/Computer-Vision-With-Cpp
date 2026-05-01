# OpenCV Getting Started (CPU + CUDA)

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

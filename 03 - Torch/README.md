# Torch C++ Examples (LibTorch)

This folder contains practical C++ examples for deep learning and image-processing workflows using LibTorch and, for some examples, OpenCV.

## Quick start

### Recommended: Docker

Docker mode is the easiest verified path in this repo.
It installs an ABI-compatible `libtorch-dev` + `libopencv-dev` toolchain inside the container and then builds the examples there.

```bash
cd "03 - Torch"
./run_all_examples.sh --docker
./run_all_examples.sh --docker --build-only
./run_all_examples.sh --help
```

Notes:
- Docker image: `pytorch/pytorch:latest`
- The script installs `build-essential`, `pkg-config`, `libopencv-dev`, and `libtorch-dev` inside the container when needed.
- Current Docker build mode is **CPU-oriented** because Ubuntu `libtorch-dev` does not ship `libtorch_cuda`.

### Local WSL

Use local mode if you already have a LibTorch install at `~/libtorch` (or another path you pass via `--libtorch`).

```bash
sudo apt install libopencv-dev
cd "03 - Torch"
./run_all_examples.sh --local
./run_all_examples.sh --local --libtorch ~/libtorch
```

## Included examples

| File | Topic |
|------|-------|
| `00-Torch_Basics.cpp` | Tensor creation and device basics |
| `01-Torch_Image_Load.cpp` | OpenCV image loading into tensors |
| `02-Torch_Tensor_Manipulation.cpp` | Shape manipulation and tensor ops |
| `03-Torch_Image_Normalization.cpp` | Preprocessing pipeline |
| `04-Torch_Augmentation.cpp` | Simple augmentation flow |
| `05-Torch_CUDA_Device_Management.cpp` | CUDA availability / device reporting |
| `10-Torch_Model_Inference.cpp` | Inference with a saved model |
| `11-Torch_Shape_Debug.cpp` | Shape inspection helpers |
| `12-Torch_Train_CNN.cpp` | Basic CNN training |
| `13-Torch_Train_CNN_MNIST.cpp` | MNIST training |
| `14-Torch_Advanced_Training.cpp` | Early stopping + LR decay |
| `21-Torch_Custom_Dataset.cpp` | Custom dataset example |
| `30-MNIST_Desktop_Demo.cpp` | Desktop MNIST inference demo |
| `30-Torch_Transfer_Learning.cpp` | Transfer-learning example |
| `31-TorchScript_ServerAPI.cpp` | Optional TorchScript server example |
| `32-MNIST_Desktop_Demo.cpp` | Alternate desktop demo |

## Optional examples excluded from the default build

Two examples are intentionally excluded from the default `make` / `run_all_examples.sh` build to keep the standard workflow self-contained:

- `20-Torch_Use_TorchVision.cpp`
  - Requires TorchVision C++.
- `31-TorchScript_ServerAPI.cpp`
  - Requires Crow / `crow.h`.

## Notes on models and data

- `13-Torch_Train_CNN_MNIST.cpp` expects the MNIST dataset under `./data/MNIST/raw`.
  The runner skips it automatically when the dataset is missing.
- The MNIST desktop demos expect either:
  - `mnist_cnn_best.pt` for a native LibTorch model, or
  - `mnist_cnn.pt` for a TorchScript model.
- If you use `torch::save` / `torch::load`, the C++ model architecture must match exactly across training and loading code.

## Manual build

```bash
make
make clean
```

If you keep LibTorch somewhere other than `~/libtorch`, update `Makefile` or use the runner with `--libtorch /path/to/libtorch`.

## References

- [PyTorch C++ API docs](https://pytorch.org/cppdocs/)
- [LibTorch install guide](https://pytorch.org/cppdocs/installing.html)
- [OpenCV docs](https://docs.opencv.org/)

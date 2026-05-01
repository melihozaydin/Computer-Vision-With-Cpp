# Pure CUDA Examples

This folder contains self-contained CUDA C++ examples focused on GPU fundamentals and pure-CUDA training primitives.
No OpenCV, Eigen, or LibTorch required.

## Included examples

| File | Topic |
|------|-------|
| `00-CUDA_Setup.cu` | Query CUDA devices and verify the runtime |
| `01-Vector_Add.cu` | Basic kernel launch + memory transfer |
| `02-Matrix_Multiply.cu` | Tiled matrix multiplication with shared memory |
| `03-Activation_And_Softmax.cu` | ReLU + single-row softmax kernels |
| `10-Linear_Regression_SGD.cu` | Pure CUDA gradient descent for $y = wx + b$ |
| `11-Logistic_Regression_Training.cu` | Binary classifier training with CUDA kernels |

## Quick start

### Recommended: Docker

```bash
cd "05 - CUDA"
./run_all_examples.sh                  # docker mode by default
./run_all_examples.sh --build-only
./run_all_examples.sh --help
```

Default Docker image:
- `nvidia/cuda:12.3.2-devel-ubuntu22.04`

### Local toolchain

Requires:
- NVIDIA driver
- CUDA Toolkit with `nvcc`
- `build-essential`

```bash
cd "05 - CUDA"
./run_all_examples.sh --local
./run_all_examples.sh --local --build-only
```

## Build manually

```bash
make all
make clean
```

## Notes

- These examples are deliberately small and educational rather than fully optimized.
- The training examples use full-batch gradient descent to keep the kernels easy to read.
- Docker mode is recommended when your host WSL does not have `nvcc` installed.
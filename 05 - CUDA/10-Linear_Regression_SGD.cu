#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at line " << __LINE__ << std::endl; \
        return 1; \
    } \
} while (0)

__global__ void linear_regression_grad_kernel(const float* x, const float* y, float* grad_w, float* grad_b, float w, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float pred = w * x[i] + b;
        float err = pred - y[i];
        atomicAdd(grad_w, (2.0f / n) * err * x[i]);
        atomicAdd(grad_b, (2.0f / n) * err);
    }
}

int main() {
    const int n = 1024;
    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = -1.0f + 2.0f * i / static_cast<float>(n - 1);
        y[i] = 3.0f * x[i] + 2.0f;
    }

    float *d_x = nullptr, *d_y = nullptr, *d_grad_w = nullptr, *d_grad_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_w, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    float w = 0.0f, b = 0.0f;
    const float lr = 0.1f;
    const int epochs = 200;
    const int block = 256;
    const int grid = (n + block - 1) / block;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        CUDA_CHECK(cudaMemset(d_grad_w, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_b, 0, sizeof(float)));
        linear_regression_grad_kernel<<<grid, block>>>(d_x, d_y, d_grad_w, d_grad_b, w, b, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float grad_w = 0.0f, grad_b = 0.0f;
        CUDA_CHECK(cudaMemcpy(&grad_w, d_grad_w, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost));

        w -= lr * grad_w;
        b -= lr * grad_b;

        if ((epoch + 1) % 50 == 0) {
            std::cout << "epoch=" << (epoch + 1) << " w=" << w << " b=" << b << std::endl;
        }
    }

    std::cout << "Final parameters: w=" << w << ", b=" << b << std::endl;

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_grad_w); cudaFree(d_grad_b);
    return 0;
}

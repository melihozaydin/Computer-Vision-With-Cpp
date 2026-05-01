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

__device__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void logistic_grad_kernel(const float* x1, const float* x2, const float* y,
                                     float* grad_w1, float* grad_w2, float* grad_b,
                                     float w1, float w2, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float z = w1 * x1[i] + w2 * x2[i] + b;
        float pred = sigmoidf(z);
        float err = pred - y[i];
        atomicAdd(grad_w1, err * x1[i] / n);
        atomicAdd(grad_w2, err * x2[i] / n);
        atomicAdd(grad_b, err / n);
    }
}

int main() {
    const int n = 1024;
    std::vector<float> x1(n), x2(n), y(n);
    for (int i = 0; i < n; ++i) {
        float a = (i % 32) / 31.0f;
        float b = (i / 32) / 31.0f;
        x1[i] = a * 2.0f - 1.0f;
        x2[i] = b * 2.0f - 1.0f;
        y[i] = (x1[i] + x2[i] > 0.0f) ? 1.0f : 0.0f;
    }

    float *d_x1 = nullptr, *d_x2 = nullptr, *d_y = nullptr;
    float *d_gw1 = nullptr, *d_gw2 = nullptr, *d_gb = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x1, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x2, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw1, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw2, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x1, x1.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, x2.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    float w1 = 0.0f, w2 = 0.0f, b0 = 0.0f;
    const float lr = 1.0f;
    const int epochs = 200;
    const int block = 256;
    const int grid = (n + block - 1) / block;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        CUDA_CHECK(cudaMemset(d_gw1, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gw2, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gb, 0, sizeof(float)));
        logistic_grad_kernel<<<grid, block>>>(d_x1, d_x2, d_y, d_gw1, d_gw2, d_gb, w1, w2, b0, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float gw1 = 0.0f, gw2 = 0.0f, gb = 0.0f;
        CUDA_CHECK(cudaMemcpy(&gw1, d_gw1, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&gw2, d_gw2, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&gb, d_gb, sizeof(float), cudaMemcpyDeviceToHost));

        w1 -= lr * gw1;
        w2 -= lr * gw2;
        b0 -= lr * gb;
    }

    int correct = 0;
    for (int i = 0; i < n; ++i) {
        float p = 1.0f / (1.0f + std::exp(-(w1 * x1[i] + w2 * x2[i] + b0)));
        correct += ((p > 0.5f) == (y[i] > 0.5f));
    }

    std::cout << "Trained logistic regression: w1=" << w1 << ", w2=" << w2 << ", b=" << b0
              << ", accuracy=" << (100.0f * correct / n) << "%" << std::endl;

    cudaFree(d_x1); cudaFree(d_x2); cudaFree(d_y); cudaFree(d_gw1); cudaFree(d_gw2); cudaFree(d_gb);
    return 0;
}

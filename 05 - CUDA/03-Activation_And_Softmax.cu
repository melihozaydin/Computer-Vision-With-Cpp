#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at line " << __LINE__ << std::endl; \
        return 1; \
    } \
} while (0)

__global__ void relu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ void softmax_row_kernel(const float* logits, float* probs, int cols) {
    if (blockIdx.x > 0) return; // single-row demo
    float max_val = logits[0];
    for (int i = 1; i < cols; ++i) max_val = max(max_val, logits[i]);

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float e = expf(logits[i] - max_val);
        probs[i] = e;
        sum += e;
    }
    for (int i = 0; i < cols; ++i) probs[i] /= sum;
}

int main() {
    std::vector<float> activations = {-2.0f, -0.5f, 0.25f, 1.0f, 2.5f};
    std::vector<float> logits = {1.0f, 2.0f, 0.5f, -1.0f};
    std::vector<float> probs(logits.size(), 0.0f);

    float *d_act = nullptr, *d_logits = nullptr, *d_probs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_act, activations.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, logits.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs, probs.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_act, activations.data(), activations.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));

    relu_kernel<<<1, 32>>>(d_act, static_cast<int>(activations.size()));
    softmax_row_kernel<<<1, 1>>>(d_logits, d_probs, static_cast<int>(logits.size()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(activations.data(), d_act, activations.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(probs.data(), d_probs, probs.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "ReLU output:";
    for (float v : activations) std::cout << ' ' << v;
    std::cout << "\nSoftmax output:";
    for (float v : probs) std::cout << ' ' << v;
    std::cout << std::endl;

    cudaFree(d_act); cudaFree(d_logits); cudaFree(d_probs);
    return 0;
}

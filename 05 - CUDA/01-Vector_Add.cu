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

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    std::vector<float> h_a(n, 1.25f), h_b(n, 2.75f), h_c(n, 0.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_error = std::max(max_error, std::fabs(h_c[i] - 4.0f));
    }

    std::cout << "Vector add complete. max_error=" << max_error << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at line " << __LINE__ << std::endl; \
        return 1; \
    } \
} while (0)

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::cout << "CUDA device count: " << device_count << std::endl;
    if (device_count == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global memory: " << std::fixed << std::setprecision(2)
                  << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads/block: " << prop.maxThreadsPerBlock << std::endl;
    }

    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "\nSelected device 0 successfully." << std::endl;
    return 0;
}

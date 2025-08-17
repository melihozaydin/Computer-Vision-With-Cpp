// 14-Torch_CUDA_Device_Management.cpp
// Comprehensive example: device management, tensor/model transfer, CUDA checks
#include <torch/torch.h>
#include <iostream>

int main() {
    // 1. Check CUDA availability
    bool cuda_available = torch::cuda::is_available();
    std::cout << "CUDA available: " << std::boolalpha << cuda_available << std::endl;

    // 2. List all CUDA devices
    int num_devices = torch::cuda::device_count();
    std::cout << "Number of CUDA devices: " << num_devices << std::endl;
    // Note: torch::cuda::device_properties is not available in most LibTorch builds.
    // This is because the C++ API does not always expose all CUDA runtime features, especially in prebuilt binaries.
    // For more device info, use the Python API or the nvidia-smi tool.
    for (int i = 0; i < num_devices; ++i) {
        std::cout << "  Device " << i << std::endl;
    }

    // 3. Set device (default: 0 if available, else CPU)
    torch::Device device = cuda_available ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    std::cout << "Using device: " << device.str() << std::endl;

    // 4. Create tensors on CPU and move to CUDA
    torch::Tensor cpu_tensor = torch::rand({3, 3});
    std::cout << "cpu_tensor.device: " << cpu_tensor.device() << std::endl;
    torch::Tensor cuda_tensor = cpu_tensor.to(device);
    std::cout << "cuda_tensor.device: " << cuda_tensor.device() << std::endl;

    // 5. Perform operations on CUDA (if available)
    if (cuda_available) {
        torch::Tensor a = torch::ones({2,2}, device);
        torch::Tensor b = torch::eye(2, device);
        torch::Tensor c = a + b;
        std::cout << "a + b on CUDA:\n" << c << std::endl;
    }

    // 6. Model to CUDA
    struct NetImpl : torch::nn::Module {
        torch::nn::Linear fc{nullptr};
        NetImpl() { fc = register_module("fc", torch::nn::Linear(4, 2)); }
        torch::Tensor forward(torch::Tensor x) { return fc(x); }
    };
    using Net = std::shared_ptr<NetImpl>;
    Net model = std::make_shared<NetImpl>();
    model->to(device); // Move model to device
    std::cout << "Model moved to: " << device.str() << std::endl;

    // 7. Training step on CUDA (if available)
    if (cuda_available) {
        torch::Tensor x = torch::rand({5,4}, device);
        torch::Tensor y = torch::rand({5,2}, device);
        auto output = model->forward(x);
        auto loss = torch::mse_loss(output, y);
        std::cout << "Loss on CUDA: " << loss.item<float>() << std::endl;
    }

    // 8. Move tensor/model back to CPU
    torch::Tensor back_to_cpu = cuda_tensor.to(torch::kCPU);
    std::cout << "Moved tensor back to: " << back_to_cpu.device() << std::endl;
    model->to(torch::kCPU);
    std::cout << "Model moved back to CPU." << std::endl;

    // 9. Check device of parameters
    for (const auto& p : model->parameters()) {
        std::cout << "Parameter device: " << p.device() << std::endl;
    }

    // 10. Device mismatch lesson (no runtime error, just a comment):
    // -------------------------------------------------------------
    // Lesson: All tensors and models involved in an operation must be on the same device.
    // For example, the following would throw an error if uncommented:
    //
    // torch::Tensor cpu_x = torch::rand({2,2});
    // torch::Tensor cuda_y = torch::rand({2,2}, device);
    // auto z = cpu_x + cuda_y; // ERROR: Expected all tensors to be on the same device
    //
    // Always use .to(device) to ensure device consistency:
    // auto z = cpu_x.to(device) + cuda_y;
    //
    // This is one of the most common sources of bugs in CUDA code!

    return 0;
}

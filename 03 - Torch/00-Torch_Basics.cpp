// 00-Torch_Basics.cpp
// Basic usage of LibTorch (PyTorch C++ API): tensor creation, types, and operations
// Compile with: g++ 00-Torch_Basics.cpp -o 00-Torch_Basics -I/path/to/libtorch/include -I/path/to/libtorch/include/torch/csrc/api/include -L/path/to/libtorch/lib -ltorch -lc10 -Wl,-rpath,/path/to/libtorch/lib -std=c++17
// Make sure to adjust the include and lib paths to your LibTorch installation.

#include <torch/torch.h>
#include <iostream>

int main() {
    // Create a tensor filled with zeros
    torch::Tensor zeros = torch::zeros({3, 3});
    std::cout << "Zeros tensor:\n" << zeros << std::endl;

    // Create a tensor filled with ones
    torch::Tensor ones = torch::ones({2, 4});
    std::cout << "Ones tensor:\n" << ones << std::endl;

    // Create a tensor with random values
    torch::Tensor rand = torch::rand({2, 2});
    std::cout << "Random tensor:\n" << rand << std::endl;

    // Create a tensor from data
    torch::Tensor data = torch::tensor({{1, 2}, {3, 4}});
    std::cout << "Data tensor:\n" << data << std::endl;

    // Tensor operations
    torch::Tensor ones_same = torch::ones({3, 3});
    torch::Tensor sum = zeros + ones_same;
    std::cout << "Sum of zeros and ones (same shape):\n" << sum << std::endl;

    torch::Tensor mul = data * 2;
    std::cout << "Data tensor multiplied by 2:\n" << mul << std::endl;

    // Tensor properties
    std::cout << "Shape of data tensor: ";
    for (auto s : data.sizes()) std::cout << s << " ";
    std::cout << std::endl;
    std::cout << "Data type: " << data.dtype() << std::endl;

    // Move tensor to CUDA if available
    if (torch::cuda::is_available()) {
        auto data_cuda = data.to(torch::kCUDA);
        std::cout << "Tensor on CUDA:\n" << data_cuda << std::endl;
    } else {
        std::cout << "CUDA not available.\n";
    }

    return 0;
}

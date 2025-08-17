// 02-Torch_Tensor_Manipulation.cpp
// Demonstrates slicing, reshaping, permuting, and other tensor manipulations in LibTorch
#include <torch/torch.h>
#include <iostream>

int main() {
	torch::Tensor t = torch::arange(0, 24).reshape({2, 3, 4});
	std::cout << "Original tensor (2x3x4):\n" << t << std::endl;
	// Slicing
	auto slice = t.index({0, torch::indexing::Slice(), torch::indexing::Slice(1, 3)});
	std::cout << "Slice [0, :, 1:3]:\n" << slice << std::endl;
	// Reshape
	auto flat = t.reshape({-1});
	std::cout << "Flattened tensor:\n" << flat << std::endl;
	// Permute
	auto perm = t.permute({2, 0, 1});
	std::cout << "Permuted tensor (4x2x3):\n" << perm << std::endl;
	// Concatenate
	auto cat = torch::cat({t, t}, 0);
	std::cout << "Concatenated along dim 0:\n" << cat.sizes() << std::endl;
	return 0;
}

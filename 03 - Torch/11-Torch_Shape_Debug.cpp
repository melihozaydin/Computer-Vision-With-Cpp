// 11-Torch_Shape_Debug.cpp
// Print tensor shapes, debug shape mismatches, and use asserts
#include <torch/torch.h>
#include <iostream>

int main() {
	torch::Tensor t = torch::rand({2, 3, 4});
	std::cout << "Tensor shape: ";
	for (auto s : t.sizes()) std::cout << s << " ";
	std::cout << std::endl;
	// Shape assertion
	TORCH_CHECK(t.sizes() == torch::IntArrayRef({2,3,4}), "Shape mismatch!");
	// Example: shape mismatch
	try {
		auto bad = t.reshape({3,8});
	} catch (const c10::Error& e) {
		std::cerr << "Caught error: " << e.what() << std::endl;
	}
	return 0;
}

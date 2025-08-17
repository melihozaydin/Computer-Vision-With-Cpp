// 20-Torch_Use_TorchVision.cpp
// Use TorchVision C++ API for datasets and transforms
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <iostream>

int main() {
	// Example: Use torchvision transforms
	auto transform = torchvision::transforms::Compose({
		torchvision::transforms::Resize({32,32}),
		torchvision::transforms::ToTensor(),
		torchvision::transforms::Normalize({0.5,0.5,0.5},{0.5,0.5,0.5})
	});
	// Example: Use CIFAR10 dataset
	auto dataset = torchvision::datasets::CIFAR10("../data", torchvision::datasets::CIFAR10::Mode::kTrain).map(transform);
	std::cout << "Loaded CIFAR10 dataset, size: " << dataset.size().value() << std::endl;
	return 0;
}

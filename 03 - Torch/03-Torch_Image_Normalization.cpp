// 03-Torch_Image_Normalization.cpp
// Normalize and denormalize image tensors for deep learning workflows
#include <torch/torch.h>
#include <iostream>

int main() {
	// Simulate an image tensor (CHW, float32, 0-1)
	torch::Tensor img = torch::rand({3, 224, 224});
	// Mean and std for normalization (e.g., ImageNet)
	torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}).view({3,1,1});
	torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}).view({3,1,1});
	// Normalize
	torch::Tensor norm = (img - mean) / std;
	std::cout << "Normalized tensor stats: mean=" << norm.mean().item<float>() << ", std=" << norm.std().item<float>() << std::endl;
	// Denormalize
	torch::Tensor denorm = norm * std + mean;
	std::cout << "Denormalized tensor stats: mean=" << denorm.mean().item<float>() << ", std=" << denorm.std().item<float>() << std::endl;
	return 0;
}

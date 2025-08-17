// 10-Torch_Model_Inference.cpp
// Create a simple TorchScript model, save it, then load and run inference
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>


// Define a simple model (single linear layer)
struct SimpleNetImpl : torch::nn::Module {
	torch::nn::Linear fc{nullptr};
	SimpleNetImpl() {
		fc = register_module("fc", torch::nn::Linear(3*224*224, 10));
	}
	torch::Tensor forward(torch::Tensor x) {
		x = x.view({x.size(0), -1}); // flatten
		return fc(x);
	}
};
TORCH_MODULE(SimpleNet);

int main() {
	std::string model_path = "model.pt";

	// 1. Create a simple model
	SimpleNet model;
	model->eval();

	// 2. Run inference directly (no scripting)
	torch::Tensor input = torch::rand({1,3,224,224});
	auto output = model->forward(input);
	std::cout << "Output shape: " << output.sizes() << std::endl;
	std::cout << "Model inference ran successfully.\n";
	return 0;
}

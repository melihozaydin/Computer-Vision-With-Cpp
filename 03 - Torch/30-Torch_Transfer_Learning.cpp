// 22-Torch_Transfer_Learning.cpp
// Transfer learning demo: create, save, load, and run a real TorchScript ResNet18 model
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

int main() {
	std::string model_path = "resnet18_scripted.pt";

	// If model file does not exist, call Python to export a real TorchScript ResNet18
	std::ifstream f(model_path);
	if (!f.good()) {
		std::cout << "Exporting ResNet18 TorchScript model from Python...\n";
		int ret = std::system("~/python-env/bin/python -c 'import torch; import torchvision.models as m; m=m.resnet18(pretrained=True); m.eval(); s=torch.jit.script(m); s.save(\"resnet18_scripted.pt\")'");
		if (ret != 0) {
			std::cerr << "Failed to export ResNet18 TorchScript model from Python.\n";
			return 1;
		}
		std::cout << "Model exported: " << model_path << std::endl;
	}

	// Load the model (TorchScript)
	torch::jit::script::Module model;
	try {
		model = torch::jit::load(model_path);
	} catch (const c10::Error& e) {
		std::cerr << "Error loading model.\n";
		return -1;
	}
	std::cout << "Model loaded from " << model_path << std::endl;

	// Run inference with dummy input
	torch::Tensor input = torch::rand({1,3,224,224});
	auto output = model.forward({input}).toTensor();
	std::cout << "Output shape: " << output.sizes() << std::endl;

	// Freeze all parameters (no gradient updates)
	for (auto param : model.parameters()) param.set_requires_grad(false);
	std::cout << "All parameters frozen (no gradients will be computed)." << std::endl;

	// Note: Replacing layers and fine-tuning requires advanced scripting and is best done in Python.
	return 0;
}

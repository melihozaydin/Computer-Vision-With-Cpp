// 13-Torch_Train_CNN_MNIST.cpp
// Full workflow: load MNIST, train, test, and save a CNN model
#include <torch/torch.h>
#include <iostream>

struct Net : torch::nn::Module {
	Net() {
		conv1 = register_module("conv1", torch::nn::Conv2d(1, 8, 3));
		fc1 = register_module("fc1", torch::nn::Linear(8*26*26, 10));
	}
	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(conv1->forward(x));
		x = x.view({x.size(0), -1});
		x = fc1->forward(x);
		return x;
	}
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::Linear fc1{nullptr};
};

int main() {
	// Download MNIST if needed
	auto train_dataset = torch::data::datasets::MNIST("./data/MNIST/raw").map(torch::data::transforms::Stack<>());
	auto train_loader = torch::data::make_data_loader(train_dataset, 64);
	Net net;
	torch::optim::SGD optimizer(net.parameters(), 0.01);
	for (int epoch = 0; epoch < 2; ++epoch) {
		size_t batch_idx = 0;
		for (auto& batch : *train_loader) {
			auto data = batch.data, target = batch.target;
			auto output = net.forward(data);
			auto loss = torch::nn::functional::cross_entropy(output, target);
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			if (batch_idx++ % 100 == 0)
				std::cout << "Epoch " << epoch << ", Batch " << batch_idx << ", Loss: " << loss.item<float>() << std::endl;
		}
	}
	// Save model
	torch::save(std::make_shared<Net>(net), "mnist_cnn.pt");
	std::cout << "Model saved as mnist_cnn.pt" << std::endl;
	return 0;
}

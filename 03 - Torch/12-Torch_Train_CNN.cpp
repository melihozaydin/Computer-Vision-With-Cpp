// 12-Torch_Train_CNN.cpp
// Train a simple CNN on random data (end-to-end workflow)
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
	Net net;
	torch::optim::SGD optimizer(net.parameters(), 0.01);
	for (int epoch = 0; epoch < 3; ++epoch) {
		auto data = torch::rand({16,1,28,28});
		auto target = torch::randint(0,10,{16});
		auto output = net.forward(data);
		auto loss = torch::nn::functional::cross_entropy(output, target);
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
		std::cout << "Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
	}
	return 0;
}

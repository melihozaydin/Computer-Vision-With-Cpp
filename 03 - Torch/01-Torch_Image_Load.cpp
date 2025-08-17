// 01-Torch_Image_Load.cpp
// Load an image using OpenCV and convert it to a Torch tensor
// Requires OpenCV and LibTorch

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
	std::string img_path = "./images/lenna.png"; // Change path as needed
	cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cerr << "Failed to load image: " << img_path << std::endl;
		return 1;
	}
	std::cout << "Loaded image: " << img.cols << "x" << img.rows << std::endl;
	// Convert to float and scale
	img.convertTo(img, CV_32F, 1.0/255);
	// Convert to torch tensor (HWC -> CHW)
	auto tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32);
	tensor = tensor.permute({2, 0, 1}).clone(); // CHW
	std::cout << "Tensor shape: " << tensor.sizes() << std::endl;
	return 0;
}

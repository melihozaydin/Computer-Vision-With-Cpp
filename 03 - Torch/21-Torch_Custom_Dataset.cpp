// 21-Torch_Custom_Dataset.cpp
// Implement a custom dataset class for loading images from folders
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

struct ImageFolderDataset : torch::data::datasets::Dataset<ImageFolderDataset> {
	std::vector<std::string> files;
	ImageFolderDataset(const std::string& folder) {
		for (const auto& entry : fs::directory_iterator(folder)) {
			if (entry.is_regular_file()) files.push_back(entry.path().string());
		}
	}
	torch::data::Example<> get(size_t idx) override {
		cv::Mat img = cv::imread(files[idx], cv::IMREAD_COLOR);
		img.convertTo(img, CV_32F, 1.0/255);
		auto tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32);
		tensor = tensor.permute({2,0,1}).clone();
		return {tensor, torch::tensor(0)}; // Dummy label
	}
	torch::optional<size_t> size() const override { return files.size(); }
};

int main() {
	ImageFolderDataset dataset("./images");
	std::cout << "Loaded " << *dataset.size() << " images." << std::endl;
	auto sample = dataset.get(0);
	std::cout << "Sample tensor shape: " << sample.data.sizes() << std::endl;
	return 0;
}

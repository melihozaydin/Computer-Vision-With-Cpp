// 04-Torch_Augmentation.cpp
// Basic data augmentation: flipping, cropping, rotating tensors/images


#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// --- Visualization helpers ---
// Converts a CHW float tensor [0,1] to a cv::Mat for display. Optionally resizes to out_h x out_w.
// Shows a grid of images with labels using OpenCV.
cv::Mat tensor_to_mat(const torch::Tensor& t, int out_h = 128, int out_w = 128) {
	torch::Tensor t_disp = t.detach().cpu();
	if (t_disp.sizes()[1] != out_h || t_disp.sizes()[2] != out_w) {
		t_disp = torch::nn::functional::interpolate(t_disp.unsqueeze(0),
			torch::nn::functional::InterpolateFuncOptions()
				.size(std::vector<int64_t>({out_h, out_w}))
				.mode(torch::kBilinear)
				.align_corners(false)).squeeze(0);
	}
	t_disp = t_disp.permute({1,2,0}).mul(255).clamp(0,255).to(torch::kU8).contiguous();
	return cv::Mat(out_h, out_w, CV_8UC3, t_disp.data_ptr()).clone();
}

void show_grid(const std::vector<std::vector<torch::Tensor>>& tensor_grid, const std::vector<std::string>& labels, int cell_h = 128, int cell_w = 128) {
	std::vector<cv::Mat> grid_rows;
	for (const auto& row : tensor_grid) {
		std::vector<cv::Mat> mats;
		for (const auto& t : row) mats.push_back(tensor_to_mat(t, cell_h, cell_w));
		cv::Mat row_mat; cv::hconcat(mats, row_mat); grid_rows.push_back(row_mat);
	}
	cv::Mat grid; cv::vconcat(grid_rows, grid);
	// Add labels
	int nrows = tensor_grid.size(), ncols = tensor_grid[0].size();
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			int idx = i*ncols+j;
			cv::putText(grid, labels[idx], cv::Point(j*cell_w+5, i*cell_h+20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
		}
	}
	cv::imshow("Augmentation Grid", grid);
	std::cout << "Press any key in the image window to exit..." << std::endl;
	cv::waitKey(0);
}

int main() {
	// --- Main augmentations ---
	// Load image from disk using OpenCV (BGR format)
	cv::Mat img_bgr = cv::imread("./images/lenna.png", cv::IMREAD_COLOR);
	if (img_bgr.empty()) {
		std::cerr << "Could not load ./images/lenna.png" << std::endl;
		return 1;
	}
	// Convert to RGB and resize to 128x128 for consistency
	cv::Mat img_rgb;
	cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
	cv::resize(img_rgb, img_rgb, cv::Size(128, 128));
	// Convert to float32 and scale to [0,1]
	cv::Mat img_float;
	img_rgb.convertTo(img_float, CV_32F, 1.0/255.0);
	// Convert to torch tensor (CHW)
	torch::Tensor img = torch::from_blob(img_float.data, {128, 128, 3}, torch::kFloat32).permute({2,0,1}).clone();

	// Horizontal flip: mirror image left-right
	auto hflip = img.flip({2});
	// Vertical flip: mirror image top-bottom
	auto vflip = img.flip({1});
	// Center crop: extract a 64x64 patch from the center
	int crop = 64;
	int start = (128 - crop) / 2;
	auto crop_img = img.index({torch::indexing::Slice(), torch::indexing::Slice(start, start+crop), torch::indexing::Slice(start, start+crop)});
	// Rotate 90 degrees: transpose and flip
	auto rot90 = img.transpose(1,2).flip(2);
	// Brightness: multiply by 1.5, clamp to [0,1]
	auto bright = (img * 1.5).clamp(0, 1);
	// Contrast: shift away from mean, then clamp
	auto mean = img.mean();
	auto contrast = ((img - mean) * 2 + mean).clamp(0, 1);
	// Gaussian blur: use OpenCV to blur the image
	torch::Tensor img_hwc = img.permute({1,2,0}).clone();
	cv::Mat mat_blur(img_hwc.size(0), img_hwc.size(1), CV_32FC3, img_hwc.data_ptr<float>());
	cv::Mat mat_blurred;
	cv::GaussianBlur(mat_blur, mat_blurred, cv::Size(9,9), 3);
	torch::Tensor blurred = torch::from_blob(mat_blurred.data, {128,128,3}, torch::kFloat32).clone().permute({2,0,1});
	// Add random noise: add Gaussian noise and clamp
	auto noise = img + 0.2 * torch::randn_like(img);
	noise = noise.clamp(0, 1);

	// --- Visualization ---
	// Display all results in a labeled 3x3 grid
	std::vector<std::vector<torch::Tensor>> tensor_grid = {
		{img, crop_img, hflip},
		{vflip, rot90, bright},
		{contrast, blurred, noise}
	};
	std::vector<std::string> labels = {
		"Original", "Crop (center)", "Horizontal Flip",
		"Vertical Flip", "Rotated 90°", "Brightened",
		"High Contrast", "Blurred", "Noisy"
	};
	show_grid(tensor_grid, labels);
	return 0;
}

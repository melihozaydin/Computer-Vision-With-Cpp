#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

// Define the same Net as in training
struct Net : torch::nn::Module {
    Net() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1))); // 28x28 -> 28x28
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1))); // 28x28 -> 28x28
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
        dropout1 = register_module("dropout1", torch::nn::Dropout(0.25));
        fc1 = register_module("fc1", torch::nn::Linear(64 * 7 * 7, 128));
        dropout2 = register_module("dropout2", torch::nn::Dropout(0.5));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
        pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = pool(x); // 28x28 -> 14x14
        x = torch::relu(bn2->forward(conv2->forward(x)));
        x = pool(x); // 14x14 -> 7x7
        x = dropout1->forward(x);
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = dropout2->forward(x);
        x = fc2->forward(x);
        return x;
    }
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
};
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// Draw callback state
struct DrawState {
    bool drawing = false;
    cv::Point last_pt;
    cv::Mat canvas;
};

void on_mouse(int event, int x, int y, int, void* userdata) {
    DrawState* state = (DrawState*)userdata;
    if (event == cv::EVENT_LBUTTONDOWN) {
        state->drawing = true;
        state->last_pt = cv::Point(x, y);
    } else if (event == cv::EVENT_MOUSEMOVE && state->drawing) {
        cv::line(state->canvas, state->last_pt, cv::Point(x, y), cv::Scalar(255), 12, cv::LINE_AA);
        state->last_pt = cv::Point(x, y);
    } else if (event == cv::EVENT_LBUTTONUP) {
        state->drawing = false;
    }
}

int main() {
    // Support both LibTorch and TorchScript models
    std::string model_path = "mnist_cnn_best.pt";
    bool use_torchscript = false;
    if (const char* env_p = std::getenv("MNIST_USE_TORCHSCRIPT")) {
        use_torchscript = std::string(env_p) == "1";
    }
    std::shared_ptr<Net> libtorch_model;
    torch::jit::script::Module torchscript_model;
    bool model_loaded = false;
    torch::Device model_device(torch::kCPU);
    if (use_torchscript) {
        try {
            torchscript_model = torch::jit::load(model_path);
            // Try to move to CUDA if available
            if (torch::cuda::is_available()) {
                torchscript_model.to(torch::kCUDA);
                model_device = torch::kCUDA;
            }
            model_loaded = true;
        } catch (const c10::Error& e) {
            std::cerr << "Error: Could not load TorchScript model '" << model_path << "'.\n";
            std::cerr << e.what() << std::endl;
            return 1;
        }
        torchscript_model.eval();
        std::cout << "Loaded TorchScript model: " << model_path << std::endl;
    } else {
        try {
            libtorch_model = std::make_shared<Net>();
            torch::load(libtorch_model, model_path);
            // Try to move to CUDA if available
            if (torch::cuda::is_available()) {
                libtorch_model->to(torch::kCUDA);
                model_device = torch::kCUDA;
            }
            model_loaded = true;
        } catch (const c10::Error& e) {
            std::cerr << "Error: Could not load LibTorch model '" << model_path << "'.\n";
            std::cerr << e.what() << std::endl;
            return 1;
        }
        libtorch_model->eval();
        std::cout << "Loaded LibTorch model: " << model_path << std::endl;
    }
    if (!model_loaded) {
        std::cerr << "Model was not loaded correctly! Exiting.\n";
        return 1;
    }
    std::cout << "Model device: " << (model_device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Create drawing canvas
    DrawState state;
    state.canvas = cv::Mat::zeros(280, 280, CV_8UC1);
    const std::string win_title = "Draw a digit (press 'c' to clear, 'q' to quit, 'p' to predict)";
    cv::namedWindow(win_title);
    cv::setMouseCallback(win_title, on_mouse, &state);

    // Instructions overlay (drawn below the canvas)
    const std::vector<std::string> instructions = {
        "Draw a digit with your mouse:",
        "p: Predict digit   c: Clear canvas   q: Quit"
    };

    int canvas_h = 280, canvas_w = 280;
    int info_h = 50; // Height for instructions area
    std::string last_prediction = "";

    while (true) {
        // Create display image with extra space for instructions
        cv::Mat display(canvas_h + info_h, canvas_w, CV_8UC3, cv::Scalar(30,30,30));
        // Copy canvas to display
        cv::Mat roi = display(cv::Rect(0, 0, canvas_w, canvas_h));
        cv::Mat canvas_bgr;
        cv::cvtColor(state.canvas, canvas_bgr, cv::COLOR_GRAY2BGR);
        canvas_bgr.copyTo(roi);
        // Draw instructions below
        int y0 = canvas_h + 20, dy = 22;
        for (size_t i = 0; i < instructions.size(); ++i) {
            cv::putText(display, instructions[i], cv::Point(10, y0 + i*dy),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 1, cv::LINE_AA);
        }
        // Draw last prediction if available
        if (!last_prediction.empty()) {
            cv::putText(display, last_prediction, cv::Point(10, canvas_h - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2, cv::LINE_AA);
        }
        cv::imshow(win_title, display);
        char key = (char)cv::waitKey(1);
        if (key == 'q') break;
        if (key == 'c') {
            state.canvas = cv::Mat::zeros(canvas_h, canvas_w, CV_8UC1);
            last_prediction = "";
        }
        if (key == 'p') {
            // Preprocess: resize to 28x28, invert, normalize
            cv::Mat img;
            cv::resize(state.canvas, img, cv::Size(28,28));
            img = 255 - img;
            img.convertTo(img, CV_32F, 1.0/255.0);
            auto tensor = torch::from_blob(img.data, {1,1,28,28}, torch::kFloat32).clone();
            // Move tensor to model device
            tensor = tensor.to(model_device);
            // Run inference
            torch::Tensor output;
            if (use_torchscript) {
                if (!model_loaded) {
                    std::cerr << "TorchScript model not loaded!\n";
                    last_prediction = "Model not loaded!";
                } else {
                    output = torchscript_model.forward({tensor}).toTensor();
                }
            } else {
                if (!model_loaded) {
                    std::cerr << "LibTorch model not loaded!\n";
                    last_prediction = "Model not loaded!";
                } else {
                    output = libtorch_model->forward({tensor});
                }
            }
            if (model_loaded) {
                std::cout << "Raw model output tensor: " << output << std::endl;
                int pred = output.argmax(1).item<int>();
                last_prediction = "Predicted digit: " + std::to_string(pred);
                std::cout << last_prediction << std::endl;
            }
        }
    }
    return 0;
}

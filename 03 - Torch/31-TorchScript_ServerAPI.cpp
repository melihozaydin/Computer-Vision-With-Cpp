// 31-TorchScript_ServerAPI.cpp
// Minimal C++ HTTP server API for TorchScript model inference (using Crow)
// TEST WITH : curl -X POST http://localhost:18080/predict --data-binary @<(base64 -w 0 images/lenna.png)
#include <crow.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// Define the same Net as in training
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
    // Support both LibTorch and TorchScript models
    std::string model_path = "mnist_cnn.pt";
    bool use_torchscript = false;
    if (const char* env_p = std::getenv("MNIST_USE_TORCHSCRIPT")) {
        use_torchscript = std::string(env_p) == "1";
    }
    std::shared_ptr<Net> libtorch_model;
    torch::jit::script::Module torchscript_model;
    if (use_torchscript) {
        try {
            torchscript_model = torch::jit::load(model_path);
        } catch (...) {
            std::cerr << "Error: Could not load TorchScript model '" << model_path << "'.\n";
            return 1;
        }
        torchscript_model.eval();
        std::cout << "Loaded TorchScript model: " << model_path << std::endl;
    } else {
        try {
            libtorch_model = std::make_shared<Net>();
            torch::load(libtorch_model, model_path);
        } catch (...) {
            std::cerr << "Error: Could not load LibTorch model '" << model_path << "'.\n";
            return 1;
        }
        libtorch_model->eval();
        std::cout << "Loaded LibTorch model: " << model_path << std::endl;
    }

    crow::SimpleApp app;

    // POST /predict, expects base64 PNG image in body
    CROW_ROUTE(app, "/predict").methods("POST"_method)
    ([&libtorch_model, &torchscript_model, use_torchscript](const crow::request& req) {
        // Decode base64 PNG to cv::Mat
        std::string b64 = req.body;
        std::string decoded = crow::utility::base64decode(b64);
        std::vector<uchar> buf(decoded.begin(), decoded.end());
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::string log = "[PREDICT] Invalid image received.";
            std::cout << log << std::endl;
            return crow::response(400, log);
        }
        std::ostringstream log;
        log << "[PREDICT] Received image: " << img.cols << "x" << img.rows << "\n";
        // Preprocess: resize, invert, normalize
        cv::resize(img, img, cv::Size(28,28));
        img = 255 - img;
        img.convertTo(img, CV_32F, 1.0/255.0);
        auto tensor = torch::from_blob(img.data, {1,1,28,28}, torch::kFloat32).clone();
        torch::Tensor output;
        if (use_torchscript) {
            output = torchscript_model.forward({tensor}).toTensor();
        } else {
            output = libtorch_model->forward({tensor});
        }
        int pred = output.argmax(1).item<int>();
        float conf = output.softmax(1).max().item<float>();
        log << "[PREDICT] Prediction: " << pred << ", Confidence: " << conf << std::endl;
        std::cout << log.str();
        return crow::response(log.str());
    });

    std::cout << "Server running on http://localhost:18080\n";
    app.port(18080).multithreaded().run();
    return 0;
}

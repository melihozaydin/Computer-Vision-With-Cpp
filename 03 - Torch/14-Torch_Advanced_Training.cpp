// 14-Torch_Advanced_Training.cpp
// Advanced training: LR scheduler, early stopping, and TensorBoard logging
// Based on 13-Torch_Train_CNN_MNIST.cpp
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <sys/stat.h>

// Simple TensorBoard logger (writes scalars to tfevents format)
struct TensorBoardLogger {
    std::ofstream ofs;
    TensorBoardLogger(const std::string& logdir) {
        ofs.open(logdir + "/scalars.txt", std::ios::out | std::ios::trunc);
    }
    void log_scalar(const std::string& tag, float value, int step) {
        ofs << tag << "," << value << "," << step << std::endl;
    }
    ~TensorBoardLogger() { ofs.close(); }
};
// Improved CNN for MNIST: deeper, more filters, batch norm, dropout, maxpool
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

int main(int argc, char** argv) {
    int num_epochs = 50;
    float learning_rate = 0.01f;
    int patience = 5;
    int batch_size = 128;
    std::string logdir = "runs";

    // .build/14-Torch_Advanced_Training 50 0.01 5 128 runs

    if (argc > 1) num_epochs = std::stoi(argv[1]);
    if (argc > 2) learning_rate = std::stof(argv[2]);
    if (argc > 3) patience = std::stoi(argv[3]);
    if (argc > 4) batch_size = std::stoi(argv[4]);
    if (argc > 5) logdir = argv[5];

    // Device selection
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    Net net;
    net.to(device);

    // Print model info and all params at very top
    std::cout << "Model architecture:\n" << net << std::endl;
    size_t param_count = 0;
    for (const auto& p : net.parameters()) param_count += p.numel();
    std::cout << "Total parameters: " << param_count << std::endl;
    std::cout << "Num epochs: " << num_epochs
              << ", LR: " << learning_rate
              << ", Patience: " << patience
              << ", Batch size: " << batch_size
              << ", Logdir: " << logdir << std::endl;
    std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Ensure logdir exists
    struct stat st = {0};
    if (stat(logdir.c_str(), &st) == -1) {
        mkdir(logdir.c_str(), 0700);
    }

    auto train_dataset = torch::data::datasets::MNIST("./data/MNIST/raw").map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader(train_dataset, batch_size);
    auto test_dataset = torch::data::datasets::MNIST("./data/MNIST/raw", torch::data::datasets::MNIST::Mode::kTest).map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader(test_dataset, batch_size);

    // Optimizer
    torch::optim::SGD optimizer(net.parameters(), learning_rate);
    // LR scheduler
    torch::optim::StepLR scheduler(optimizer, /*step_size=*/5, /*gamma=*/0.5);
    TensorBoardLogger tb(logdir);

    float best_loss = std::numeric_limits<float>::max();
    float best_acc = 0.0f;
    int best_epoch = 0;
    int epochs_no_improve = 0;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        net.train();
        float train_loss = 0.0;
        size_t batch_idx = 0;
        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device), target = batch.target.to(device);
            auto output = net.forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            train_loss += loss.item<float>() * data.size(0);
            if (batch_idx++ % 100 == 0)
                std::cout << "Epoch " << epoch << ", Batch " << batch_idx << ", Loss: " << loss.item<float>() << std::endl;
        }
        train_loss /= train_dataset.size().value();
        tb.log_scalar("train_loss", train_loss, epoch);
        // Validation
        net.eval();
        float val_loss = 0.0;
        int correct = 0, total = 0;
        for (auto& batch : *test_loader) {
            auto data = batch.data.to(device), target = batch.target.to(device);
            auto output = net.forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);
            val_loss += loss.item<float>() * data.size(0);
            auto pred = output.argmax(1);
            correct += pred.eq(target).sum().item<int>();
            total += data.size(0);
        }
        val_loss /= test_dataset.size().value();
        float val_acc = static_cast<float>(correct) / total;
        tb.log_scalar("val_loss", val_loss, epoch);
        tb.log_scalar("val_acc", val_acc, epoch);
        std::cout << "Epoch " << epoch
                  << ": Train loss: " << train_loss
                  << ", Val loss: " << val_loss
                  << ", Val acc: " << val_acc
                  << ", Patience: " << patience
                  << ", No improve: " << epochs_no_improve
                  << ", Best epoch: " << best_epoch
                  << std::endl;
        // Early stopping
        if (val_loss < best_loss) {
            best_loss = val_loss;
            best_acc = val_acc;
            best_epoch = epoch;
            epochs_no_improve = 0;
            torch::save(std::make_shared<Net>(net), "mnist_cnn_best.pt");
        } else {
            epochs_no_improve++;
            std::cout << "  [EarlyStopping] No improvement for " << epochs_no_improve << "/" << patience << " epochs." << std::endl;
            if (epochs_no_improve >= patience) {
                std::cout << "Early stopping at epoch " << epoch << std::endl;
                break;
            }
        }
        scheduler.step();
    }
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "Best epoch: " << best_epoch << std::endl;
    std::cout << "Best val loss: " << best_loss << std::endl;
    std::cout << "Best val acc: " << best_acc << std::endl;
    std::cout << "Best model saved as mnist_cnn_best.pt" << std::endl;

    // === Final test with best model ===
    std::cout << "\n=== Final Test with Best Model ===" << std::endl;
    auto best_net = std::make_shared<Net>();
    try {
        torch::load(best_net, "mnist_cnn_best.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load best model for test: " << e.what() << std::endl;
        return 1;
    }
    best_net->to(device);
    best_net->eval();
    float test_loss = 0.0;
    int test_correct = 0, test_total = 0;
    for (auto& batch : *test_loader) {
        auto data = batch.data.to(device), target = batch.target.to(device);
        auto output = best_net->forward(data);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        test_loss += loss.item<float>() * data.size(0);
        auto pred = output.argmax(1);
        test_correct += pred.eq(target).sum().item<int>();
        test_total += data.size(0);
    }
    test_loss /= test_dataset.size().value();
    float test_acc = static_cast<float>(test_correct) / test_total;
    std::cout << "Test loss: " << test_loss << ", Test acc: " << test_acc << std::endl;
    return 0;
}

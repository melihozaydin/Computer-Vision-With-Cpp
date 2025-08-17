
# Makefile for compiling all Torch C++ curriculum examples
# Usage: make -f Makefile.torch

CXX = g++
CXXFLAGS = -std=c++17 -O2 -I$(HOME)/libtorch/include -I$(HOME)/libtorch/include/torch/csrc/api/include -I/usr/include/opencv4
LDFLAGS = -L$(HOME)/libtorch/lib -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -Wl,-rpath,$(HOME)/libtorch/lib -L/usr/lib/x86_64-linux-gnu
BUILD_DIR = .build
SRC = $(filter-out 20-Torch_Use_TorchVision.cpp, $(wildcard *.cpp))
TARGETS = $(patsubst %.cpp,$(BUILD_DIR)/%,$(SRC))

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

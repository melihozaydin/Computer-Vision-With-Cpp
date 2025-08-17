EIGEN_INCLUDE = /usr/include/eigen3
SRC = $(wildcard *.cpp)
BUILD_DIR = .build
TARGETS = $(patsubst %.cpp,$(BUILD_DIR)/%,$(SRC))

CXX = g++
CXXFLAGS = -I $(EIGEN_INCLUDE) -O2 -Wall

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
CXX = g++
CXXFLAGS = -O2 -Wall
SRC = $(wildcard *.cpp)
BUILD_DIR = .build
TARGETS = $(patsubst %.cpp,$(BUILD_DIR)/%,$(SRC))

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

NVCC ?= nvcc
CUDAFLAGS ?= -std=c++17 -O2
BUILD_DIR := .build
SRC := $(wildcard *.cu)
TARGETS := $(patsubst %.cu,$(BUILD_DIR)/%,$(SRC))

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.cu | $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

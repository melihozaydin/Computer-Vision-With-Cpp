CXX ?= g++
PCL_MODULES ?= pcl_common pcl_io pcl_filters pcl_features pcl_segmentation pcl_registration pcl_kdtree pcl_search pcl_sample_consensus
CXXFLAGS += -std=c++17 -O2 $(shell pkg-config --cflags $(PCL_MODULES))
LDFLAGS += $(shell pkg-config --libs $(PCL_MODULES))
BUILD_DIR := .build
SRC := $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp,$(BUILD_DIR)/%,$(SRC))

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

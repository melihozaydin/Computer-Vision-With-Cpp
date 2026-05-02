CXX ?= g++
VTK_LIBS ?= -lvtkRenderingCore-9.1 -lvtkCommonCore-9.1 -lvtkCommonDataModel-9.1 -lvtkCommonMath-9.1 -lvtkCommonTransforms-9.1 -lvtkRenderingOpenGL2-9.1 -lvtkInteractionStyle-9.1 -lvtkRenderingFreeType-9.1 -lvtkFiltersSources-9.1 -lvtkRenderingContext2D-9.1 -lvtkCommonColor-9.1
PCL_MODULES ?= $(shell \
	if pkg-config --exists pcl_common; then \
		echo "pcl_common pcl_io pcl_filters pcl_features pcl_segmentation pcl_registration pcl_kdtree pcl_search pcl_sample_consensus pcl_visualization"; \
	elif pkg-config --exists pcl_common-1.13; then \
		echo "pcl_common-1.13 pcl_io-1.13 pcl_filters-1.13 pcl_features-1.13 pcl_segmentation-1.13 pcl_registration-1.13 pcl_kdtree-1.13 pcl_search-1.13 pcl_sample_consensus-1.13 pcl_visualization-1.13"; \
	elif pkg-config --exists pcl_common-1.12; then \
		echo "pcl_common-1.12 pcl_io-1.12 pcl_filters-1.12 pcl_features-1.12 pcl_segmentation-1.12 pcl_registration-1.12 pcl_kdtree-1.12 pcl_search-1.12 pcl_sample_consensus-1.12 pcl_visualization-1.12"; \
	else \
		echo "pcl_common"; \
	fi)
CXXFLAGS += -std=c++17 -O2 $(shell pkg-config --cflags $(PCL_MODULES)) $(shell pkg-config --cflags vtk-9.1 2>/dev/null) -I/usr/include/vtk-9.1
LDFLAGS += $(shell pkg-config --libs --static $(PCL_MODULES)) $(shell pkg-config --libs vtk-9.1 2>/dev/null) $(VTK_LIBS) -Wl,--copy-dt-needed-entries
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

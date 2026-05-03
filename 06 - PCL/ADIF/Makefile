# =============================================================================
# ADIF — Automated Dimensional Inspection Framework
# =============================================================================
# Targets:
#   all            — build both binaries
#   generate_data  — build the data generator
#   adif           — build the inspection pipeline
#   generate       — generate data/reference_part.pcd + data/scanned_part.pcd
#   clean          — remove build artefacts
#
# Usage:
#   make generate      # creates data/ PCD files
#   make adif
#   .build/adif data/reference_part.pcd data/scanned_part.pcd --tolerance 0.001
# =============================================================================

CXX ?= g++

VTK_LIBS ?= \
  -lvtkRenderingCore-9.1 -lvtkRenderingLOD-9.1 -lvtkCommonCore-9.1 \
  -lvtkCommonDataModel-9.1 -lvtkCommonMath-9.1 -lvtkCommonTransforms-9.1 \
  -lvtkRenderingOpenGL2-9.1 -lvtkInteractionStyle-9.1 -lvtkRenderingFreeType-9.1 \
  -lvtkFiltersSources-9.1 -lvtkRenderingContext2D-9.1 -lvtkCommonColor-9.1

PCL_MODULES ?= $(shell \
  if pkg-config --exists pcl_common; then \
    echo "pcl_common pcl_io pcl_filters pcl_features pcl_segmentation \
          pcl_registration pcl_kdtree pcl_search pcl_sample_consensus \
          pcl_visualization"; \
  elif pkg-config --exists pcl_common-1.13; then \
    echo "pcl_common-1.13 pcl_io-1.13 pcl_filters-1.13 pcl_features-1.13 \
          pcl_segmentation-1.13 pcl_registration-1.13 pcl_kdtree-1.13 \
          pcl_search-1.13 pcl_sample_consensus-1.13 pcl_visualization-1.13"; \
  elif pkg-config --exists pcl_common-1.12; then \
    echo "pcl_common-1.12 pcl_io-1.12 pcl_filters-1.12 pcl_features-1.12 \
          pcl_segmentation-1.12 pcl_registration-1.12 pcl_kdtree-1.12 \
          pcl_search-1.12 pcl_sample_consensus-1.12 pcl_visualization-1.12"; \
  else \
    echo "pcl_common"; \
  fi)

CXXFLAGS += \
  -std=c++17 -O2 \
  $(shell pkg-config --cflags $(PCL_MODULES)) \
  $(shell pkg-config --cflags vtk-9.1 2>/dev/null) \
  -I/usr/include/vtk-9.1 \
  -I..                          # for pcl_viewer_utils.h

LDFLAGS += \
  $(shell pkg-config --libs --static $(PCL_MODULES)) \
  $(shell pkg-config --libs vtk-9.1 2>/dev/null) \
  $(VTK_LIBS) \
  -Wl,--copy-dt-needed-entries

BUILD_DIR := .build

all: $(BUILD_DIR) $(BUILD_DIR)/generate_data $(BUILD_DIR)/adif

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/generate_data: generate_data.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD_DIR)/adif: main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Convenience: build + run the data generator
generate: $(BUILD_DIR)/generate_data
	mkdir -p data
	$(BUILD_DIR)/generate_data

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean generate

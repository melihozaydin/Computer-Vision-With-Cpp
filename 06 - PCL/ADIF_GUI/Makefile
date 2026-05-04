# =============================================================================
# ADIF_GUI — Interactive Dimensional Inspection GUI
# =============================================================================
# Targets:
#   all          — build adif_gui binary
#   clean        — remove build artefacts
#
# Usage:
#   make
#   .build/adif_gui <scan.pcd> [<reference.pcd>]
#
# Quick start (reuse ADIF synthetic data):
#   cd ../ADIF && make generate && cd ../ADIF_GUI && make
#   .build/adif_gui ../ADIF/data/scanned_part.pcd ../ADIF/data/reference_part.pcd
# =============================================================================

CXX ?= g++

VTK_LIBS ?= \
  -lvtkRenderingCore-9.1 -lvtkRenderingLOD-9.1 -lvtkCommonCore-9.1 \
  -lvtkCommonDataModel-9.1 -lvtkCommonMath-9.1 -lvtkCommonTransforms-9.1 \
  -lvtkRenderingOpenGL2-9.1 -lvtkInteractionStyle-9.1 -lvtkRenderingFreeType-9.1 \
  -lvtkFiltersSources-9.1 -lvtkRenderingContext2D-9.1 -lvtkCommonColor-9.1

PCL_MODULES ?= $(shell \
  if pkg-config --exists pcl_common; then \
    echo "pcl_common pcl_io pcl_filters pcl_features pcl_kdtree pcl_search \
          pcl_visualization"; \
  elif pkg-config --exists pcl_common-1.13; then \
    echo "pcl_common-1.13 pcl_io-1.13 pcl_filters-1.13 pcl_features-1.13 \
          pcl_kdtree-1.13 pcl_search-1.13 pcl_visualization-1.13"; \
  elif pkg-config --exists pcl_common-1.12; then \
    echo "pcl_common-1.12 pcl_io-1.12 pcl_filters-1.12 pcl_features-1.12 \
          pcl_kdtree-1.12 pcl_search-1.12 pcl_visualization-1.12"; \
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

all: $(BUILD_DIR) $(BUILD_DIR)/adif_gui

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/adif_gui: adif_gui.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

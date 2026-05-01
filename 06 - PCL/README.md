# PCL Point Cloud Processing Examples

This folder contains small PCL (Point Cloud Library) examples for synthetic 3D point-cloud processing.
All examples are self-contained and generate their own clouds, so no external `.pcd` dataset is required.

## Included examples

| File | Topic |
|------|-------|
| `00-PCL_Setup.cpp` | Verify PCL install and create a tiny cloud |
| `01-Cloud_Generation.cpp` | Generate and save a synthetic PCD file |
| `02-VoxelGrid_Downsampling.cpp` | Downsample a cloud using `VoxelGrid` |
| `03-Normal_Estimation.cpp` | Estimate normals with a KD-tree |
| `04-Plane_Segmentation.cpp` | RANSAC plane fitting |
| `05-ICP_Registration.cpp` | Register two clouds with ICP |

## Quick start

### Local

```bash
sudo apt install build-essential pkg-config libpcl-dev
cd "06 - PCL"
./run_all_examples.sh
./run_all_examples.sh --build-only
```

### Docker

```bash
cd "06 - PCL"
./run_all_examples.sh --docker
```

Docker mode uses `ubuntu:22.04` and installs `libpcl-dev` automatically.

## Build manually

```bash
make all
make clean
```

## Notes

- These examples avoid visualization windows so they work better in headless WSL and CI-like environments.
- `01-Cloud_Generation.cpp` writes `synthetic_line_cloud.pcd` in this folder.
- PCL can be heavy to install; Docker mode is convenient when you do not want a local PCL setup.

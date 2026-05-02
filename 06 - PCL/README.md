# PCL Point Cloud Processing Examples

Real-time visualization of 3D point cloud algorithms using the Point Cloud Library (PCL).
All examples are self-contained, generate synthetic point clouds, and include interactive viewers so you can rotate and inspect the results as they run.

## Included examples

| File | What it does | The math | Memory |
|------|------|----------|--------|
| `00-PCL_Setup.cpp` | Verifies PCL is installed and creates a minimal 4-point cloud. Demonstrates basic viewer setup. | Coordinate system: points are XYZ triplets in ℝ³. | Stores 4 points on stack initially, then loads into `PCLVisualizer`'s GPU memory for rendering. |
| `01-Cloud_Generation.cpp` | Generates a wavy parametric line cloud (200 points) using a sine function, saves it to PCD format, and visualizes the curve. | Parametric curve: **P(t) = (t, 0.5t, 0.1·sin(2πt))** where t ∈ [0, 2). | Heap-allocated cloud (200 points × 12 bytes/point = ~2.4 KB) plus visualization buffer on GPU. |
| `02-VoxelGrid_Downsampling.cpp` | Downsamples a 50×50×4 dense grid into a sparse set using cubic voxels. Shows before (red/small) and after (blue/large) side-by-side. | Voxel grid: divides space into leaf_size³ cubes and replaces all points in each cube with their centroid **C = (1/N) Σ pᵢ**. | Original: 10,000 points. Filtered: ~500–1,000 points (depends on density). Allocates two clouds (~12–120 KB each). |
| `03-Normal_Estimation.cpp` | Estimates surface normals by fitting a plane to each point's K-nearest neighbors (K=15). Shows green cloud with cyan normal vectors. | For each point, fits plane **ax + by + cz + d = 0** via PCA on the local neighborhood; the normal is **(a, b, c)** normalized to unit length. | Cloud: 625 points. KD-tree structure: ~5–10 KB. Normal cloud: 625 normals (~2.5 KB). Viewer renders one line segment per normal. |
| `04-Plane_Segmentation.cpp` | Uses RANSAC to find the dominant plane in noisy data. Colors plane inliers (green) separately from outliers (red). | RANSAC: randomly sample 3 points, fit plane, count inliers within threshold (0.01 m), repeat 100+ times and keep best fit. Plane equation: **ax + by + cz + d = 0**. | Cloud: 10,020 points total. RANSAC stores candidate coefficients (4 floats) per iteration (~400 B overhead). Filtered clouds: ~1,200 inliers (plane) + 20 outliers. |
| `05-ICP_Registration.cpp` | Aligns a source cloud to a target cloud using Iterative Closest Point. Shows source (red), aligned result (green), and target (blue) overlaid. | ICP: for each source point, find nearest target point, solve for optimal rigid transform **T = (R, t)** that minimizes MSE of correspondence distances; repeat until convergence **‖T_new − T_old‖ < ε**. | Source + Target: 100 points each (~1.2 KB each). ICP state: 4×4 transformation matrix (64 bytes). Aligned cloud: 100 points (~1.2 KB). Total: ~5 KB working memory. |

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

## Visualization

Each example opens an interactive 3D viewer when run locally:
- **Rotate**: click and drag with left mouse button
- **Zoom**: scroll wheel or middle mouse button drag
- **Pan**: right mouse button drag
- **Press 'q'** to close the viewer and exit

**Headless environments** (WSL without X server, Docker without display): The viewer will catch exceptions and print a skip message; all computation still runs and prints results to stdout.

## Notes

- These examples use `pcl::visualization::PCLVisualizer` to provide real-time 3D feedback, making physics-like phenomena directly observable (point clouds rotating, planes fitting, alignments converging).
- `01-Cloud_Generation.cpp` writes `synthetic_line_cloud.pcd` in this folder (can be loaded by other PCL tools).
- PCL can be heavy to install; Docker mode is convenient when you do not want a local PCL setup.
- For batch or server-side processing, you can suppress the visualizer by modifying the try-catch to skip viewer calls.

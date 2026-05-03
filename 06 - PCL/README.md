# PCL for Inspection Systems

This folder is no longer a grab-bag of toy point-cloud demos. It is now a compact inspection-focused PCL track built around the workflow you actually need for dimensional verification:

**clean the scan → isolate the part → estimate geometry → register to the master → measure deviation**

The examples are written in modern C++, use **PCL + Eigen**, generate or load realistic synthetic data, and keep the visualizers so you can still rotate the scene and sanity-check the algorithm with your eyeballs. Industrial metrology still benefits from “show me the cloud” moments.

## What this folder teaches

These examples focus on the four pillars behind a real inspection system:

1. **Data representation** — point types, alignment, organised vs unorganised clouds
2. **Pre-processing** — PassThrough, VoxelGrid, outlier rejection
3. **Geometry extraction** — normals, KD-trees, plane segmentation
4. **Registration and inspection** — ICP locally, and the `ADIF/` subproject for global + local alignment plus deviation analysis

## Example roadmap

| File | Focus | What you learn |
|---|---|---|
| `00-PCL_Setup.cpp` | PCL foundations | Verifies `PCL` + `Eigen`, explains common point types, memory alignment, and visualises a coloured XYZ axis cloud using `PointXYZRGB`. |
| `01-Cloud_Generation.cpp` | Synthetic inspection data | Builds a clean reference cylinder and a scanned variant with Gaussian noise, a smooth dent, and a rigid misalignment. Saves `data/reference_part.pcd` and `data/scanned_part.pcd`. |
| `02-VoxelGrid_Downsampling.cpp` | Inspection pre-processing | Demonstrates a real pre-processing chain: **PassThrough → VoxelGrid → StatisticalOutlierRemoval** on a dirty scan containing fixture points and random spikes. |
| `03-Normal_Estimation.cpp` | Normals + KD-tree reasoning | Estimates normals on a sphere where the correct answer is known analytically, compares different neighbourhood sizes, and explains why normals matter for ICP, FPFH, and signed deviation. |
| `04-Plane_Segmentation.cpp` | Fixture removal | Uses iterative **RANSAC plane segmentation** with `ExtractIndices` to strip a dominant floor plane and isolate the actual part cloud. |
| `05-ICP_Registration.cpp` | Local registration | Loads the clouds from `01-Cloud_Generation.cpp`, runs ICP, prints the recovered $4 \times 4$ transform, and compares it with known ground truth. |

## Inspection pipeline at a glance

The examples are meant to build toward this workflow:

```text
Scanner / synthetic scan
	|
	v
PassThrough crop
	|
	v
VoxelGrid density normalisation
	|
	v
Outlier removal
	|
	v
Plane / fixture segmentation
	|
	v
Normal estimation + KD-tree search
	|
	v
Registration to golden master
	|
	v
Deviation analysis + pass/fail
```

## The `ADIF/` subfolder

The new `ADIF/` subfolder is the capstone project in this section:

**ADIF = Automated Dimensional Inspection Framework**

It demonstrates the full inspection story instead of stopping at pretty alignment screenshots.

### Included files

| File | Purpose |
|---|---|
| `ADIF/generate_data.cpp` | Generates a reference part and a manufactured/scanned part with noise, pose error, and a dent defect. |
| `ADIF/main.cpp` | Full inspection pipeline: load → preprocess → normals → FPFH → global registration → ICP → signed deviation → colour-coded pass/fail visualisation. |
| `ADIF/Makefile` | Builds the data generator and inspection executable. |

### ADIF capabilities

- Loads **reference** and **target** point clouds from `.pcd`
- Applies **PassThrough** cropping and **VoxelGrid** downsampling
- Estimates normals with parallel normal estimation
- Performs **global registration** using **FPFH + SampleConsensusPrerejective**
- Performs **local registration** using **ICP**
- Computes nearest-neighbour deviation using a **KD-tree**
- Produces a colour-coded inspection result:
  - **Green** = within tolerance
  - **Red** = oversized / proud material
  - **Blue** = undersized / sunken material
- Prints a console report with:
  - MSE
  - RMSE
  - max absolute deviation
  - percentage within tolerance
  - PASS / FAIL decision

## Quick start

### Run the teaching examples

If you want to build and run everything in this folder:

```bash
cd "06 - PCL"
./run_all_examples.sh
```

Build only:

```bash
cd "06 - PCL"
./run_all_examples.sh --build-only
```

### Build manually

```bash
cd "06 - PCL"
make all
make clean
```

## Running ADIF

### 1. Generate synthetic inspection data

```bash
cd "06 - PCL/ADIF"
make generate
```

This creates:

- `ADIF/data/reference_part.pcd`
- `ADIF/data/scanned_part.pcd`

### 2. Build the inspection executable

```bash
cd "06 - PCL/ADIF"
make all
```

### 3. Run the inspection

```bash
cd "06 - PCL/ADIF"
.build/adif data/reference_part.pcd data/scanned_part.pcd --tolerance 0.001
```

The default tolerance above is $0.001\text{ m} = 1.0\text{ mm}$.

You can also change voxel size:

```bash
.build/adif data/reference_part.pcd data/scanned_part.pcd --tolerance 0.0005 --voxel 0.0015
```

## Recommended learning order

If your goal is industrial inspection rather than generic point-cloud graphics, go in this order:

1. `00-PCL_Setup.cpp` — understand point types and cloud structure
2. `02-VoxelGrid_Downsampling.cpp` — understand preprocessing first
3. `04-Plane_Segmentation.cpp` — isolate the part from the environment
4. `03-Normal_Estimation.cpp` — learn why KD-tree neighbourhoods matter
5. `05-ICP_Registration.cpp` — understand local alignment limitations
6. `ADIF/main.cpp` — study the complete inspection workflow

Yes, `03` comes after `04` in this reading order. That is intentional: in inspection, cleaning and isolating the part usually matters before fancier local geometry work.

## Key concepts worth mastering

### KD-trees

KD-trees are the quiet heroes of inspection pipelines. They make nearest-neighbour queries fast enough for:

- normal estimation
- FPFH feature computation
- ICP correspondence search
- deviation analysis against the golden master

Without them, every “find nearest point” step becomes painfully brute-force.

### Surface normals

Normals are essential when you care about **directional** error, not just distance.

They help answer:

- Is the scanned point outside the part or inside it?
- Is the deviation a proud feature or a dent?
- Is a local patch flat, curved, or noisy?

### Precision and tolerances

Inspection systems live and die by unit discipline.

In this folder, dimensions are stored in **metres**, but the console often reports values in **millimetres** for readability. Keep that conversion straight:

- $1\text{ mm} = 0.001\text{ m}$
- $0.5\text{ mm} = 0.0005\text{ m}$

If you are tuning voxel size, correspondence thresholds, or pass/fail logic, always think in the same unit system.

## Visualisation notes

Each example tries to open an interactive `PCLVisualizer` window when supported.

Controls:

- **Rotate:** left mouse drag
- **Pan:** right mouse drag
- **Zoom:** scroll wheel
- **Quit:** `q`

In headless environments, the code catches viewer exceptions and still runs the computation. So if no window appears, the algorithms may still have completed successfully.

## Local vs Docker

### Local

For interactive visualisation, local mode is the most reliable:

```bash
cd "06 - PCL"
./run_all_examples.sh --local --gui
```

### Docker

Docker is useful for reproducible builds and compute-only checks:

```bash
cd "06 - PCL"
./build_docker_env.sh
./run_all_examples.sh --docker --gui
```

The runtime image is `cv-pcl-runtime:22.04`.

> Docker builds are generally fine for compile and pipeline validation, but VTK/PCL GUI windows may still depend on your WSLg/X forwarding setup. If the viewer is stubborn, local mode is the safer bet.

## Notes and gotchas

- `pcl::visualization::PCLVisualizer::addCoordinateSystem(...)` should be called with a valid string id, not a null string pointer.
- If you switch between local and Docker build contexts, clean stale binaries in `.build/` to avoid library mismatch headaches.
- `05-ICP_Registration.cpp` is intentionally a **local** registration example. For large pose errors, use the `ADIF/` pipeline’s feature-based global alignment first.
- The data in `01-Cloud_Generation.cpp` and `ADIF/generate_data.cpp` is synthetic by design so you can reason about ground truth. Real scans are messier, which is rude but educational.

## Where to go next

If you want to push this folder further toward production-style metrology, the next useful upgrades would be:

- point-to-plane ICP
- mesh-to-cloud deviation instead of cloud-to-cloud nearest point
- region-of-interest inspection masks
- CSV/PDF export of defect statistics
- live scanner input instead of synthetic `.pcd` generation
- tolerance bands per feature instead of one global threshold

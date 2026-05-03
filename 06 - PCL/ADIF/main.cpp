// =============================================================================
// ADIF — Automated Dimensional Inspection Framework
// =============================================================================
//
// USAGE:
//   ./adif <reference.pcd> <scanned.pcd> [--tolerance <metres>] [--voxel <metres>]
//
//   reference.pcd  — the "golden master" model (CAD-derived or certified scan)
//   scanned.pcd    — the part to inspect (raw scanner output)
//   --tolerance    — deviation threshold in metres  (default: 0.001 = 1 mm)
//   --voxel        — VoxelGrid leaf size in metres  (default: 0.002 = 2 mm)
//
// PIPELINE:
//
//   1. Load  — read both PCD files, report sizes.
//
//   2. Pre-process — PassThrough crop + VoxelGrid on both clouds.
//                    Ensures equal density before feature extraction.
//
//   3. Normal estimation — required by FPFH feature descriptor.
//
//   4. FPFH feature extraction
//        Fast Point Feature Histogram: for each point encode the angular
//        differences between the point's normal and its neighbours' normals
//        into a 33-bin histogram.  Gives a rotation/translation invariant
//        description of local surface geometry.
//
//   5. Global alignment — SampleConsensusPrerejective (FPFH + RANSAC)
//        Randomly draw 3 source points, find matching target points via
//        nearest-neighbour search in FPFH space, compute the rigid transform,
//        count inliers.  Much faster than brute-force because point-feature
//        matches that are geometrically inconsistent ("prerejected") are
//        discarded before the costly inlier count.
//        Gives a rough alignment good enough for ICP to converge correctly.
//
//   6. ICP fine alignment — point-to-point ICP refines to sub-millimetre.
//
//   7. Deviation analysis
//        For every aligned scanned point:
//          a) Find nearest reference point (KD-tree, O(log N)).
//          b) Signed deviation = dot(displacement, reference_normal)
//             Positive → scanned surface is OUTSIDE reference (oversized).
//             Negative → scanned surface is INSIDE  reference (undersized).
//
//   8. Colour coding
//        Green  : |deviation| ≤ tolerance
//        Red    : deviation  >  tolerance (oversized / proud)
//        Blue   : deviation  < −tolerance (undersized / sunk)
//
//   9. Statistics & pass/fail report
//        MSE, RMSE, max deviation, percentage in tolerance, PASS / FAIL.
//
// =============================================================================

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include "../pcl_viewer_utils.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ── Convenience aliases ───────────────────────────────────────────────────────
using PointT    = pcl::PointXYZ;
using PointNT   = pcl::PointNormal;
using PointRGB  = pcl::PointXYZRGB;
using CloudT    = pcl::PointCloud<PointT>;
using CloudNT   = pcl::PointCloud<PointNT>;
using CloudRGB  = pcl::PointCloud<PointRGB>;
using FeatureT  = pcl::FPFHSignature33;
using CloudFeat = pcl::PointCloud<FeatureT>;

// ── CLI parsing ───────────────────────────────────────────────────────────────
struct Config {
    std::string ref_path;
    std::string scan_path;
    float tolerance = 0.001f;   // 1 mm
    float voxel     = 0.002f;   // 2 mm
};

Config parseArgs(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: adif <reference.pcd> <scanned.pcd>"
                     " [--tolerance <m>] [--voxel <m>]\n";
        std::exit(1);
    }
    Config cfg;
    cfg.ref_path  = argv[1];
    cfg.scan_path = argv[2];
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string key = argv[i];
        float val = std::stof(argv[i + 1]);
        if (key == "--tolerance") cfg.tolerance = val;
        else if (key == "--voxel") cfg.voxel    = val;
    }
    return cfg;
}

// ── Pre-processing: PassThrough (axis-aligned) + VoxelGrid ───────────────────
CloudT::Ptr preprocess(const CloudT::Ptr& raw, float voxel,
                       float z_min = -0.01f, float z_max =  0.20f,
                       float xy_bound = 0.12f)
{
    auto tmp = CloudT::Ptr(new CloudT());
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(raw);

    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_min, z_max);
    pass.filter(*tmp);

    auto tmp2 = CloudT::Ptr(new CloudT());
    pass.setInputCloud(tmp);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-xy_bound, xy_bound);
    pass.filter(*tmp2);

    pass.setInputCloud(tmp2);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-xy_bound, xy_bound);
    pass.filter(*tmp);

    auto out = CloudT::Ptr(new CloudT());
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(tmp);
    vg.setLeafSize(voxel, voxel, voxel);
    vg.filter(*out);
    return out;
}

// ── Estimate normals (OMP parallel) ──────────────────────────────────────────
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const CloudT::Ptr& cloud,
                                                 float radius)
{
    auto normals = pcl::PointCloud<pcl::Normal>::Ptr(
                       new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setRadiusSearch(radius);
    auto tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    ne.compute(*normals);
    return normals;
}

// ── Compute FPFH features ─────────────────────────────────────────────────────
CloudFeat::Ptr computeFPFH(const CloudT::Ptr& cloud,
                            const pcl::PointCloud<pcl::Normal>::Ptr& normals,
                            float radius)
{
    auto features = CloudFeat::Ptr(new CloudFeat());
    pcl::FPFHEstimationOMP<PointT, pcl::Normal, FeatureT> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setRadiusSearch(radius);
    auto tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>());
    fpfh.setSearchMethod(tree);
    fpfh.compute(*features);
    return features;
}

// ── Global alignment: FPFH + SampleConsensusPrerejective ─────────────────────
// Returns the aligned cloud + whether it converged.
std::pair<CloudT::Ptr, bool> globalAlign(
    const CloudT::Ptr& source,     const CloudFeat::Ptr& source_feat,
    const CloudT::Ptr& target,     const CloudFeat::Ptr& target_feat,
    float voxel)
{
    pcl::SampleConsensusPrerejective<PointT, PointT, FeatureT> align;
    align.setInputSource(source);
    align.setSourceFeatures(source_feat);
    align.setInputTarget(target);
    align.setTargetFeatures(target_feat);
    align.setMaximumIterations(50000);
    align.setNumberOfSamples(3);
    align.setCorrespondenceRandomness(5);
    align.setSimilarityThreshold(0.9f);
    align.setMaxCorrespondenceDistance(voxel * 2.5f);
    align.setInlierFraction(0.25f);

    auto result = CloudT::Ptr(new CloudT());
    align.align(*result);
    return {result, align.hasConverged()};
}

// ── Deviation computation & colour coding ────────────────────────────────────
// Returns coloured cloud + deviation values (signed, metres).
std::pair<CloudRGB::Ptr, std::vector<float>> computeDeviations(
    const CloudT::Ptr& aligned_scan,
    const CloudT::Ptr& reference,
    const pcl::PointCloud<pcl::Normal>::Ptr& ref_normals,
    float tolerance)
{
    // Build KD-tree on reference
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(reference);

    auto coloured = CloudRGB::Ptr(new CloudRGB());
    std::vector<float> deviations;
    deviations.reserve(aligned_scan->size());

    std::vector<int>   nn_idx(1);
    std::vector<float> nn_dist(1);

    for (const auto& pt : aligned_scan->points) {
        kdtree.nearestKSearch(pt, 1, nn_idx, nn_dist);
        int idx = nn_idx[0];

        const auto& ref_pt  = reference->points[idx];
        const auto& ref_n   = ref_normals->points[idx];

        // Displacement vector: aligned → reference
        Eigen::Vector3f disp(pt.x - ref_pt.x, pt.y - ref_pt.y, pt.z - ref_pt.z);
        Eigen::Vector3f nrm(ref_n.normal_x, ref_n.normal_y, ref_n.normal_z);

        // Signed deviation: positive = outside (proud), negative = inside (sunk)
        float dev = 0.0f;
        if (nrm.norm() > 0.5f)
            dev = disp.dot(nrm.normalized());
        else
            dev = std::sqrt(nn_dist[0]);   // fallback: unsigned distance
        deviations.push_back(dev);

        // Colour
        PointRGB p{};
        p.x = pt.x; p.y = pt.y; p.z = pt.z;
        if (std::abs(dev) <= tolerance) {
            p.r =  60; p.g = 200; p.b =  60;   // green  — in tolerance
        } else if (dev > tolerance) {
            p.r = 220; p.g =  60; p.b =  60;   // red    — oversized
        } else {
            p.r =  60; p.g =  60; p.b = 220;   // blue   — undersized
        }
        coloured->push_back(p);
    }

    return {coloured, deviations};
}

// ── Inspection report ─────────────────────────────────────────────────────────
void printReport(const std::vector<float>& devs, float tolerance) {
    double sum_sq = 0.0;
    float  max_dev = 0.0f;
    int    in_tol  = 0;

    for (float d : devs) {
        sum_sq  += d * d;
        max_dev  = std::max(max_dev, std::abs(d));
        if (std::abs(d) <= tolerance) ++in_tol;
    }

    double mse  = sum_sq / devs.size();
    double rmse = std::sqrt(mse);
    double pct  = 100.0 * in_tol / devs.size();
    bool   pass = (pct >= 95.0);   // PASS if ≥ 95 % of points within tolerance

    const std::string sep(52, '=');
    std::cout << "\n" << sep << "\n";
    std::cout << "  ADIF INSPECTION REPORT\n";
    std::cout << sep << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Points analysed  : " << devs.size()    << "\n";
    std::cout << "  Tolerance (±)    : " << tolerance*1000 << " mm\n";
    std::cout << "  MSE              : " << mse * 1e6      << " mm²\n";
    std::cout << "  RMSE             : " << rmse * 1000    << " mm\n";
    std::cout << "  Max |deviation|  : " << max_dev * 1000 << " mm\n";
    std::cout << "  In tolerance     : " << in_tol << " / " << devs.size()
              << "  (" << std::setprecision(1) << pct << "%)\n";
    std::cout << sep << "\n";
    std::cout << "  RESULT           : " << (pass ? "PASS ✓" : "FAIL ✗") << "\n";
    std::cout << sep << "\n\n";
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Config cfg = parseArgs(argc, argv);

    // 1. Load
    auto ref_raw  = CloudT::Ptr(new CloudT());
    auto scan_raw = CloudT::Ptr(new CloudT());
    if (pcl::io::loadPCDFile(cfg.ref_path,  *ref_raw)  < 0 ||
        pcl::io::loadPCDFile(cfg.scan_path, *scan_raw) < 0) {
        std::cerr << "Failed to load PCD files.\n";
        return 1;
    }
    std::cout << "Loaded reference : " << ref_raw->size()  << " pts\n";
    std::cout << "Loaded scanned   : " << scan_raw->size() << " pts\n";

    // 2. Pre-process
    auto ref  = preprocess(ref_raw,  cfg.voxel);
    auto scan = preprocess(scan_raw, cfg.voxel);
    std::cout << "After pre-process — ref: " << ref->size()
              << "  scan: " << scan->size() << " pts\n";

    // 3. Normals (radius = 3× voxel is a safe rule of thumb)
    float normal_radius = cfg.voxel * 3.0f;
    float fpfh_radius   = cfg.voxel * 6.0f;

    std::cout << "Computing normals  (radius " << normal_radius*1000 << " mm)...\n";
    auto ref_normals  = computeNormals(ref,  normal_radius);
    auto scan_normals = computeNormals(scan, normal_radius);

    // 4. FPFH features
    std::cout << "Computing FPFH     (radius " << fpfh_radius*1000  << " mm)...\n";
    auto ref_feat  = computeFPFH(ref,  ref_normals,  fpfh_radius);
    auto scan_feat = computeFPFH(scan, scan_normals, fpfh_radius);

    // 5. Global alignment
    std::cout << "Global alignment   (SampleConsensusPrerejective + FPFH)...\n";
    auto [rough, global_ok] = globalAlign(scan, scan_feat, ref, ref_feat, cfg.voxel);
    std::cout << "  Global converged : " << std::boolalpha << global_ok << "\n";

    // 6. ICP fine alignment
    std::cout << "ICP fine alignment...\n";
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(global_ok ? rough : scan);
    icp.setInputTarget(ref);
    icp.setMaxCorrespondenceDistance(cfg.voxel * 3.0f);
    icp.setMaximumIterations(200);
    icp.setTransformationEpsilon(1e-9);
    icp.setEuclideanFitnessEpsilon(1e-9);

    auto aligned = CloudT::Ptr(new CloudT());
    icp.align(*aligned);
    std::cout << "  ICP converged    : " << icp.hasConverged() << "\n";
    std::cout << "  ICP fitness score: " << std::scientific
              << icp.getFitnessScore() << "  m²\n";
    std::cout << "  ICP RMSE         : " << std::fixed
              << std::sqrt(icp.getFitnessScore()) * 1000 << " mm\n";

    // 7–8. Deviation + colour coding
    std::cout << "Computing deviations...\n";
    auto [coloured, deviations] = computeDeviations(
        aligned, ref, ref_normals, cfg.tolerance);

    // 9. Report
    printReport(deviations, cfg.tolerance);

    // Visualise
    if (!canLaunchViewer()) {
        printViewerSkipMessage("Visualisation");
        return 0;
    }

    try {
        pcl::visualization::PCLVisualizer viewer("ADIF — Inspection Result");
        int v1(0), v2(0);
        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        viewer.setBackgroundColor(0.05, 0.05, 0.05, v1);
        viewer.setBackgroundColor(0.05, 0.05, 0.05, v2);

        viewer.addText("Reference (white) + Scanned before ICP (orange)", 5,15,11, 1,1,1,"t1",v1);
        viewer.addText("Deviation map:  Green=OK  Red=Oversized  Blue=Undersized",
                       5, 15, 11, 1,1,1, "t2", v2);

        // Left: reference (white) + raw scan (orange)
        pcl::visualization::PointCloudColorHandlerCustom<PointT>
            cRef(ref, 200, 200, 200);
        viewer.addPointCloud<PointT>(ref, cRef, "ref", v1);
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "ref");

        pcl::visualization::PointCloudColorHandlerCustom<PointT>
            cScan(scan, 230, 140, 50);
        viewer.addPointCloud<PointT>(scan, cScan, "scan", v1);
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scan");

        // Right: deviation heat-map
        viewer.addPointCloud<PointRGB>(coloured, "devmap", v2);
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "devmap");

        viewer.addCoordinateSystem(0.03, "ax", 0);
        setupInitialView(viewer);

        std::cout << "Press 'q' to quit viewer.\n";
        while (!viewer.wasStopped())
            viewer.spinOnce(100);
    } catch (const std::exception& e) {
        std::cout << "Visualisation skipped: " << e.what() << "\n";
    }

    return 0;
}

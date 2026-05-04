// =============================================================================
// ADIF — Automated Dimensional Inspection Framework
// =============================================================================
//
// USAGE:
//   ./adif <reference.pcd> <scanned.pcd>
//          [--tolerance <metres>] [--voxel <metres>]
//          [--pass-threshold <%>] [--latency-target-ms <ms>]
//          [--profile] [--output <path.pcd>]
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
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using PointT    = pcl::PointXYZ;
using PointNT   = pcl::PointNormal;
using PointRGB  = pcl::PointXYZRGB;
using CloudT    = pcl::PointCloud<PointT>;
using CloudNT   = pcl::PointCloud<PointNT>;
using CloudRGB  = pcl::PointCloud<PointRGB>;
using FeatureT  = pcl::FPFHSignature33;
using CloudFeat = pcl::PointCloud<FeatureT>;

struct Config {
    std::string ref_path;
    std::string scan_path;
    float tolerance = 0.001f;
    float voxel = 0.002f;
    float pass_thresh = 95.0f;
    float latency_target_ms = 120.0f;
    bool profile = false;
    std::string output;
};

struct StageTiming {
    std::string name;
    double ms = 0.0;
};

struct ResidualDiagnostics {
    double mean = 0.0;
    double stddev = 0.0;
    double median = 0.0;
    double mad = 0.0;
    double inlier_ratio = 0.0;
    double outlier_ratio = 0.0;
    double confidence_score = 0.0;
    std::vector<std::string> warnings;
    double rmse = 0.0;
    double mse = 0.0;
    double max_abs = 0.0;
    int in_tol = 0;
};

struct RegionMetrology {
    bool valid = false;
    double flatness_mm = 0.0;
    double height_mm = 0.0;
    double diameter_mm = 0.0;
    double position_mm = 0.0;
    double nominal_height_mm = 0.0;
    double nominal_diameter_mm = 0.0;
};

struct NominalModel {
    float z_min = 0.0f;
    float z_max = 0.0f;
    float plate_top_z = 0.0f;
    float boss_top_z = 0.0f;
    Eigen::Vector2f boss_center = Eigen::Vector2f::Zero();
    float boss_radius = 0.02f;
};

Config parseArgs(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: adif <reference.pcd> <scanned.pcd>"
                     " [--tolerance <m>] [--voxel <m>]"
                     " [--pass-threshold <%>] [--latency-target-ms <ms>]"
                     " [--output <path.pcd>] [--profile]\n";
        std::exit(1);
    }

    Config cfg;
    cfg.ref_path = argv[1];
    cfg.scan_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--profile") {
            cfg.profile = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for option: " << key << "\n";
            std::exit(1);
        }

        std::string value = argv[++i];
        if      (key == "--tolerance")         cfg.tolerance = std::stof(value);
        else if (key == "--voxel")             cfg.voxel = std::stof(value);
        else if (key == "--pass-threshold")    cfg.pass_thresh = std::stof(value);
        else if (key == "--latency-target-ms") cfg.latency_target_ms = std::stof(value);
        else if (key == "--output")            cfg.output = value;
        else {
            std::cerr << "Unknown option: " << key << "\n";
            std::exit(1);
        }
    }
    return cfg;
}

CloudT::Ptr preprocess(const CloudT::Ptr& raw, float voxel,
                       float z_min = -0.01f, float z_max = 0.20f,
                       float xy_bound = 0.12f) {
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

pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const CloudT::Ptr& cloud, float radius) {
    auto normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setRadiusSearch(radius);
    auto tree = pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    ne.compute(*normals);
    return normals;
}

CloudFeat::Ptr computeFPFH(const CloudT::Ptr& cloud,
                           const pcl::PointCloud<pcl::Normal>::Ptr& normals,
                           float radius) {
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

std::tuple<CloudT::Ptr, bool, Eigen::Matrix4f> globalAlign(
    const CloudT::Ptr& source,
    const CloudFeat::Ptr& source_feat,
    const CloudT::Ptr& target,
    const CloudFeat::Ptr& target_feat,
    float voxel) {
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
    return {result, align.hasConverged(), align.getFinalTransformation()};
}

CloudNT::Ptr mergeWithNormals(const CloudT::Ptr& cloud,
                              const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    auto out = CloudNT::Ptr(new CloudNT());
    out->resize(cloud->size());
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        out->points[i].x = cloud->points[i].x;
        out->points[i].y = cloud->points[i].y;
        out->points[i].z = cloud->points[i].z;
        out->points[i].normal_x = normals->points[i].normal_x;
        out->points[i].normal_y = normals->points[i].normal_y;
        out->points[i].normal_z = normals->points[i].normal_z;
    }
    out->width = cloud->width;
    out->height = cloud->height;
    out->is_dense = cloud->is_dense;
    return out;
}

std::pair<CloudRGB::Ptr, std::vector<float>> computeDeviations(
    const CloudT::Ptr& aligned_scan,
    const CloudT::Ptr& reference,
    const pcl::PointCloud<pcl::Normal>::Ptr& ref_normals,
    float tolerance) {
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(reference);

    auto coloured = CloudRGB::Ptr(new CloudRGB());
    std::vector<float> deviations;
    deviations.reserve(aligned_scan->size());

    std::vector<int> nn_idx(1);
    std::vector<float> nn_dist(1);

    for (const auto& pt : aligned_scan->points) {
        kdtree.nearestKSearch(pt, 1, nn_idx, nn_dist);
        int idx = nn_idx[0];
        const auto& ref_pt = reference->points[idx];
        const auto& ref_n = ref_normals->points[idx];

        Eigen::Vector3f disp(pt.x - ref_pt.x, pt.y - ref_pt.y, pt.z - ref_pt.z);
        Eigen::Vector3f nrm(ref_n.normal_x, ref_n.normal_y, ref_n.normal_z);
        float dev = (nrm.norm() > 0.5f) ? disp.dot(nrm.normalized()) : std::sqrt(nn_dist[0]);
        deviations.push_back(dev);

        PointRGB p{};
        p.x = pt.x; p.y = pt.y; p.z = pt.z;
        if (std::abs(dev) <= tolerance) { p.r = 60; p.g = 200; p.b = 60; }
        else if (dev > tolerance)      { p.r = 220; p.g = 60; p.b = 60; }
        else                            { p.r = 60; p.g = 60; p.b = 220; }
        coloured->push_back(p);
    }
    return {coloured, deviations};
}

double robustMedian(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::size_t n = values.size();
    std::nth_element(values.begin(), values.begin() + n / 2, values.end());
    double m = values[n / 2];
    if (n % 2 == 0) {
        auto max_it = std::max_element(values.begin(), values.begin() + n / 2);
        m = 0.5 * (m + *max_it);
    }
    return m;
}

NominalModel estimateNominalModel(const CloudT::Ptr& reference) {
    NominalModel nm;
    if (reference->empty()) return nm;

    nm.z_min = nm.z_max = reference->points.front().z;
    for (const auto& p : reference->points) {
        nm.z_min = std::min(nm.z_min, p.z);
        nm.z_max = std::max(nm.z_max, p.z);
    }
    const float z_range = std::max(1e-6f, nm.z_max - nm.z_min);
    const float boss_top_gate = nm.z_max - std::max(0.0008f, 0.08f * z_range);

    std::vector<Eigen::Vector2f> top_xy;
    for (const auto& p : reference->points)
        if (p.z >= boss_top_gate) top_xy.emplace_back(p.x, p.y);

    if (!top_xy.empty()) {
        Eigen::Vector2f c = Eigen::Vector2f::Zero();
        for (const auto& xy : top_xy) c += xy;
        c /= static_cast<float>(top_xy.size());
        nm.boss_center = c;

        float max_r = 0.0f;
        for (const auto& xy : top_xy)
            max_r = std::max(max_r, (xy - c).norm());
        nm.boss_radius = std::max(0.004f, max_r);
    }

    std::vector<double> plate_candidates;
    for (const auto& p : reference->points) {
        float nz = (p.z - nm.z_min) / z_range;
        float r = std::hypot(p.x - nm.boss_center.x(), p.y - nm.boss_center.y());
        if (nz > 0.25f && nz < 0.60f && r > nm.boss_radius * 1.15f)
            plate_candidates.push_back(p.z);
    }

    nm.plate_top_z = plate_candidates.empty()
        ? (nm.z_min + 0.38f * z_range)
        : static_cast<float>(robustMedian(plate_candidates));
    nm.boss_top_z = nm.z_max;
    return nm;
}

RegionMetrology computeRegionMetrology(const CloudT::Ptr& aligned_scan,
                                       const NominalModel& nm) {
    RegionMetrology out;

    std::vector<Eigen::Vector3f> plate_roi;
    std::vector<Eigen::Vector3f> boss_top_roi;
    std::vector<Eigen::Vector3f> boss_side_roi;

    for (const auto& p : aligned_scan->points) {
        float r = std::hypot(p.x - nm.boss_center.x(), p.y - nm.boss_center.y());
        if (std::abs(p.z - nm.plate_top_z) <= 0.0030f && r > nm.boss_radius * 1.20f)
            plate_roi.emplace_back(p.x, p.y, p.z);
        if (std::abs(p.z - nm.boss_top_z) <= 0.0035f && r <= nm.boss_radius * 0.80f)
            boss_top_roi.emplace_back(p.x, p.y, p.z);
        if (p.z > (nm.plate_top_z + 0.001f) && p.z < (nm.boss_top_z - 0.001f) &&
            r >= nm.boss_radius * 0.70f && r <= nm.boss_radius * 1.30f)
            boss_side_roi.emplace_back(p.x, p.y, p.z);
    }

    if (plate_roi.size() < 20 || boss_top_roi.size() < 20 || boss_side_roi.size() < 20)
        return out;

    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto& p : plate_roi) c += p;
    c /= static_cast<float>(plate_roi.size());

    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (const auto& p : plate_roi) {
        Eigen::Vector3f d = p - c;
        cov += d * d.transpose();
    }
    cov /= static_cast<float>(plate_roi.size());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
    Eigen::Vector3f n = es.eigenvectors().col(0).normalized();

    float d_min = std::numeric_limits<float>::max();
    float d_max = -std::numeric_limits<float>::max();
    std::vector<double> plate_z;
    for (const auto& p : plate_roi) {
        float d = n.dot(p - c);
        d_min = std::min(d_min, d);
        d_max = std::max(d_max, d);
        plate_z.push_back(p.z());
    }
    out.flatness_mm = static_cast<double>(d_max - d_min) * 1000.0;

    std::vector<double> boss_top_z;
    Eigen::Vector2f c_meas = Eigen::Vector2f::Zero();
    for (const auto& p : boss_top_roi) {
        boss_top_z.push_back(p.z());
        c_meas += p.head<2>();
    }
    c_meas /= static_cast<float>(boss_top_roi.size());

    out.height_mm = (robustMedian(boss_top_z) - robustMedian(plate_z)) * 1000.0;

    std::vector<double> side_r;
    for (const auto& p : boss_side_roi)
        side_r.push_back(std::hypot(p.x() - c_meas.x(), p.y() - c_meas.y()));
    out.diameter_mm = 2.0 * robustMedian(side_r) * 1000.0;

    out.position_mm = (c_meas - nm.boss_center).norm() * 1000.0;
    out.nominal_height_mm = static_cast<double>(nm.boss_top_z - nm.plate_top_z) * 1000.0;
    out.nominal_diameter_mm = static_cast<double>(2.0f * nm.boss_radius) * 1000.0;
    out.valid = true;
    return out;
}

ResidualDiagnostics computeResidualDiagnostics(const std::vector<float>& devs, float tolerance) {
    ResidualDiagnostics d;
    if (devs.empty()) return d;

    std::vector<double> vals;
    vals.reserve(devs.size());
    for (float v : devs) vals.push_back(static_cast<double>(v));

    d.mean = std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<double>(vals.size());

    double sq = 0.0;
    for (double v : vals) {
        sq += (v - d.mean) * (v - d.mean);
        d.max_abs = std::max(d.max_abs, std::abs(v));
        if (std::abs(v) <= tolerance) ++d.in_tol;
    }

    d.mse = sq / static_cast<double>(vals.size());
    d.rmse = std::sqrt(d.mse);
    d.stddev = std::sqrt(d.mse);
    d.inlier_ratio = static_cast<double>(d.in_tol) / static_cast<double>(vals.size());
    d.median = robustMedian(vals);

    std::vector<double> abs_dev;
    abs_dev.reserve(vals.size());
    for (double v : vals) abs_dev.push_back(std::abs(v - d.median));
    d.mad = robustMedian(abs_dev);

    double robust_sigma = 1.4826 * d.mad;
    double outlier_gate = std::max(3.0 * robust_sigma, static_cast<double>(tolerance));
    int outliers = 0;
    for (double v : vals)
        if (std::abs(v - d.median) > outlier_gate) ++outliers;
    d.outlier_ratio = static_cast<double>(outliers) / static_cast<double>(vals.size());

    double bias_term = std::exp(-std::abs(d.mean) / std::max(1e-12, 0.35 * tolerance));
    double spread_term = std::exp(-d.stddev / std::max(1e-12, 0.70 * tolerance));
    d.confidence_score = 100.0 * std::clamp(
        0.55 * d.inlier_ratio + 0.25 * bias_term + 0.20 * spread_term,
        0.0, 1.0);

    if (d.outlier_ratio > 0.08) d.warnings.emplace_back("HIGH_OUTLIER_RATIO");
    if (std::abs(d.mean) > 0.35 * tolerance) d.warnings.emplace_back("SYSTEMATIC_BIAS");
    if (d.stddev > 0.80 * tolerance) d.warnings.emplace_back("HIGH_SPREAD");
    if (d.confidence_score < 75.0) d.warnings.emplace_back("LOW_CONFIDENCE");
    return d;
}

void printReport(const std::vector<float>& devs,
                 const ResidualDiagnostics& diag,
                 const RegionMetrology& met,
                 const std::vector<StageTiming>& timings,
                 float tolerance,
                 float pass_thresh,
                 float latency_target_ms,
                 bool profile_mode) {
    double total_latency_ms = 0.0;
    for (const auto& t : timings) total_latency_ms += t.ms;

    bool geometric_pass = (100.0 * diag.inlier_ratio) >= pass_thresh;
    bool latency_pass = total_latency_ms <= latency_target_ms;
    bool overall_pass = geometric_pass && latency_pass;

    const std::string sep(72, '=');
    std::cout << "\n" << sep << "\n";
    std::cout << "  ADIF INSPECTION REPORT\n";
    std::cout << sep << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Points analysed         : " << devs.size() << "\n";
    std::cout << "  Tolerance (+-)          : " << tolerance * 1000 << " mm\n";
    std::cout << "  MSE                     : " << diag.mse * 1e6 << " mm^2\n";
    std::cout << "  RMSE                    : " << diag.rmse * 1000 << " mm\n";
    std::cout << "  Max |deviation|         : " << diag.max_abs * 1000 << " mm\n";
    std::cout << "  In tolerance            : " << diag.in_tol << " / " << devs.size()
              << "  (" << std::setprecision(1) << (100.0 * diag.inlier_ratio) << "%)\n";

    std::cout << "\n  Residual diagnostics:\n";
    std::cout << "    mean/std/median/MAD   : " << std::setprecision(3)
              << diag.mean * 1000 << " / " << diag.stddev * 1000 << " / "
              << diag.median * 1000 << " / " << diag.mad * 1000 << " mm\n";
    std::cout << "    inlier / outlier      : " << std::setprecision(2)
              << (100.0 * diag.inlier_ratio) << "% / "
              << (100.0 * diag.outlier_ratio) << "%\n";
    std::cout << "    confidence score      : " << diag.confidence_score << " / 100\n";

    std::cout << "\n  Region metrology:\n";
    if (met.valid) {
        std::cout << "    Flatness              : " << met.flatness_mm << " mm\n";
        std::cout << "    Height                : " << met.height_mm << " mm"
                  << "  (nominal " << met.nominal_height_mm << " mm)\n";
        std::cout << "    Diameter              : " << met.diameter_mm << " mm"
                  << "  (nominal " << met.nominal_diameter_mm << " mm)\n";
        std::cout << "    Position offset       : " << met.position_mm << " mm\n";
    } else {
        std::cout << "    insufficient ROI support after filtering/alignment\n";
    }

    std::cout << "\n  Real-time metrics:\n";
    std::cout << "    latency total         : " << total_latency_ms << " ms\n";
    std::cout << "    latency target        : " << latency_target_ms << " ms\n";
    std::cout << "    throughput            : " << (total_latency_ms > 0.0 ? 1000.0 / total_latency_ms : 0.0)
              << " frames/s\n";

    if (profile_mode && !timings.empty()) {
        std::cout << "\n  Stage timings (--profile):\n";
        for (const auto& st : timings) {
            double pct = (total_latency_ms > 0.0) ? (100.0 * st.ms / total_latency_ms) : 0.0;
            std::cout << "    - " << std::setw(24) << std::left << st.name
                      << " : " << std::setw(9) << std::right << std::setprecision(3)
                      << st.ms << " ms  (" << std::setw(6) << std::setprecision(2)
                      << pct << "%)\n";
        }
    }

    std::cout << "\n  Decision:\n";
    std::cout << "    geometric             : " << (geometric_pass ? "PASS" : "FAIL") << "\n";
    std::cout << "    latency               : " << (latency_pass ? "PASS" : "FAIL") << "\n";
    std::cout << "    OVERALL               : " << (overall_pass ? "PASS" : "FAIL") << "\n";
    std::cout << sep << "\n\n";
}

int main(int argc, char** argv) {
    Config cfg = parseArgs(argc, argv);
    std::vector<StageTiming> timings;

    auto runStage = [&](const std::string& name, auto&& fn) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        timings.push_back({name, std::chrono::duration<double, std::milli>(t1 - t0).count()});
    };

    auto ref_raw = CloudT::Ptr(new CloudT());
    auto scan_raw = CloudT::Ptr(new CloudT());
    bool load_ok = true;
    runStage("load", [&] {
        if (pcl::io::loadPCDFile(cfg.ref_path, *ref_raw) < 0 ||
            pcl::io::loadPCDFile(cfg.scan_path, *scan_raw) < 0) {
            load_ok = false;
        }
    });
    if (!load_ok) {
        std::cerr << "Failed to load PCD files.\n";
        return 1;
    }

    CloudT::Ptr ref(new CloudT()), scan(new CloudT());
    runStage("preprocess", [&] {
        ref = preprocess(ref_raw, cfg.voxel);
        scan = preprocess(scan_raw, cfg.voxel);
    });

    auto nominal = estimateNominalModel(ref);

    float normal_radius = cfg.voxel * 3.0f;
    float fpfh_radius = cfg.voxel * 6.0f;
    pcl::PointCloud<pcl::Normal>::Ptr ref_normals, scan_normals;

    runStage("normals (parallel ref+scan)", [&] {
        auto f1 = std::async(std::launch::async, [&] { return computeNormals(ref, normal_radius); });
        auto f2 = std::async(std::launch::async, [&] { return computeNormals(scan, normal_radius); });
        ref_normals = f1.get();
        scan_normals = f2.get();
    });

    CloudFeat::Ptr ref_feat, scan_feat;
    runStage("FPFH (parallel ref+scan)", [&] {
        auto f1 = std::async(std::launch::async, [&] { return computeFPFH(ref, ref_normals, fpfh_radius); });
        auto f2 = std::async(std::launch::async, [&] { return computeFPFH(scan, scan_normals, fpfh_radius); });
        ref_feat = f1.get();
        scan_feat = f2.get();
    });

    CloudT::Ptr rough;
    bool global_ok = false;
    Eigen::Matrix4f global_tf = Eigen::Matrix4f::Identity();
    runStage("global alignment", [&] {
        auto ret = globalAlign(scan, scan_feat, ref, ref_feat, cfg.voxel);
        rough = std::get<0>(ret);
        global_ok = std::get<1>(ret);
        global_tf = std::get<2>(ret);
    });

    const Eigen::Matrix3f R_global =
        (global_ok ? global_tf : Eigen::Matrix4f::Identity()).block<3, 3>(0, 0);
    auto rough_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>(*scan_normals));
    for (auto& n : rough_normals->points) {
        Eigen::Vector3f nv(n.normal_x, n.normal_y, n.normal_z);
        nv = R_global * nv;
        n.normal_x = nv.x(); n.normal_y = nv.y(); n.normal_z = nv.z();
    }

    auto source_nt = mergeWithNormals(global_ok ? rough : scan, rough_normals);
    auto target_nt = mergeWithNormals(ref, ref_normals);

    pcl::IterativeClosestPointWithNormals<PointNT, PointNT> icp;
    icp.setInputSource(source_nt);
    icp.setInputTarget(target_nt);
    icp.setMaxCorrespondenceDistance(cfg.voxel * 3.0f);
    icp.setMaximumIterations(200);
    icp.setTransformationEpsilon(1e-9);
    icp.setEuclideanFitnessEpsilon(1e-9);

    auto aligned_nt = CloudNT::Ptr(new CloudNT());
    runStage("ICP fine alignment", [&] { icp.align(*aligned_nt); });

    auto aligned = CloudT::Ptr(new CloudT());
    pcl::copyPointCloud(*aligned_nt, *aligned);

    CloudRGB::Ptr coloured;
    std::vector<float> deviations;
    runStage("deviation map", [&] {
        auto pair = computeDeviations(aligned, ref, ref_normals, cfg.tolerance);
        coloured = pair.first;
        deviations = std::move(pair.second);
    });

    ResidualDiagnostics diag;
    RegionMetrology met;
    runStage("metrology + diagnostics", [&] {
        diag = computeResidualDiagnostics(deviations, cfg.tolerance);
        met = computeRegionMetrology(aligned, nominal);
    });

    printReport(deviations, diag, met, timings,
                cfg.tolerance, cfg.pass_thresh, cfg.latency_target_ms, cfg.profile);

    if (!cfg.output.empty()) {
        pcl::io::savePCDFileBinary(cfg.output, *coloured);
        std::cout << "Deviation map saved → " << cfg.output << "\n";
    }

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

        viewer.addText("Reference (white) + Scanned before ICP (orange)", 5, 15, 11, 1, 1, 1, "t1", v1);
        viewer.addText("Deviation map:  Green=OK  Red=Oversized  Blue=Undersized",
                       5, 15, 11, 1, 1, 1, "t2", v2);

        pcl::visualization::PointCloudColorHandlerCustom<PointT> cRef(ref, 200, 200, 200);
        viewer.addPointCloud<PointT>(ref, cRef, "ref", v1);
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "ref");

        pcl::visualization::PointCloudColorHandlerCustom<PointT> cScan(scan, 230, 140, 50);
        viewer.addPointCloud<PointT>(scan, cScan, "scan", v1);
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scan");

        viewer.addPointCloud<PointRGB>(coloured, "devmap", v2);
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "devmap");

        viewer.addCoordinateSystem(0.03, "ax", 0);
        setupInitialView(viewer);

        std::cout << "Press 'q' to quit viewer.\n";
        while (!viewer.wasStopped()) viewer.spinOnce(100);
    } catch (const std::exception& e) {
        std::cout << "Visualisation skipped: " << e.what() << "\n";
    }

    return 0;
}

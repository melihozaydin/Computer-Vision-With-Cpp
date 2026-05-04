// =============================================================================
// ADIF GUI — Interactive Dimensional Inspection GUI
// =============================================================================
// Load a PCD file, select ROIs by drawing a rubber-band box, then run
// metrology tools on the selection.
//
// Usage:
//   ./adif_gui <scan.pcd> [<reference.pcd>]
//
// Controls:
//   x          toggle ROI-draw mode  (press x, then drag a box)
//   f          flatness on current ROI
//   c          circle fit (diameter + centre) on current ROI
//   t          statistics on current ROI
//   1 / 2      tag current ROI as Region 1 / Region 2
//   d          height delta between Region 1 and Region 2
//   n          toggle normal arrows for current ROI
//   v          toggle deviation-map vs reference (needs reference.pcd)
//   e          export current ROI to roi_export.pcd
//   0          clear all selections
//   i          print controls to stdout
//   q          quit
//
// Note: keys f, r, s, p, g etc. are also used by PCL's built-in handler.
//       The tools above will still fire alongside the PCL defaults.
// =============================================================================

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include "../pcl_viewer_utils.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using PointT   = pcl::PointXYZ;
using PointRGB = pcl::PointXYZRGB;
using CloudT   = pcl::PointCloud<PointT>;
using CloudRGB = pcl::PointCloud<PointRGB>;

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
static double medianOf(std::vector<double> v) {
    if (v.empty()) return 0.0;
    size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    double m = v[n / 2];
    if (n % 2 == 0) {
        auto it = std::max_element(v.begin(), v.begin() + n / 2);
        m = 0.5 * (m + *it);
    }
    return m;
}

static void printControls() {
    std::cout << "\n"
        "  ┌─────────────────────────────────────────────────────────┐\n"
        "  │  ADIF GUI — controls                                    │\n"
        "  ├─────────────────────────────────────────────────────────┤\n"
        "  │  x          toggle ROI-draw mode (then drag a box)      │\n"
        "  │  f          flatness on current ROI                     │\n"
        "  │  c          circle fit (diameter + centre)              │\n"
        "  │  t          statistics on current ROI                   │\n"
        "  │  1 / 2      tag ROI as Region 1 / Region 2             │\n"
        "  │  d          height delta (Region 2 − Region 1)          │\n"
        "  │  n          toggle normal arrows for ROI                │\n"
        "  │  v          toggle deviation map vs reference           │\n"
        "  │  e          export ROI → roi_export.pcd                 │\n"
        "  │  0          clear all selections                        │\n"
        "  │  i          print this help                             │\n"
        "  │  q          quit                                        │\n"
        "  └─────────────────────────────────────────────────────────┘\n\n";
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
struct AppState {
    CloudT::Ptr   cloud;                  // loaded scan cloud
    CloudT::Ptr   reference;              // optional reference (may be null)
    CloudRGB::Ptr display;                // coloured display copy

    std::vector<int> roi;                 // current rubber-band selection
    std::vector<int> region1;             // tagged for height-delta
    std::vector<int> region2;             // tagged for height-delta

    bool showing_normals  = false;
    bool deviation_mode   = false;        // v toggle

    pcl::visualization::PCLVisualizer* viewer = nullptr;
    std::string status_a;                 // top status line
    std::string status_b;                 // result line 1
    std::string status_c;                 // result line 2
};

// ---------------------------------------------------------------------------
// Overlay text helpers (add-or-update pattern)
// ---------------------------------------------------------------------------
static void setText(pcl::visualization::PCLVisualizer& viewer,
                    const std::string& text, int x, int y, int sz,
                    double r, double g, double b, const std::string& id) {
    if (!viewer.updateText(text, x, y, sz, r, g, b, id))
        viewer.addText(text, x, y, sz, r, g, b, id);
}

// ---------------------------------------------------------------------------
// Rebuild display cloud colours and refresh viewer
// ---------------------------------------------------------------------------
static void rebuildDisplay(AppState& st) {
    if (!st.display || st.display->size() != st.cloud->size()) {
        st.display.reset(new CloudRGB());
        st.display->resize(st.cloud->size());
        for (size_t i = 0; i < st.cloud->size(); ++i) {
            st.display->points[i].x = st.cloud->points[i].x;
            st.display->points[i].y = st.cloud->points[i].y;
            st.display->points[i].z = st.cloud->points[i].z;
        }
        st.display->width  = static_cast<uint32_t>(st.cloud->size());
        st.display->height = 1;
        st.display->is_dense = st.cloud->is_dense;
    }

    if (st.deviation_mode && st.reference && !st.reference->empty()) {
        // Colour entire cloud by distance to reference
        pcl::KdTreeFLANN<PointT> kd;
        kd.setInputCloud(st.reference);
        const float tol = 0.001f; // 1 mm
        for (size_t i = 0; i < st.cloud->size(); ++i) {
            std::vector<int>   nn(1);
            std::vector<float> nd(1);
            kd.nearestKSearch(st.cloud->points[i], 1, nn, nd);
            float dist = std::sqrt(nd[0]);
            auto& p = st.display->points[i];
            if (dist <= tol)      { p.r = 60;  p.g = 200; p.b = 60;  }
            else if (dist > tol)  { p.r = 220; p.g = 60;  p.b = 60;  }
        }
    } else {
        // Normal mode: grey base, tinted ROI / regions
        for (auto& p : st.display->points) { p.r = 180; p.g = 180; p.b = 180; }
        for (int i : st.region1) {
            auto& p = st.display->points[i]; p.r = 0; p.g = 200; p.b = 220; // cyan
        }
        for (int i : st.region2) {
            auto& p = st.display->points[i]; p.r = 220; p.g = 0; p.b = 200; // magenta
        }
        for (int i : st.roi) {
            auto& p = st.display->points[i]; p.r = 255; p.g = 220; p.b = 0; // yellow
        }
    }

    pcl::visualization::PointCloudColorHandlerRGBField<PointRGB> rgb(st.display);
    if (!st.viewer->updatePointCloud<PointRGB>(st.display, rgb, "main")) {
        st.viewer->addPointCloud<PointRGB>(st.display, rgb, "main");
    }
    st.viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "main");
}

// ---------------------------------------------------------------------------
// Update overlay status text
// ---------------------------------------------------------------------------
static void updateStatus(AppState& st) {
    const std::string roi_s   = st.roi.empty()     ? "none" : std::to_string(st.roi.size())     + " pts";
    const std::string r1_s    = st.region1.empty() ? "none" : std::to_string(st.region1.size()) + " pts";
    const std::string r2_s    = st.region2.empty() ? "none" : std::to_string(st.region2.size()) + " pts";
    const std::string dev_s   = st.deviation_mode  ? "  [DEVIATION MAP ON]" : "";

    st.status_a = "ROI: " + roi_s + "   R1: " + r1_s + "   R2: " + r2_s + dev_s;

    setText(*st.viewer, st.status_a, 5, 55, 12, 0.9, 0.9, 0.5, "st_a");
    setText(*st.viewer, st.status_b, 5, 35, 12, 0.5, 1.0, 0.5, "st_b");
    setText(*st.viewer, st.status_c, 5, 15, 12, 0.5, 1.0, 0.5, "st_c");
}

static void setResult(AppState& st, const std::string& line1,
                      const std::string& line2 = "") {
    st.status_b = line1;
    st.status_c = line2;
    std::cout << "[ADIF GUI] " << line1 << "\n";
    if (!line2.empty()) std::cout << "           " << line2 << "\n";
    updateStatus(st);
}

// ---------------------------------------------------------------------------
// Analysis tools
// ---------------------------------------------------------------------------
static void runStats(AppState& st) {
    if (st.roi.empty()) { setResult(st, "Stats: no ROI selected"); return; }

    double sx = 0, sy = 0, sz = 0;
    double minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9, minz = 1e9, maxz = -1e9;
    for (int i : st.roi) {
        const auto& p = st.cloud->points[i];
        sx += p.x; sy += p.y; sz += p.z;
        minx = std::min(minx, (double)p.x); maxx = std::max(maxx, (double)p.x);
        miny = std::min(miny, (double)p.y); maxy = std::max(maxy, (double)p.y);
        minz = std::min(minz, (double)p.z); maxz = std::max(maxz, (double)p.z);
    }
    size_t n = st.roi.size();
    std::ostringstream a, b;
    a << std::fixed << std::setprecision(2)
      << "[Stats] n=" << n
      << "  bbox: " << (maxx - minx) * 1e3 << " x " << (maxy - miny) * 1e3
      << " x " << (maxz - minz) * 1e3 << " mm";
    b << "  centroid: (" << sx / n * 1e3 << ", " << sy / n * 1e3
      << ", " << sz / n * 1e3 << ") mm";
    setResult(st, a.str(), b.str());
}

static void runFlatness(AppState& st) {
    if (st.roi.size() < 4) { setResult(st, "Flatness: need >= 4 pts in ROI"); return; }

    Eigen::Vector3d cen = Eigen::Vector3d::Zero();
    for (int i : st.roi) {
        cen.x() += st.cloud->points[i].x;
        cen.y() += st.cloud->points[i].y;
        cen.z() += st.cloud->points[i].z;
    }
    cen /= static_cast<double>(st.roi.size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int i : st.roi) {
        Eigen::Vector3d d(st.cloud->points[i].x - cen.x(),
                          st.cloud->points[i].y - cen.y(),
                          st.cloud->points[i].z - cen.z());
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(st.roi.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d normal = es.eigenvectors().col(0).normalized(); // min eigenvalue

    double dmin =  1e9, dmax = -1e9;
    for (int i : st.roi) {
        Eigen::Vector3d p(st.cloud->points[i].x,
                          st.cloud->points[i].y,
                          st.cloud->points[i].z);
        double proj = normal.dot(p - cen);
        dmin = std::min(dmin, proj);
        dmax = std::max(dmax, proj);
    }

    std::ostringstream os;
    os << std::fixed << std::setprecision(4)
       << "[Flatness] " << (dmax - dmin) * 1000.0 << " mm  (n=" << st.roi.size() << ")";
    setResult(st, os.str());
}

static void runCircleFit(AppState& st) {
    if (st.roi.size() < 5) { setResult(st, "Circle fit: need >= 5 pts in ROI"); return; }

    // Algebraic 2D circle fit on XY projection:
    // Solve  a*x + b*y + c = x²+y²  in the least-squares sense.
    size_t n = st.roi.size();
    Eigen::MatrixXd A(n, 3);
    Eigen::VectorXd bv(n);
    for (size_t k = 0; k < n; ++k) {
        double x = st.cloud->points[st.roi[k]].x;
        double y = st.cloud->points[st.roi[k]].y;
        A(k, 0) = x; A(k, 1) = y; A(k, 2) = 1.0;
        bv(k) = x * x + y * y;
    }
    Eigen::Vector3d sol = A.colPivHouseholderQr().solve(bv);
    double cx = sol(0) / 2.0;
    double cy = sol(1) / 2.0;
    double r  = std::sqrt(std::max(0.0, sol(2) + cx * cx + cy * cy));

    std::ostringstream a, b;
    a << std::fixed << std::setprecision(3)
      << "[Circle] diameter=" << 2.0 * r * 1e3 << " mm  radius=" << r * 1e3 << " mm";
    b << "  centre=(" << cx * 1e3 << ", " << cy * 1e3 << ") mm  n=" << n;
    setResult(st, a.str(), b.str());
}

static void runHeightDelta(AppState& st) {
    if (st.region1.empty() || st.region2.empty()) {
        setResult(st, "Height delta: tag regions 1 and 2 first (keys 1, 2)");
        return;
    }
    std::vector<double> z1, z2;
    for (int i : st.region1) z1.push_back(st.cloud->points[i].z);
    for (int i : st.region2) z2.push_back(st.cloud->points[i].z);

    double delta_mm = (medianOf(z2) - medianOf(z1)) * 1000.0;
    std::ostringstream a, b;
    a << std::fixed << std::setprecision(4)
      << "[Height delta] R2 - R1 = " << delta_mm << " mm";
    b << "  R1 median Z=" << medianOf(z1) * 1e3 << " mm"
      << "  R2 median Z=" << medianOf(z2) * 1e3 << " mm";
    setResult(st, a.str(), b.str());
}

static void runNormals(AppState& st) {
    if (st.showing_normals) {
        st.viewer->removePointCloud("normals");
        st.showing_normals = false;
        setResult(st, "Normals: hidden");
        return;
    }
    if (st.roi.size() < 5) { setResult(st, "Normals: need >= 5 pts in ROI"); return; }

    CloudT::Ptr sub(new CloudT());
    sub->reserve(st.roi.size());
    for (int i : st.roi) sub->push_back(st.cloud->points[i]);

    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(sub);
    ne.setSearchMethod(pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>()));
    ne.setKSearch(std::min(static_cast<int>(sub->size()), 10));
    auto normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
    ne.compute(*normals);

    st.viewer->addPointCloudNormals<PointT, pcl::Normal>(sub, normals, 1, 0.003f, "normals");
    st.viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 0.3, "normals");
    st.showing_normals = true;
    setResult(st, "Normals: shown for ROI (" + std::to_string(sub->size()) + " pts)");
}

static void runDeviation(AppState& st) {
    if (!st.reference || st.reference->empty()) {
        setResult(st, "Deviation map: no reference cloud loaded",
                  "  Pass reference.pcd as the 2nd argument");
        return;
    }
    st.deviation_mode = !st.deviation_mode;
    rebuildDisplay(st);
    setResult(st, std::string("Deviation map: ") + (st.deviation_mode ? "ON" : "OFF"));
}

static void exportROI(AppState& st) {
    if (st.roi.empty()) { setResult(st, "Export: no ROI selected"); return; }

    CloudT::Ptr sub(new CloudT());
    sub->reserve(st.roi.size());
    for (int i : st.roi) sub->push_back(st.cloud->points[i]);

    const std::string path = "roi_export.pcd";
    if (pcl::io::savePCDFileBinary(path, *sub) >= 0) {
        setResult(st, "Exported " + std::to_string(sub->size()) + " pts -> " + path);
    } else {
        setResult(st, "Export failed: could not write " + path);
    }
}

static void clearAll(AppState& st) {
    st.roi.clear();
    st.region1.clear();
    st.region2.clear();
    st.deviation_mode = false;
    if (st.showing_normals) {
        st.viewer->removePointCloud("normals");
        st.showing_normals = false;
    }
    rebuildDisplay(st);
    setResult(st, "All selections cleared");
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------
static void areaPickCallback(const pcl::visualization::AreaPickingEvent& event,
                             void* userdata) {
    AppState& st = *reinterpret_cast<AppState*>(userdata);

    std::vector<int> indices;
    if (!event.getPointsIndices(indices) || indices.empty()) {
        setResult(st, "ROI draw: no points in selection");
        return;
    }

    // Clamp indices to valid range
    const auto max_idx = static_cast<int>(st.cloud->size()) - 1;
    indices.erase(std::remove_if(indices.begin(), indices.end(),
                                 [max_idx](int i) { return i < 0 || i > max_idx; }),
                  indices.end());

    st.roi = indices;
    rebuildDisplay(st);
    setResult(st, "ROI selected: " + std::to_string(st.roi.size()) + " pts");
}

static void keyboardCallback(const pcl::visualization::KeyboardEvent& event,
                             void* userdata) {
    if (!event.keyDown()) return;
    AppState& st = *reinterpret_cast<AppState*>(userdata);

    const std::string key = event.getKeySym();

    if      (key == "f")      runFlatness(st);
    else if (key == "c")      runCircleFit(st);
    else if (key == "t")      runStats(st);
    else if (key == "d")      runHeightDelta(st);
    else if (key == "n")      runNormals(st);
    else if (key == "v")      runDeviation(st);
    else if (key == "e")      exportROI(st);
    else if (key == "0")      clearAll(st);
    else if (key == "i")      printControls();
    else if (key == "1") {
        if (!st.roi.empty()) {
            st.region1 = st.roi;
            rebuildDisplay(st);
            setResult(st, "Region 1 tagged: " + std::to_string(st.region1.size()) + " pts");
        } else {
            setResult(st, "Tag R1: draw an ROI first");
        }
    }
    else if (key == "2") {
        if (!st.roi.empty()) {
            st.region2 = st.roi;
            rebuildDisplay(st);
            setResult(st, "Region 2 tagged: " + std::to_string(st.region2.size()) + " pts");
        } else {
            setResult(st, "Tag R2: draw an ROI first");
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scan.pcd> [<reference.pcd>]\n";
        return 1;
    }

    AppState st;
    st.cloud.reset(new CloudT());
    if (pcl::io::loadPCDFile(argv[1], *st.cloud) < 0) {
        std::cerr << "Failed to load: " << argv[1] << "\n";
        return 1;
    }
    std::cout << "Loaded scan: " << st.cloud->size() << " points  (" << argv[1] << ")\n";

    if (argc >= 3) {
        st.reference.reset(new CloudT());
        if (pcl::io::loadPCDFile(argv[2], *st.reference) < 0) {
            std::cerr << "Warning: could not load reference: " << argv[2] << "\n";
            st.reference.reset();
        } else {
            std::cout << "Loaded reference: " << st.reference->size()
                      << " points  (" << argv[2] << ")\n";
        }
    }

    printControls();

    if (!canLaunchViewer()) {
        printViewerSkipMessage("ADIF GUI");
        return 0;
    }

    try {
        pcl::visualization::PCLVisualizer viewer("ADIF GUI — Dimensional Inspection");
        st.viewer = &viewer;

        viewer.setBackgroundColor(0.08, 0.08, 0.08);

        // Initial display cloud
        rebuildDisplay(st);
        updateStatus(st);

        // Help line pinned at top
        viewer.addText("x=draw-ROI  f=flatness  c=circle  t=stats  1/2=tag  d=delta"
                       "  n=normals  v=deviation  e=export  0=clear  i=help",
                       5, 5, 10, 0.6, 0.6, 0.6, "help");

        viewer.registerAreaPickingCallback(areaPickCallback, &st);
        viewer.registerKeyboardCallback(keyboardCallback, &st);

        viewer.addCoordinateSystem(0.03, "axes", 0);
        setupInitialView(viewer);

        std::cout << "Viewer ready. Press 'x' to toggle ROI-draw mode, then drag.\n";
        while (!viewer.wasStopped())
            viewer.spinOnce(50);

    } catch (const std::exception& e) {
        std::cerr << "Viewer error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

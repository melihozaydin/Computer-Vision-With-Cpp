// =============================================================================
// 09 - Region Metrology Primitives
// =============================================================================
//
// PURPOSE:
//   Isolate four inspection measurements used later in ADIF:
//     - Flatness (top plate region)
//     - Height   (boss top vs plate top)
//     - Diameter (boss side ring)
//     - Position (boss centre offset)
//
// NOTE:
//   Synthetic deterministic geometry keeps this lesson interview-friendly and
//   reproducible without external scanner input.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

struct Pt3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const std::size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    double m = v[n / 2];
    if (n % 2 == 0) {
        auto max_it = std::max_element(v.begin(), v.begin() + n / 2);
        m = 0.5 * (m + *max_it);
    }
    return m;
}

int main() {
    constexpr double plate_w = 0.150;
    constexpr double plate_d = 0.100;
    constexpr double plate_top_z = 0.005;
    constexpr double boss_h = 0.008;
    constexpr double boss_r_nominal = 0.020;

    // Simulated measured part:
    const double boss_center_x = 0.0007; // 0.7 mm true-position offset
    const double boss_center_y = -0.0004;
    const double boss_r_measured = 0.0197; // 19.7 mm diameter shrink
    const double boss_top_measured_z = plate_top_z + 0.0076; // nominal 8.0 mm -> measured 7.6 mm

    std::mt19937 rng(1234);
    std::normal_distribution<double> nxy(0.0, 0.0002);
    std::normal_distribution<double> nz(0.0, 0.00015);

    std::vector<Pt3> plate_pts;
    std::vector<Pt3> boss_top_pts;
    std::vector<Pt3> boss_side_pts;

    // Plate top points (with slight warp + noise)
    for (int i = 0; i < 140; ++i) {
        for (int j = 0; j < 100; ++j) {
            double x = -plate_w / 2 + plate_w * i / 139.0;
            double y = -plate_d / 2 + plate_d * j / 99.0;
            double r = std::sqrt(x * x + y * y);
            if (r < 0.030) continue; // exclude boss footprint region
            double warp = 0.00025 * std::sin(8.0 * x) * std::cos(10.0 * y);
            plate_pts.push_back({x + nxy(rng), y + nxy(rng), plate_top_z + warp + nz(rng)});
        }
    }

    // Boss top region
    for (int a = 0; a < 720; ++a) {
        double t = 2.0 * M_PI * a / 720.0;
        for (int k = 1; k <= 12; ++k) {
            double rr = boss_r_measured * k / 12.0;
            boss_top_pts.push_back({
                boss_center_x + rr * std::cos(t) + nxy(rng),
                boss_center_y + rr * std::sin(t) + nxy(rng),
                boss_top_measured_z + nz(rng)
            });
        }
    }

    // Boss side ring points
    for (int a = 0; a < 1200; ++a) {
        double t = 2.0 * M_PI * a / 1200.0;
        for (int h = 1; h <= 24; ++h) {
            double z = plate_top_z + (boss_top_measured_z - plate_top_z) * h / 24.0;
            boss_side_pts.push_back({
                boss_center_x + boss_r_measured * std::cos(t) + nxy(rng),
                boss_center_y + boss_r_measured * std::sin(t) + nxy(rng),
                z + nz(rng)
            });
        }
    }

    // Flatness: span of top-surface heights around robust centre
    std::vector<double> plate_z;
    plate_z.reserve(plate_pts.size());
    for (const auto& p : plate_pts) plate_z.push_back(p.z);
    double z_med = median(plate_z);
    double z_min = *std::min_element(plate_z.begin(), plate_z.end());
    double z_max = *std::max_element(plate_z.begin(), plate_z.end());
    double flatness = z_max - z_min;

    // Height: boss top median minus plate top median
    std::vector<double> boss_top_z;
    boss_top_z.reserve(boss_top_pts.size());
    for (const auto& p : boss_top_pts) boss_top_z.push_back(p.z);
    double measured_height = median(boss_top_z) - z_med;

    // Diameter + position from boss side points
    double cx = 0.0, cy = 0.0;
    for (const auto& p : boss_side_pts) { cx += p.x; cy += p.y; }
    cx /= static_cast<double>(boss_side_pts.size());
    cy /= static_cast<double>(boss_side_pts.size());

    std::vector<double> radii;
    radii.reserve(boss_side_pts.size());
    for (const auto& p : boss_side_pts)
        radii.push_back(std::hypot(p.x - cx, p.y - cy));
    double measured_diameter = 2.0 * median(radii);
    double position_error = std::hypot(cx, cy);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "09 - Region metrology primitives\n";
    std::cout << "  Flatness (plate top)      : " << flatness * 1000.0 << " mm\n";
    std::cout << "  Height   (boss-top delta) : " << measured_height * 1000.0 << " mm"
              << "  [nominal " << boss_h * 1000.0 << " mm]\n";
    std::cout << "  Diameter (boss side fit)  : " << measured_diameter * 1000.0 << " mm"
              << "  [nominal " << (2.0 * boss_r_nominal) * 1000.0 << " mm]\n";
    std::cout << "  Position (centre offset)  : " << position_error * 1000.0 << " mm\n";

    return 0;
}

// =============================================================================
// 10 - Uncertainty & Outlier Diagnostics
// =============================================================================
//
// PURPOSE:
//   Compute residual-distribution diagnostics in isolation:
//     - mean / std / median / MAD
//     - inlier / outlier ratios
//     - confidence score + warning flags
// =============================================================================

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    double m = v[n / 2];
    if (n % 2 == 0) {
        auto max_it = std::max_element(v.begin(), v.begin() + n / 2);
        m = 0.5 * (m + *max_it);
    }
    return m;
}

int main() {
    constexpr double tolerance = 0.001; // 1.0 mm

    std::mt19937 rng(7);
    std::normal_distribution<double> core(0.00005, 0.00028);
    std::normal_distribution<double> outlier(0.0028, 0.0008);

    std::vector<double> residuals;
    residuals.reserve(8000);
    for (int i = 0; i < 7600; ++i) residuals.push_back(core(rng));
    for (int i = 0; i < 400; ++i) residuals.push_back(outlier(rng));

    const double mean = std::accumulate(residuals.begin(), residuals.end(), 0.0)
                        / static_cast<double>(residuals.size());

    double sq = 0.0;
    for (double r : residuals) sq += (r - mean) * (r - mean);
    const double stddev = std::sqrt(sq / static_cast<double>(residuals.size()));

    const double med = median(residuals);
    std::vector<double> abs_dev;
    abs_dev.reserve(residuals.size());
    for (double r : residuals) abs_dev.push_back(std::abs(r - med));
    const double mad = median(abs_dev);
    const double robust_sigma = 1.4826 * mad;

    int in_tol = 0;
    int outlier_count = 0;
    const double outlier_gate = std::max(3.0 * robust_sigma, tolerance);
    for (double r : residuals) {
        if (std::abs(r) <= tolerance) ++in_tol;
        if (std::abs(r - med) > outlier_gate) ++outlier_count;
    }

    const double inlier_ratio = static_cast<double>(in_tol) / residuals.size();
    const double outlier_ratio = static_cast<double>(outlier_count) / residuals.size();

    // Confidence score in [0,100], blended from inlier ratio + bias + spread.
    const double bias_term = std::exp(-std::abs(mean) / std::max(1e-12, 0.35 * tolerance));
    const double spread_term = std::exp(-stddev / std::max(1e-12, 0.70 * tolerance));
    const double confidence = 100.0 * std::clamp(
        0.55 * inlier_ratio + 0.25 * bias_term + 0.20 * spread_term,
        0.0, 1.0);

    std::vector<std::string> warnings;
    if (outlier_ratio > 0.08) warnings.emplace_back("HIGH_OUTLIER_RATIO");
    if (std::abs(mean) > 0.35 * tolerance) warnings.emplace_back("SYSTEMATIC_BIAS");
    if (stddev > 0.80 * tolerance) warnings.emplace_back("HIGH_SPREAD");
    if (confidence < 75.0) warnings.emplace_back("LOW_CONFIDENCE");

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "10 - Uncertainty & outlier diagnostics\n";
    std::cout << "  Samples               : " << residuals.size() << "\n";
    std::cout << "  Mean residual         : " << mean * 1000.0 << " mm\n";
    std::cout << "  Std. deviation        : " << stddev * 1000.0 << " mm\n";
    std::cout << "  Median residual       : " << med * 1000.0 << " mm\n";
    std::cout << "  MAD                   : " << mad * 1000.0 << " mm\n";
    std::cout << "  Inlier ratio          : " << (100.0 * inlier_ratio) << " %\n";
    std::cout << "  Outlier ratio         : " << (100.0 * outlier_ratio) << " %\n";
    std::cout << "  Confidence score      : " << confidence << " / 100\n";
    std::cout << "  Warning flags         : ";
    if (warnings.empty()) std::cout << "none";
    else {
        for (std::size_t i = 0; i < warnings.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << warnings[i];
        }
    }
    std::cout << "\n";

    return 0;
}

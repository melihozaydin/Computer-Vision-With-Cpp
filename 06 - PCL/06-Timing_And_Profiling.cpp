// =============================================================================
// 06 - Timing & Profiling Basics (Deterministic Demo)
// =============================================================================
//
// PURPOSE:
//   Demonstrate stage-level timing instrumentation in isolation before we
//   integrate it into ADIF. The workload is synthetic and deterministic so
//   repeated runs are comparable.
//
// RUN:
//   .build/06-Timing_And_Profiling
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

struct StageTiming {
    std::string name;
    double ms = 0.0;
};

double heavyDeterministicWork(std::size_t n, int rounds) {
    std::vector<double> data(n);
    for (std::size_t i = 0; i < n; ++i)
        data[i] = 0.001 * static_cast<double>(i % 1000);

    double checksum = 0.0;
    for (int r = 0; r < rounds; ++r) {
        for (auto& x : data)
            x = std::sin(x + 0.01 * r) + std::cos(0.5 * x);
        checksum += std::accumulate(data.begin(), data.end(), 0.0);
    }
    return checksum;
}

template <typename Fn>
double timeStage(const std::string& name, Fn&& fn, std::vector<StageTiming>& out) {
    auto t0 = Clock::now();
    fn();
    auto t1 = Clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.push_back({name, ms});
    return ms;
}

int main() {
    std::cout << "06 - Timing & Profiling Basics\n\n";

    std::vector<StageTiming> timings;
    double checksum = 0.0;

    timeStage("load", [&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }, timings);

    timeStage("preprocess", [&] {
        checksum += heavyDeterministicWork(250000, 2);
    }, timings);

    timeStage("feature_extract", [&] {
        checksum += heavyDeterministicWork(300000, 2);
    }, timings);

    timeStage("registration", [&] {
        checksum += heavyDeterministicWork(350000, 2);
    }, timings);

    const double total_ms = std::accumulate(
        timings.begin(), timings.end(), 0.0,
        [](double s, const StageTiming& st) { return s + st.ms; });

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Stage timings (ms):\n";
    for (const auto& st : timings) {
        const double pct = (total_ms > 0.0) ? (100.0 * st.ms / total_ms) : 0.0;
        std::cout << "  - " << std::setw(16) << std::left << st.name
                  << " : " << std::setw(9) << std::right << st.ms
                  << "  (" << std::setw(6) << pct << "%)\n";
    }
    std::cout << "\nTotal latency: " << total_ms << " ms\n";
    std::cout << "Throughput   : " << (total_ms > 0.0 ? 1000.0 / total_ms : 0.0)
              << " frames/s\n";
    std::cout << "Checksum     : " << checksum << " (ignore: anti-optimisation)\n";

    return 0;
}

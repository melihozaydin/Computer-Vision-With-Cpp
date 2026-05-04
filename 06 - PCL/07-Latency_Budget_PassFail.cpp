// =============================================================================
// 07 - Latency Budget & PASS/FAIL Decision
// =============================================================================
//
// PURPOSE:
//   Show how to turn per-stage timings into a deterministic latency decision.
//   This is the exact logic we later compose into ADIF final reporting.
// =============================================================================

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

struct StageTiming {
    std::string name;
    double ms;
};

int main(int argc, char** argv) {
    double target_ms = 120.0;
    if (argc > 1) {
        try {
            target_ms = std::stod(argv[1]);
        } catch (...) {
            std::cerr << "Invalid latency target; using default 120.0 ms\n";
        }
    }

    // Deterministic synthetic stage timings.
    std::vector<StageTiming> stages = {
        {"load", 6.4},
        {"preprocess", 11.1},
        {"normals+fpfh", 48.9},
        {"global_align", 21.5},
        {"icp", 15.2},
        {"metrology+diagnostics", 7.9}
    };

    const double total_ms = std::accumulate(stages.begin(), stages.end(), 0.0,
        [](double s, const StageTiming& x) { return s + x.ms; });
    const bool latency_pass = total_ms <= target_ms;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Latency budget report\n";
    std::cout << "---------------------\n";
    for (const auto& st : stages)
        std::cout << "  " << std::setw(22) << std::left << st.name
                  << " : " << std::setw(8) << std::right << st.ms << " ms\n";

    std::cout << "\nTotal latency      : " << total_ms << " ms\n";
    std::cout << "Target latency     : " << target_ms << " ms\n";
    std::cout << "Latency decision   : " << (latency_pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Throughput         : " << (total_ms > 0.0 ? 1000.0 / total_ms : 0.0)
              << " frames/s\n";

    return 0;
}

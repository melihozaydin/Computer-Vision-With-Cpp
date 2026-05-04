// =============================================================================
// 08 - OMP / Parallel Benchmarks (Deterministic Workload)
// =============================================================================
//
// PURPOSE:
//   Benchmark a heavy per-point computation in serial vs parallel mode.
//   If OpenMP is enabled during compilation, we use it. Otherwise we fall back
//   to std::thread chunking so the lesson still runs independently.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::high_resolution_clock;

std::vector<float> makeDeterministicSignal(std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = 0.0005f * static_cast<float>(i % 2000);
    return v;
}

double serialCompute(const std::vector<float>& in, std::vector<float>& out) {
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < in.size(); ++i) {
        float x = in[i];
        out[i] = std::sin(x) + std::cos(0.5f * x) + std::sqrt(1.0f + x * x);
    }
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double threadParallelCompute(const std::vector<float>& in, std::vector<float>& out, unsigned workers) {
    auto t0 = Clock::now();
    workers = std::max(1u, workers);
    const std::size_t n = in.size();
    const std::size_t chunk = (n + workers - 1) / workers;

    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned w = 0; w < workers; ++w) {
        const std::size_t b = static_cast<std::size_t>(w) * chunk;
        const std::size_t e = std::min(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&, b, e] {
            for (std::size_t i = b; i < e; ++i) {
                float x = in[i];
                out[i] = std::sin(x) + std::cos(0.5f * x) + std::sqrt(1.0f + x * x);
            }
        });
    }
    for (auto& th : pool) th.join();

    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

#ifdef _OPENMP
double ompParallelCompute(const std::vector<float>& in, std::vector<float>& out) {
    auto t0 = Clock::now();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(in.size()); ++i) {
        float x = in[static_cast<std::size_t>(i)];
        out[static_cast<std::size_t>(i)] =
            std::sin(x) + std::cos(0.5f * x) + std::sqrt(1.0f + x * x);
    }
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
#endif

int main() {
    constexpr std::size_t N = 3'000'000;
    auto input = makeDeterministicSignal(N);
    std::vector<float> out_serial(N, 0.0f), out_parallel(N, 0.0f);

    const double t_serial = serialCompute(input, out_serial);

    double t_parallel = 0.0;
    std::string mode;
#ifdef _OPENMP
    t_parallel = ompParallelCompute(input, out_parallel);
    mode = "OpenMP";
#else
    const unsigned workers = std::max(1u, std::thread::hardware_concurrency());
    t_parallel = threadParallelCompute(input, out_parallel, workers);
    mode = "std::thread fallback";
#endif

    double checksum = 0.0;
    for (std::size_t i = 0; i < out_parallel.size(); i += 97)
        checksum += out_parallel[i];

    const double speedup = (t_parallel > 0.0) ? t_serial / t_parallel : 0.0;
    const double throughput_serial = (t_serial > 0.0) ? (1000.0 * N / t_serial) : 0.0;
    const double throughput_parallel = (t_parallel > 0.0) ? (1000.0 * N / t_parallel) : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "08 - Parallel benchmark\n";
    std::cout << "  points                : " << N << "\n";
    std::cout << "  hardware threads      : " << std::max(1u, std::thread::hardware_concurrency()) << "\n";
    std::cout << "  parallel mode         : " << mode << "\n";
    std::cout << "  serial latency        : " << t_serial << " ms\n";
    std::cout << "  parallel latency      : " << t_parallel << " ms\n";
    std::cout << "  speedup               : " << speedup << "x\n";
    std::cout << "  serial throughput     : " << throughput_serial / 1e6 << " Mpts/s\n";
    std::cout << "  parallel throughput   : " << throughput_parallel / 1e6 << " Mpts/s\n";
    std::cout << "  checksum              : " << checksum << "\n";

    return 0;
}

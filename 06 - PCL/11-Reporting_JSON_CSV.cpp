// =============================================================================
// 11 - Reporting to JSON & CSV
// =============================================================================
//
// PURPOSE:
//   Demonstrate lightweight export of inspection results to JSON and CSV.
//   No external library required; deterministic data keeps outputs stable.
// =============================================================================

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Report {
    std::string run_id;
    bool geometric_pass = false;
    bool latency_pass = false;
    bool overall_pass = false;
    double tolerance_mm = 1.0;
    double in_tolerance_pct = 0.0;
    double rmse_mm = 0.0;
    double max_dev_mm = 0.0;
    double latency_ms = 0.0;
    double latency_target_ms = 120.0;
    double flatness_mm = 0.0;
    double height_mm = 0.0;
    double diameter_mm = 0.0;
    double position_mm = 0.0;
    double confidence = 0.0;
};

int main() {
    Report r;
    r.run_id = "demo-2026-05-04";
    r.tolerance_mm = 1.0;
    r.in_tolerance_pct = 96.8;
    r.rmse_mm = 0.52;
    r.max_dev_mm = 2.31;
    r.latency_ms = 87.4;
    r.latency_target_ms = 120.0;
    r.flatness_mm = 0.37;
    r.height_mm = 7.64;
    r.diameter_mm = 39.42;
    r.position_mm = 0.81;
    r.confidence = 88.6;
    r.geometric_pass = (r.in_tolerance_pct >= 95.0);
    r.latency_pass = (r.latency_ms <= r.latency_target_ms);
    r.overall_pass = (r.geometric_pass && r.latency_pass);

    std::filesystem::create_directories("data");
    const std::string json_path = "data/adif_report_demo.json";
    const std::string csv_path = "data/adif_report_demo.csv";

    {
        std::ofstream js(json_path, std::ios::binary);
        js << std::fixed << std::setprecision(3);
        js << "{\n";
        js << "  \"run_id\": \"" << r.run_id << "\",\n";
        js << "  \"decision\": {\n";
        js << "    \"geometric_pass\": " << (r.geometric_pass ? "true" : "false") << ",\n";
        js << "    \"latency_pass\": " << (r.latency_pass ? "true" : "false") << ",\n";
        js << "    \"overall_pass\": " << (r.overall_pass ? "true" : "false") << "\n";
        js << "  },\n";
        js << "  \"geometry\": {\n";
        js << "    \"tolerance_mm\": " << r.tolerance_mm << ",\n";
        js << "    \"in_tolerance_pct\": " << r.in_tolerance_pct << ",\n";
        js << "    \"rmse_mm\": " << r.rmse_mm << ",\n";
        js << "    \"max_dev_mm\": " << r.max_dev_mm << "\n";
        js << "  },\n";
        js << "  \"latency\": {\n";
        js << "    \"latency_ms\": " << r.latency_ms << ",\n";
        js << "    \"target_ms\": " << r.latency_target_ms << "\n";
        js << "  },\n";
        js << "  \"metrology\": {\n";
        js << "    \"flatness_mm\": " << r.flatness_mm << ",\n";
        js << "    \"height_mm\": " << r.height_mm << ",\n";
        js << "    \"diameter_mm\": " << r.diameter_mm << ",\n";
        js << "    \"position_mm\": " << r.position_mm << "\n";
        js << "  },\n";
        js << "  \"confidence\": {\n";
        js << "    \"score\": " << r.confidence << "\n";
        js << "  }\n";
        js << "}\n";
    }

    {
        std::ofstream csv(csv_path, std::ios::binary);
        csv << "run_id,geometric_pass,latency_pass,overall_pass,tolerance_mm,in_tolerance_pct,rmse_mm,max_dev_mm,latency_ms,latency_target_ms,flatness_mm,height_mm,diameter_mm,position_mm,confidence\n";
        csv << r.run_id << ','
            << (r.geometric_pass ? 1 : 0) << ','
            << (r.latency_pass ? 1 : 0) << ','
            << (r.overall_pass ? 1 : 0) << ','
            << r.tolerance_mm << ','
            << r.in_tolerance_pct << ','
            << r.rmse_mm << ','
            << r.max_dev_mm << ','
            << r.latency_ms << ','
            << r.latency_target_ms << ','
            << r.flatness_mm << ','
            << r.height_mm << ','
            << r.diameter_mm << ','
            << r.position_mm << ','
            << r.confidence << '\n';
    }

    std::cout << "Wrote JSON: " << json_path << "\n";
    std::cout << "Wrote CSV : " << csv_path << "\n";
    std::cout << "Tip: open both files and compare schema readability vs row compactness.\n";

    return 0;
}

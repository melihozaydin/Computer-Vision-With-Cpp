// =============================================================================
// ADIF / generate_data.cpp — Synthetic Inspection Data Generator
// =============================================================================
//
// PURPOSE:
//   Produce two PCD files that the main ADIF inspection pipeline consumes:
//
//   data/reference_part.pcd  — the "golden master" flat plate with a boss.
//   data/scanned_part.pcd    — the "manufactured" part: same geometry but
//                              with sensor noise, a pressed-in dent, and a
//                              small rigid pose offset (fixture placement
//                              error), simulating a real scanner result.
//
// PART GEOMETRY — flat rectangular plate with a centred cylindrical boss:
//
//        ┌─────────────────────┐   ← plate top face z = 0.005
//        │     ┌──────┐        │
//        │     │ boss │  h=8mm │
//        │     └──────┘        │   boss sits on plate top face
//        └─────────────────────┘   ← plate bottom z = 0.000
//          150 mm × 100 mm plate
//
//   In inspection terms:
//   - The flat faces check flatness (GD&T: ◻).
//   - The boss top checks height (GD&T: ⊥ and true position).
//   - The dent is a negative deviation on the flat face.
//
// =============================================================================

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <Eigen/Geometry>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

// ── Part dimensions (metres) ──────────────────────────────────────────────────
constexpr float PLATE_W       = 0.150f;   // 150 mm
constexpr float PLATE_D       = 0.100f;   // 100 mm
constexpr float PLATE_H       = 0.005f;   // 5 mm thick
constexpr float BOSS_R        = 0.020f;   // 20 mm boss radius
constexpr float BOSS_H        = 0.008f;   // 8 mm boss height
constexpr float NOISE_SIGMA   = 0.0004f;  // 0.4 mm sensor noise
constexpr float DENT_DEPTH    = 0.0025f;  // 2.5 mm dent
constexpr float DENT_RADIUS   = 0.018f;   // 18 mm dent patch
// Dent centre on the top face (slightly off-centre)
constexpr float DENT_CX       = -0.030f;
constexpr float DENT_CY       =  0.020f;

// A raised bump simulates weld spatter or a material build-up (oversized).
// Having BOTH a dent (undersized, −Z) and a bump (oversized, +Z) ensures all
// three colour bands appear in the deviation map: blue, green, and red.
// Placed on the opposite side of the plate from the dent so they don't overlap.
constexpr float BUMP_DEPTH    =  0.002f;  // 2.0 mm proud above nominal surface
constexpr float BUMP_RADIUS   =  0.012f;  // 12 mm affected patch
constexpr float BUMP_CX       =  0.040f;
constexpr float BUMP_CY       = -0.025f;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Uniformly distribute N points on a rectangle [x0,x1]×[y0,y1] at z = z0
void sampleRect(pcl::PointCloud<pcl::PointXYZ>& cloud,
                float x0, float x1, float y0, float y1, float z,
                int nx, int ny)
{
    for (int xi = 0; xi < nx; ++xi)
        for (int yi = 0; yi < ny; ++yi)
            cloud.push_back({x0 + (x1 - x0) * xi / (nx - 1),
                             y0 + (y1 - y0) * yi / (ny - 1),
                             z});
}

// Sample a cylinder lateral surface  r = radius,  z ∈ [z0, z0+height]
void sampleCylSurface(pcl::PointCloud<pcl::PointXYZ>& cloud,
                      float cx, float cy, float radius,
                      float z0, float height,
                      int theta_n, int z_n)
{
    for (int ti = 0; ti < theta_n; ++ti) {
        float theta = 2.0f * static_cast<float>(M_PI) * ti / theta_n;
        for (int zi = 0; zi <= z_n; ++zi)
            cloud.push_back({cx + radius * std::cos(theta),
                             cy + radius * std::sin(theta),
                             z0 + height * zi / z_n});
    }
}

// Sample a disk (filled) at z = z0, radius = r, centred at (cx, cy)
void sampleDisk(pcl::PointCloud<pcl::PointXYZ>& cloud,
                float cx, float cy, float r, float z, int rings)
{
    for (int ri = 1; ri <= rings; ++ri) {
        float rad = r * ri / rings;
        int   n   = std::max(6, static_cast<int>(
                       2.0f * static_cast<float>(M_PI) * rad / (r / rings)));
        for (int ti = 0; ti < n; ++ti) {
            float theta = 2.0f * static_cast<float>(M_PI) * ti / n;
            cloud.push_back({cx + rad * std::cos(theta),
                             cy + rad * std::sin(theta), z});
        }
    }
}

// ── Build clean reference ─────────────────────────────────────────────────────
pcl::PointCloud<pcl::PointXYZ>::Ptr buildReference() {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                     new pcl::PointCloud<pcl::PointXYZ>());

    float hx = PLATE_W / 2;
    float hy = PLATE_D / 2;

    // Top face of plate (z = PLATE_H)
    sampleRect(*cloud, -hx, hx, -hy, hy, PLATE_H, 76, 51);

    // Bottom face of plate (z = 0)
    sampleRect(*cloud, -hx, hx, -hy, hy, 0.0f, 76, 51);

    // Four side faces
    sampleRect(*cloud,  hx, hx, -hy, hy,        0.0f, 2,  26);  // +X face (degenerate, just edges)
    for (int zi = 0; zi <= 10; ++zi) {
        float z = PLATE_H * zi / 10.0f;
        for (int yi = -25; yi <= 25; ++yi) {
            float y = PLATE_D * yi / 50.0f;
            cloud->push_back({ hx, y, z});
            cloud->push_back({-hx, y, z});
        }
        for (int xi = -37; xi <= 37; ++xi) {
            float x = PLATE_W * xi / 75.0f;
            cloud->push_back({x,  hy, z});
            cloud->push_back({x, -hy, z});
        }
    }

    // Boss: cylinder on top of plate
    sampleCylSurface(*cloud, 0.0f, 0.0f, BOSS_R, PLATE_H, BOSS_H, 100, 40);

    // Boss top disk
    sampleDisk(*cloud, 0.0f, 0.0f, BOSS_R, PLATE_H + BOSS_H, 10);

    cloud->width  = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

// ── Add defects + noise + misalignment ───────────────────────────────────────
pcl::PointCloud<pcl::PointXYZ>::Ptr buildScanned(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& ref)
{
    std::mt19937 rng(123);
    std::normal_distribution<float> gauss(0.0f, NOISE_SIGMA);

    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                     new pcl::PointCloud<pcl::PointXYZ>(*ref));

    Eigen::Vector3f dent_centre(DENT_CX, DENT_CY, PLATE_H);

    for (auto& p : cloud->points) {
        // Gaussian noise
        p.x += gauss(rng);
        p.y += gauss(rng);
        p.z += gauss(rng);

        // A 2 mm guard (>> sensor noise σ = 0.4 mm) isolates the top face
        // without clipping legitimate surface geometry near the plate edge.
        if (std::abs(p.z - PLATE_H) < 0.002f) {
            Eigen::Vector3f pt(p.x, p.y, p.z);

            // Dent — push inward (−Z).  Cosine taper gives a smooth C¹
            // boundary so there are no unrealistic sharp edges in the cloud.
            float dist_dent = (pt - dent_centre).norm();
            if (dist_dent < DENT_RADIUS) {
                float alpha = 0.5f * (1.0f + std::cos(
                                  static_cast<float>(M_PI) * dist_dent / DENT_RADIUS));
                p.z -= DENT_DEPTH * alpha;
            }

            // Bump — push outward (+Z, oversized → red in the deviation map).
            Eigen::Vector3f bump_centre(BUMP_CX, BUMP_CY, PLATE_H);
            float dist_bump = (pt - bump_centre).norm();
            if (dist_bump < BUMP_RADIUS) {
                float alpha = 0.5f * (1.0f + std::cos(
                                  static_cast<float>(M_PI) * dist_bump / BUMP_RADIUS));
                p.z += BUMP_DEPTH * alpha;
            }
        }
    }

    // Rigid misalignment — simulates a part placed on the fixture with small
    // angular and positional errors.
    //
    // Eigen::Affine3f accumulation rules (important for getting the order right):
    //   .rotate(R)       → T = T * R        (post-multiply)
    //   .pretranslate(t) → T = Trans(t) * T (pre-multiply)
    //
    // After two rotates:     T = R_yaw * R_pitch
    // After pretranslate(t): T = Trans(t) * R_yaw * R_pitch
    // Applied to point p:    p' = R_yaw * R_pitch * p  +  t
    //
    // Interpretation: the part is first rotated (fixture angular error),
    // then offset by t in the world frame (fixture positional error).
    // This is the natural model for a mis-seated part on a fixture.
    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.rotate(Eigen::AngleAxisf(0.087f,  Eigen::Vector3f::UnitZ()));  // 5° yaw
    T.rotate(Eigen::AngleAxisf(0.017f,  Eigen::Vector3f::UnitY()));  // 1° pitch
    T.pretranslate(Eigen::Vector3f(0.002f, -0.001f, 0.0005f));       // world-frame offset
    pcl::transformPointCloud(*cloud, *cloud, T);

    return cloud;
}

int main() {
    std::filesystem::create_directories("data");

    auto ref = buildReference();
    std::cout << "Reference part : " << ref->size() << " points\n";
    // Binary PCD is ~10× faster to write/read than ASCII and preserves full
    // float precision (ASCII truncates to ~7 significant digits, which
    // introduces rounding errors at sub-millimetre scales).
    pcl::io::savePCDFileBinary("data/reference_part.pcd", *ref);
    std::cout << "Saved → data/reference_part.pcd\n";

    auto scan = buildScanned(ref);
    std::cout << "Scanned part   : " << scan->size() << " points\n";
    pcl::io::savePCDFileBinary("data/scanned_part.pcd", *scan);
    std::cout << "Saved → data/scanned_part.pcd\n\n";

    std::cout << "Part parameters:\n"
              << std::fixed << std::setprecision(1)
              << "  Plate          : " << PLATE_W*1000 << " × " << PLATE_D*1000
              << " × " << PLATE_H*1000 << " mm\n"
              << "  Boss           : R=" << BOSS_R*1000 << " mm  H=" << BOSS_H*1000 << " mm\n"
              << "  Sensor noise σ : " << NOISE_SIGMA*1000 << " mm\n"
              << "  Dent depth     : " << DENT_DEPTH*1000  << " mm\n"
              << "  Dent radius    : " << DENT_RADIUS*1000 << " mm\n"
              << "  Bump height    : " << BUMP_DEPTH*1000  << " mm\n"
              << "  Bump radius    : " << BUMP_RADIUS*1000 << " mm\n"
              << "  Misalignment   : 5° yaw, 1° pitch, 2 mm translation\n\n"
              << "Run 'adif data/reference_part.pcd data/scanned_part.pcd' next.\n";

    return 0;
}

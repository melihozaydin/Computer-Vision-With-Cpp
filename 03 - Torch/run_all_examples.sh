#!/bin/bash

# LibTorch Examples Runner (03 - Torch)
# ============================================================
# Builds and runs all LibTorch C++ examples.
#
# Modes:
#   --local  (default)  Use ~/libtorch on the host WSL system.
#   --docker            Build + run inside pytorch/pytorch Docker image.
#                       The container already ships LibTorch; no local
#                       installation needed.  OpenCV is installed via apt
#                       inside the container on first run.
#
# Usage:
#   ./run_all_examples.sh                      # local build + run
#   ./run_all_examples.sh --docker             # Docker build + run (CUDA if GPU available)
#   ./run_all_examples.sh --docker --image pytorch/pytorch:latest
#   ./run_all_examples.sh --build-only         # local build only
#   ./run_all_examples.sh --run-only           # local run only
#   ./run_all_examples.sh --timeout N          # per-example timeout (default: 30)
#   ./run_all_examples.sh --libtorch DIR       # override ~/libtorch path (local mode)
#   ./run_all_examples.sh --clean              # remove .build/ and exit
#   ./run_all_examples.sh --help               # show this help
#
# Prerequisites:
#   Local mode:   ~/libtorch extracted from pytorch.org (see 03 - Torch/README.md)
#                 sudo apt install libopencv-dev
#   Docker mode:  Docker with optional NVIDIA runtime (nvidia-smi in WSL)
#                 Installs ABI-compatible libtorch-dev + libopencv-dev via apt

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
DOCKER_MODE=false
IMAGE="pytorch/pytorch:latest"
BUILD_ONLY=false
RUN_ONLY=false
CLEAN=false
TIMEOUT=30
LIBTORCH_DIR="${HOME}/libtorch"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BLUE='\033[0;34m'; NC='\033[0m'

print_info()    { echo -e "${CYAN}  $1${NC}"; }
print_ok()      { echo -e "${GREEN}  ✓ $1${NC}"; }
print_warn()    { echo -e "${YELLOW}  ⚠ $1${NC}"; }
print_fail()    { echo -e "${RED}  ✗ $1${NC}"; }
print_header()  { echo -e "\n${BLUE}══ $1 ══${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker)      DOCKER_MODE=true;         shift ;;
        --local)       DOCKER_MODE=false;        shift ;;
        --image)       IMAGE="$2";               shift 2 ;;
        --build-only)  BUILD_ONLY=true;          shift ;;
        --run-only)    RUN_ONLY=true;            shift ;;
        --clean)       CLEAN=true;               shift ;;
        --timeout)     TIMEOUT="$2";             shift 2 ;;
        --libtorch)    LIBTORCH_DIR="$2";        shift 2 ;;
        --help|-h)
                        cat <<'EOF'
LibTorch Examples Runner (03 - Torch)
Usage:
    ./run_all_examples.sh                      # local build + run
    ./run_all_examples.sh --docker             # Docker build + run
    ./run_all_examples.sh --docker --image pytorch/pytorch:latest
    ./run_all_examples.sh --build-only         # local build only
    ./run_all_examples.sh --run-only           # local run only
    ./run_all_examples.sh --timeout N          # per-example timeout (default: 30)
    ./run_all_examples.sh --libtorch DIR       # override ~/libtorch path (local mode)
    ./run_all_examples.sh --clean              # remove .build/ and exit
    ./run_all_examples.sh --help               # show this help
EOF
            exit 0 ;;
        *)
            echo "Unknown option: $1  (use --help)"
            exit 1 ;;
    esac
done

if [[ "$CLEAN" == "true" ]]; then
    rm -rf .build
    echo "Cleaned .build/"
    exit 0
fi

# ── Docker mode ───────────────────────────────────────────────────────────────
if [[ "$DOCKER_MODE" == "true" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

    # Detect GPU availability for --gpus flag
    GPU_FLAG=""
    if docker info 2>/dev/null | grep -q "nvidia"; then
        GPU_FLAG="--gpus all"
        print_info "NVIDIA runtime detected — enabling GPU in container"
    fi

    PASS_ARGS="--local --timeout $TIMEOUT"
    [[ "$BUILD_ONLY" == "true" ]] && PASS_ARGS="$PASS_ARGS --build-only"
    [[ "$RUN_ONLY"   == "true" ]] && PASS_ARGS="$PASS_ARGS --run-only"

    print_info "Image: $IMAGE"
    print_info "Note: First run installs libopencv-dev inside the container (~1 min)."
    print_info "      Subsequent runs on the same container layer are instant."

    # In Docker mode, install ABI-compatible system libtorch + OpenCV from apt.
    # This avoids ABI mismatch between Python-packaged torch and Ubuntu OpenCV.
    INNER_CMD='
set -e
if [[ -f /usr/include/opencv4/opencv2/core.hpp ]] && [[ -f /usr/include/torch/csrc/api/include/torch/torch.h ]] && command -v make >/dev/null 2>&1; then
    echo "==> OpenCV + LibTorch headers and build tools already present, skipping apt install"
else
    echo "==> Installing required packages (build tools + OpenCV dev + LibTorch dev)."
    echo "    This can take several minutes on first run..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends build-essential pkg-config libopencv-dev libtorch-dev
    echo "==> Package install completed"
fi

cd "/workspace/03 - Torch"

TORCH_USE_SYSTEM=1 TORCH_ABI_OVERRIDE=1 ./run_all_examples.sh --local --timeout '"$TIMEOUT"' '"$([ "$BUILD_ONLY" = true ] && echo '--build-only' || echo '')"' '"$([ "$RUN_ONLY" = true ] && echo '--run-only' || echo '')"' --libtorch /usr
'

    # shellcheck disable=SC2086
    docker run --rm $GPU_FLAG \
        -v "$REPO_ROOT:/workspace" \
        "$IMAGE" \
        bash -lc "$INNER_CMD"
    exit $?
fi

# ── Local mode ────────────────────────────────────────────────────────────────
if [[ "$RUN_ONLY" == "false" ]]; then
    ABI_DEFINE=""
    if [[ -n "${TORCH_ABI_OVERRIDE:-}" ]]; then
        ABI_DEFINE="-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI_OVERRIDE}"
        print_info "Using Torch ABI override: ${TORCH_ABI_OVERRIDE}"
    fi

    TORCH_CUDA_LIB=""
    if [[ "${TORCH_USE_SYSTEM:-0}" == "1" ]]; then
        if [[ ! -f /usr/include/torch/csrc/api/include/torch/torch.h ]]; then
            print_fail "System LibTorch headers not found (/usr/include/torch/csrc/api/include/torch/torch.h)."
            echo "Install with: sudo apt install libtorch-dev"
            exit 1
        fi

        if ldconfig -p 2>/dev/null | grep -q 'libtorch_cuda.so'; then
            TORCH_CUDA_LIB="-ltorch_cuda"
        else
            print_warn "libtorch_cuda not found in system LibTorch; building CPU-only Torch targets"
        fi

        print_header "Building all examples  (libtorch: system /usr)"
        CXXFLAGS_OVERRIDE="$ABI_DEFINE -std=c++17 -O2 \
            -I/usr/include \
            -I/usr/include/torch/csrc/api/include \
            -I/usr/include/opencv4"
        LDFLAGS_OVERRIDE="-L/usr/lib/x86_64-linux-gnu \
            -Wl,--no-as-needed \
            -ltorch -ltorch_cpu $TORCH_CUDA_LIB -lc10 \
            -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui \
            -Wl,-rpath,/usr/lib/x86_64-linux-gnu"
    else
        if [[ ! -d "$LIBTORCH_DIR" ]]; then
            print_fail "LibTorch not found at: $LIBTORCH_DIR"
            echo "Download from https://pytorch.org/cppdocs/installing.html"
            echo "Or use --docker to build without a local LibTorch install."
            exit 1
        fi

        if [[ -f "$LIBTORCH_DIR/lib/libtorch_cuda.so" ]]; then
            TORCH_CUDA_LIB="-ltorch_cuda"
        else
            print_warn "libtorch_cuda not found under $LIBTORCH_DIR/lib; building CPU-only Torch targets"
        fi

        print_header "Building all examples  (libtorch: $LIBTORCH_DIR)"
        CXXFLAGS_OVERRIDE="$ABI_DEFINE -std=c++17 -O2 \
            -I${LIBTORCH_DIR}/include \
            -I${LIBTORCH_DIR}/include/torch/csrc/api/include \
            -I/usr/include/opencv4"
        LDFLAGS_OVERRIDE="-L${LIBTORCH_DIR}/lib \
            -Wl,--no-as-needed \
            -ltorch -ltorch_cpu $TORCH_CUDA_LIB -lc10 \
            -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui \
            -Wl,-rpath,${LIBTORCH_DIR}/lib \
            -L/usr/lib/x86_64-linux-gnu"
    fi

    if ! make -f Makefile \
        CXXFLAGS="$CXXFLAGS_OVERRIDE" \
        LDFLAGS="$LDFLAGS_OVERRIDE" \
        all 2>&1; then
        print_fail "Build failed"
        exit 1
    fi
    print_ok "Build complete → .build/"
fi

[[ "$BUILD_ONLY" == "true" ]] && exit 0

# ── Run ───────────────────────────────────────────────────────────────────────
print_header "Running all examples  (timeout: ${TIMEOUT}s each)"

ok=0; timeout_count=0; fail=0
total=0
skipped=0

# Build run list from current source files to avoid stale binaries in .build/
mapfile -t targets < <(find . -maxdepth 1 -type f -name '*.cpp' -printf '%f\n' | sed 's/\.cpp$//' | sort)

for name in "${targets[@]}"; do
    # This file is excluded by Makefile unless user manually enables TorchVision C++
    [[ "$name" == "20-Torch_Use_TorchVision" ]] && continue

    binary=".build/$name"
    if [[ ! -x "$binary" ]]; then
        print_warn "$name  (binary not found: $binary)"
        skipped=$((skipped + 1))
        continue
    fi

    # Skip MNIST dataset-dependent example when dataset is unavailable
    if [[ "$name" == "13-Torch_Train_CNN_MNIST" && ! -f "./data/MNIST/raw/train-images-idx3-ubyte" ]]; then
        print_warn "$name  (skipped: MNIST dataset not found under ./data/MNIST/raw)"
        skipped=$((skipped + 1))
        continue
    fi

    total=$((total + 1))

    output=$(timeout "${TIMEOUT}s" "$binary" 2>&1)
    code=$?

    if [[ $code -eq 0 ]]; then
        print_ok "$name"
        ok=$((ok + 1))
    elif [[ $code -eq 124 ]]; then
        print_warn "$name  (timeout — normal for training loops)"
        timeout_count=$((timeout_count + 1))
    else
        print_fail "$name  (exit $code)"
        [[ -n "$output" ]] && echo "$output" | head -8 | sed 's/^/    /'
        fail=$((fail + 1))
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
print_header "Summary"
echo -e "  ${GREEN}ok=${ok}${NC}  ${YELLOW}timeout=${timeout_count}${NC}  ${RED}fail=${fail}${NC}  skipped=${skipped}  total=${total}"

[[ $fail -gt 0 ]] && exit 1 || exit 0

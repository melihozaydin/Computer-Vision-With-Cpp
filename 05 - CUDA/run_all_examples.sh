#!/bin/bash

# Pure CUDA examples runner (05 - CUDA)
# ============================================================
# Modes:
#   --docker (default)  Run inside an NVIDIA CUDA devel container.
#   --local             Use host nvcc toolchain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DOCKER_MODE=true
IMAGE="nvidia/cuda:12.3.2-devel-ubuntu22.04"
BUILD_ONLY=false
RUN_ONLY=false
CLEAN=false
TIMEOUT=12
VERBOSE=true

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BLUE='\033[0;34m'; NC='\033[0m'
print_info(){ echo -e "${CYAN}[INFO] $1${NC}"; }
print_ok(){ echo -e "${GREEN}$1${NC}"; }
print_warn(){ echo -e "${YELLOW}$1${NC}"; }
print_fail(){ echo -e "${RED}$1${NC}"; }
print_header(){ echo -e "\n${BLUE}== $1 ==${NC}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker) DOCKER_MODE=true; shift ;;
    --local) DOCKER_MODE=false; shift ;;
    --image) IMAGE="$2"; shift 2 ;;
    --build-only) BUILD_ONLY=true; shift ;;
    --run-only) RUN_ONLY=true; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --no-timeout) TIMEOUT=0; shift ;;
    --quiet) VERBOSE=false; shift ;;
    --clean) CLEAN=true; shift ;;
    --help|-h)
      cat <<'EOF'
Pure CUDA examples runner (05 - CUDA)
Usage:
  ./run_all_examples.sh               # Docker build + run (default)
  ./run_all_examples.sh --docker      # same as default
  ./run_all_examples.sh --local       # host nvcc toolchain
  ./run_all_examples.sh --build-only  # compile only
  ./run_all_examples.sh --run-only    # run existing binaries only
  ./run_all_examples.sh --timeout N   # per-example timeout (default: 12)
  ./run_all_examples.sh --no-timeout  # disable per-example timeout
  ./run_all_examples.sh --quiet       # reduce log verbosity
  ./run_all_examples.sh --clean       # remove .build/ and exit
EOF
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$CLEAN" == "true" ]]; then
  rm -rf .build
  echo "Cleaned .build/"
  exit 0
fi

if [[ "$DOCKER_MODE" == "true" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  GPU_FLAG=""
  if docker info 2>/dev/null | grep -q nvidia; then
    GPU_FLAG="--gpus all"
  fi
  PASS_ARGS="--local --timeout $TIMEOUT"
  [[ "$BUILD_ONLY" == "true" ]] && PASS_ARGS="$PASS_ARGS --build-only"
  [[ "$RUN_ONLY" == "true" ]] && PASS_ARGS="$PASS_ARGS --run-only"
  print_info "Image: $IMAGE"
  docker run --rm $GPU_FLAG -v "$REPO_ROOT:/workspace" "$IMAGE" bash -lc "
    set -e
    export DEBIAN_FRONTEND=noninteractive
    if ! command -v make >/dev/null 2>&1; then
      apt-get update
      apt-get install -y --no-install-recommends build-essential
    fi
    cd '/workspace/05 - CUDA'
    ./run_all_examples.sh $PASS_ARGS
  "
  exit $?
fi

if [[ "$RUN_ONLY" == "false" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    print_fail "nvcc not found. Install CUDA Toolkit or use --docker."
    exit 1
  fi
  if ! command -v make >/dev/null 2>&1; then
    print_fail "make not found. Install build-essential or use --docker."
    exit 1
  fi

  print_header "Building CUDA examples"
  if ! make all; then
    print_fail "Build failed"
    exit 1
  fi
  print_ok "Build complete → .build/"
fi

[[ "$BUILD_ONLY" == "true" ]] && exit 0

print_header "Running CUDA examples (timeout: ${TIMEOUT}s each)"
ok=0; timeout_count=0; fail=0; total=0
mapfile -t targets < <(find . -maxdepth 1 -type f -name '*.cu' -printf '%f\n' | sed 's/\.cu$//' | sort)

for name in "${targets[@]}"; do
  binary=".build/$name"
  [[ -x "$binary" ]] || { print_warn "$name  (binary not found)"; continue; }
  total=$((total + 1))
  [[ "$VERBOSE" == "true" ]] && print_info "Running $name"
  tmp_out=$(mktemp)
  if [[ "$TIMEOUT" -eq 0 ]]; then
    if "$binary" >"$tmp_out" 2>&1; then
      code=0
    else
      code=$?
    fi
  else
    if timeout "${TIMEOUT}s" "$binary" >"$tmp_out" 2>&1; then
      code=0
    else
      code=$?
    fi
  fi
  if [[ "$VERBOSE" == "true" ]] && [[ -s "$tmp_out" ]]; then
    sed 's/^/    /' "$tmp_out" | head -40
  fi
  if [[ $code -eq 0 ]]; then
    print_ok "✓ $name"
    ok=$((ok + 1))
  elif [[ $code -eq 124 ]]; then
    print_warn "⚠ $name  (timeout)"
    timeout_count=$((timeout_count + 1))
  else
    print_fail "✗ $name  (exit $code)"
    [[ -s "$tmp_out" ]] && sed 's/^/    /' "$tmp_out" | head -20
    fail=$((fail + 1))
  fi
  rm -f "$tmp_out"
done

print_header "Summary"
echo -e "  ${GREEN}ok=${ok}${NC}  ${YELLOW}timeout=${timeout_count}${NC}  ${RED}fail=${fail}${NC}  total=${total}"
[[ $fail -gt 0 ]] && exit 1 || exit 0

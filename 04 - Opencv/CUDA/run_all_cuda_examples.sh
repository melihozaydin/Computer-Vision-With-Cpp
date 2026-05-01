#!/usr/bin/env bash
# ==============================================================================
# OpenCV CUDA Examples: Build + Run All
# ==============================================================================
# Fast-start helper for this folder:
#   04 - Opencv/CUDA
#
# What this script does:
#   1) Generates a tiny CMake project for all 00..10 CUDA examples
#   2) Builds everything
#   3) Runs all binaries with a per-example timeout
#   4) Prints a clean summary
#
# Why this script exists:
#   - Host OpenCV setups differ a lot (CPU-only vs CUDA-enabled builds)
#   - Docker path is often the fastest reproducible way to get results
#
# Typical usage:
#   ./run_all_cuda_examples.sh --docker
#   ./run_all_cuda_examples.sh --local
#   ./run_all_cuda_examples.sh --docker --timeout 10
#
# Notes:
#   - --docker mode expects NVIDIA Docker runtime support (--gpus all)
#   - --local mode expects a local CUDA-enabled OpenCV toolchain
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="docker"                       # default: reproducible container path
TIMEOUT_SECONDS=6                    # timeout per example binary
BUILD_TYPE="Release"
DO_BUILD=1
DO_RUN=1
DO_CLEAN=0
IMAGE="thecanadianroot/opencv-cuda:latest"

# In local mode, this can be overridden with --opencv-dir
OPENCV_DIR_DEFAULT="/usr/local/lib/cmake/opencv4"

log()  { printf "[INFO] %s\n" "$*"; }
warn() { printf "[WARN] %s\n" "$*"; }
err()  { printf "[ERR ] %s\n" "$*" >&2; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --docker                 Run inside CUDA container (default)
  --local                  Run on host toolchain (WSL/Linux)
  --timeout <sec>          Timeout per example run (default: ${TIMEOUT_SECONDS})
  --image <name:tag>       Docker image for --docker mode
  --opencv-dir <path>      OpenCVConfig.cmake dir for --local mode
  --build-only             Build examples only, do not run
  --run-only               Run existing binaries only, do not rebuild
  --clean                  Remove generated build folders before work
  -h, --help               Show help

Examples:
  $(basename "$0") --docker
  $(basename "$0") --docker --timeout 10
  $(basename "$0") --local --opencv-dir /usr/local/lib/cmake/opencv4
EOF
}

OPENCV_DIR="$OPENCV_DIR_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker) MODE="docker"; shift ;;
    --local) MODE="local"; shift ;;
    --timeout)
      TIMEOUT_SECONDS="${2:-}"
      [[ -n "$TIMEOUT_SECONDS" ]] || { err "--timeout requires a value"; exit 2; }
      shift 2
      ;;
    --image)
      IMAGE="${2:-}"
      [[ -n "$IMAGE" ]] || { err "--image requires a value"; exit 2; }
      shift 2
      ;;
    --opencv-dir)
      OPENCV_DIR="${2:-}"
      [[ -n "$OPENCV_DIR" ]] || { err "--opencv-dir requires a value"; exit 2; }
      shift 2
      ;;
    --build-only)
      DO_BUILD=1
      DO_RUN=0
      shift
      ;;
    --run-only)
      DO_BUILD=0
      DO_RUN=1
      shift
      ;;
    --clean)
      DO_CLEAN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

# ----------------------------------------------------------------------------
# Build helper: write CMakeLists that includes all [0-9][0-9]-*.cpp examples
# ----------------------------------------------------------------------------
write_cmake_project() {
  local project_root="$1"
  mkdir -p "$project_root"
  cat > "$project_root/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(opencv_cuda_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV REQUIRED)

# NOTE: This script sits in 04 - Opencv/CUDA, so examples are one level up.
file(GLOB EXAMPLES "${CMAKE_CURRENT_LIST_DIR}/../[0-9][0-9]-*.cpp")

foreach(src ${EXAMPLES})
  get_filename_component(exe ${src} NAME_WE)
  add_executable(${exe} ${src})
  target_link_libraries(${exe} PRIVATE ${OpenCV_LIBS})
endforeach()
EOF
}

# ----------------------------------------------------------------------------
# Run helper: execute all binaries in a folder with timeout + summary
# ----------------------------------------------------------------------------
run_binaries_with_summary() {
  local bin_dir="$1"
  local timeout_s="$2"

  local ok=0
  local timeout_count=0
  local fail=0

  shopt -s nullglob
  for f in "$bin_dir"/*; do
    [[ -x "$f" ]] || continue
    local name
    name="$(basename "$f")"

    set +e
    timeout "${timeout_s}s" "$f" >/tmp/opencv_cuda_out.txt 2>/tmp/opencv_cuda_err.txt
    local ec=$?
    set -e

    local st
    if [[ $ec -eq 0 ]]; then
      st="ok"
      ok=$((ok + 1))
    elif [[ $ec -eq 124 ]]; then
      st="timeout"
      timeout_count=$((timeout_count + 1))
    else
      st="fail_${ec}"
      fail=$((fail + 1))
    fi

    printf "%s\t%s\n" "$name" "$st"
  done

  echo "__SUMMARY__ ok=${ok} timeout=${timeout_count} fail=${fail}"

  # Return non-zero only if there were hard failures.
  [[ $fail -eq 0 ]]
}

# ----------------------------------------------------------------------------
# Local mode: host toolchain / WSL toolchain
# ----------------------------------------------------------------------------
run_local_mode() {
  log "Mode: local"
  local cmake_src="$SCRIPT_DIR/.local_build"
  local build_dir="$cmake_src/build"

  if [[ $DO_CLEAN -eq 1 ]]; then
    log "Cleaning local build folder"
    rm -rf "$cmake_src"
  fi

  if [[ $DO_BUILD -eq 1 ]]; then
    log "Preparing CMake project in .local_build"
    write_cmake_project "$cmake_src"

    # Optional helper script from some OpenCV installs.
    if [[ -f /usr/local/bin/setup_vars_opencv4.sh ]]; then
      # Some vendor scripts assume unset vars; temporarily relax nounset.
      set +u
      # shellcheck disable=SC1091
      source /usr/local/bin/setup_vars_opencv4.sh || true
      set -u
    fi

    log "Configuring CMake (OpenCV_DIR=$OPENCV_DIR)"
    cmake -S "$cmake_src" -B "$build_dir" -DOpenCV_DIR="$OPENCV_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

    log "Building examples"
    cmake --build "$build_dir" -j"$(nproc)"
  fi

  if [[ $DO_RUN -eq 1 ]]; then
    log "Running binaries from $build_dir/bin"
    run_binaries_with_summary "$build_dir/bin" "$TIMEOUT_SECONDS"
  fi
}

# ----------------------------------------------------------------------------
# Docker mode: reproducible CUDA + OpenCV toolchain
# ----------------------------------------------------------------------------
run_docker_mode() {
  log "Mode: docker"
  log "Image: $IMAGE"

  # Convert current repo path from WSL style for Docker on Linux/WSL usage.
  # This script is intended for bash environments where Docker CLI is available.
  local mount_repo="$REPO_ROOT"

  # Run all logic INSIDE container to avoid host dependency drift.
  docker run --rm --gpus all -i \
    -v "$mount_repo:/workspace" \
    -w "/workspace/04 - Opencv/CUDA" \
    "$IMAGE" \
    bash -s -- "$TIMEOUT_SECONDS" "$DO_BUILD" "$DO_RUN" "$DO_CLEAN" "$BUILD_TYPE" <<'INNER_SCRIPT'
set -euo pipefail

TIMEOUT_SECONDS="$1"
DO_BUILD="$2"
DO_RUN="$3"
DO_CLEAN="$4"
BUILD_TYPE="$5"

log()  { printf "[INFO] %s\n" "$*"; }

# Optional helper in many OpenCV container images.
if [[ -f /usr/local/bin/setup_vars_opencv4.sh ]]; then
  # Some vendor scripts assume unset vars; temporarily relax nounset.
  set +u
  # shellcheck disable=SC1091
  source /usr/local/bin/setup_vars_opencv4.sh || true
  set -u
fi

write_cmake_project() {
  local project_root="$1"
  mkdir -p "$project_root"
  cat > "$project_root/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(opencv_cuda_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV REQUIRED)
file(GLOB EXAMPLES "${CMAKE_CURRENT_LIST_DIR}/../[0-9][0-9]-*.cpp")

foreach(src ${EXAMPLES})
  get_filename_component(exe ${src} NAME_WE)
  add_executable(${exe} ${src})
  target_link_libraries(${exe} PRIVATE ${OpenCV_LIBS})
endforeach()
EOF
}

run_binaries_with_summary() {
  local bin_dir="$1"
  local timeout_s="$2"

  local ok=0
  local timeout_count=0
  local fail=0

  shopt -s nullglob
  for f in "$bin_dir"/*; do
    [[ -x "$f" ]] || continue
    local name
    name="$(basename "$f")"

    set +e
    timeout "${timeout_s}s" "$f" >/tmp/opencv_cuda_out.txt 2>/tmp/opencv_cuda_err.txt
    local ec=$?
    set -e

    local st
    if [[ $ec -eq 0 ]]; then
      st="ok"
      ok=$((ok + 1))
    elif [[ $ec -eq 124 ]]; then
      st="timeout"
      timeout_count=$((timeout_count + 1))
    else
      st="fail_${ec}"
      fail=$((fail + 1))
    fi

    printf "%s\t%s\n" "$name" "$st"
  done

  echo "__SUMMARY__ ok=${ok} timeout=${timeout_count} fail=${fail}"
  [[ $fail -eq 0 ]]
}

CMAKE_SRC_DIR=".container_build"
BUILD_DIR="${CMAKE_SRC_DIR}/build"

if [[ "$DO_CLEAN" == "1" ]]; then
  log "Cleaning container build folder"
  rm -rf "$CMAKE_SRC_DIR"
fi

if [[ "$DO_BUILD" == "1" ]]; then
  log "Preparing CMake project in $CMAKE_SRC_DIR"
  write_cmake_project "$CMAKE_SRC_DIR"

  log "Configuring CMake"
  cmake -S "$CMAKE_SRC_DIR" -B "$BUILD_DIR" -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

  log "Building examples"
  cmake --build "$BUILD_DIR" -j"$(nproc)"
fi

if [[ "$DO_RUN" == "1" ]]; then
  log "Running binaries from $BUILD_DIR/bin"
  run_binaries_with_summary "$BUILD_DIR/bin" "$TIMEOUT_SECONDS"
fi
INNER_SCRIPT
}

main() {
  log "CUDA examples fast-start script"
  log "Folder: $SCRIPT_DIR"

  case "$MODE" in
    local)  run_local_mode ;;
    docker) run_docker_mode ;;
    *) err "Unsupported mode: $MODE"; exit 2 ;;
  esac

  log "Done."
}

main

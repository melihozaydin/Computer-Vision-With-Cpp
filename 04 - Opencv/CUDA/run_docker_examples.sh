#!/bin/bash

# Run OpenCV CUDA examples using prebuilt Docker runtime image
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNTIME_IMAGE="cv-opencv-cuda-runtime:latest"
PASS_ARGS="--docker"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime-image) RUNTIME_IMAGE="$2"; shift 2 ;;
    --help|-h)
      cat <<'EOF'
Run OpenCV CUDA examples in Docker (requires prebuilt image)
Usage:
  ./run_docker_examples.sh [any run_all_cuda_examples.sh options]
  ./run_docker_examples.sh --runtime-image cv-opencv-cuda-runtime:latest --timeout 10

Tip:
  Build image first with ./build_docker_env.sh
EOF
      exit 0 ;;
    *)
      PASS_ARGS="$PASS_ARGS $1"
      shift ;;
  esac
done

./run_all_cuda_examples.sh --docker-runtime-image "$RUNTIME_IMAGE" $PASS_ARGS

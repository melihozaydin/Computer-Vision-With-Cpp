#!/bin/bash

# Build OpenCV CUDA runtime image once (separate from running examples)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_IMAGE="thecanadianroot/opencv-cuda:latest"
RUNTIME_IMAGE="cv-opencv-cuda-runtime:latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --runtime-image) RUNTIME_IMAGE="$2"; shift 2 ;;
    --help|-h)
      cat <<'EOF'
Build Docker environment for 04 - Opencv/CUDA
Usage:
  ./build_docker_env.sh
  ./build_docker_env.sh --base-image thecanadianroot/opencv-cuda:latest
  ./build_docker_env.sh --runtime-image cv-opencv-cuda-runtime:latest
EOF
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "[INFO] Building runtime image: $RUNTIME_IMAGE"

docker build -t "$RUNTIME_IMAGE" - <<EOF
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
# Ensure core tooling used by run_all_cuda_examples.sh exists
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake coreutils pkg-config \
 && rm -rf /var/lib/apt/lists/*
EOF

echo "[OK] Built image: $RUNTIME_IMAGE"

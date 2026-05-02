#!/bin/bash

# Build CUDA runtime image once (separate from running examples)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_IMAGE="nvidia/cuda:12.3.2-devel-ubuntu22.04"
RUNTIME_IMAGE="cv-cuda-runtime:12.3.2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --runtime-image) RUNTIME_IMAGE="$2"; shift 2 ;;
    --help|-h)
      cat <<'EOF'
Build Docker environment for 05 - CUDA
Usage:
  ./build_docker_env.sh
  ./build_docker_env.sh --base-image nvidia/cuda:12.3.2-devel-ubuntu22.04
  ./build_docker_env.sh --runtime-image cv-cuda-runtime:12.3.2
EOF
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "[INFO] Building runtime image: $RUNTIME_IMAGE"

docker build -t "$RUNTIME_IMAGE" - <<EOF
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*
EOF

echo "[OK] Built image: $RUNTIME_IMAGE"

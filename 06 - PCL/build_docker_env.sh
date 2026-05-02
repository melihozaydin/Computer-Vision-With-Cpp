#!/bin/bash

# Build PCL Docker runtime once (separate from running examples)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_IMAGE="ubuntu:22.04"
RUNTIME_IMAGE="cv-pcl-runtime:22.04"
FORCE_REBUILD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --runtime-image) RUNTIME_IMAGE="$2"; shift 2 ;;
    --force|--rebuild) FORCE_REBUILD=true; shift ;;
    --help|-h)
      cat <<'EOF'
Build Docker environment for 06 - PCL
Usage:
  ./build_docker_env.sh
  ./build_docker_env.sh --base-image ubuntu:22.04
  ./build_docker_env.sh --runtime-image cv-pcl-runtime:22.04
  ./build_docker_env.sh --force
EOF
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$FORCE_REBUILD" == "false" ]] && docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1; then
  echo "[OK] Runtime image already present: $RUNTIME_IMAGE"
  echo "[INFO] Skipping rebuild. Use --force to rebuild."
  exit 0
fi

echo "[INFO] Building runtime image: $RUNTIME_IMAGE"

docker build -t "$RUNTIME_IMAGE" - <<EOF
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libpcl-dev libvtk9-dev \
 && rm -rf /var/lib/apt/lists/*
EOF

echo "[OK] Built image: $RUNTIME_IMAGE"

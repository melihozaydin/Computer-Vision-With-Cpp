#!/usr/bin/env bash
# Compatibility/consistency wrapper:
# use run_all_examples.sh across folders
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_all_cuda_examples.sh" "$@"

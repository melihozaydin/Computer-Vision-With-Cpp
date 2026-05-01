#!/bin/bash

# Eigen Examples Runner (02 - Eigen)
# ============================================================
# Builds and runs all Eigen C++ examples using the local toolchain.
# Eigen is header-only — no Docker needed.
#
# Usage:
#   ./run_all_examples.sh              # build + run all
#   ./run_all_examples.sh --build-only # build only, skip running
#   ./run_all_examples.sh --run-only   # run only (assumes built)
#   ./run_all_examples.sh --timeout N  # per-example timeout in seconds (default: 15)
#   ./run_all_examples.sh --clean      # remove .build/ and exit
#   ./run_all_examples.sh --help       # show this help
#
# Prerequisites (local WSL/Linux):
#   sudo apt install build-essential libeigen3-dev

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
BUILD_ONLY=false
RUN_ONLY=false
CLEAN=false
TIMEOUT=15

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
        --build-only)  BUILD_ONLY=true;  shift ;;
        --run-only)    RUN_ONLY=true;    shift ;;
        --clean)       CLEAN=true;       shift ;;
        --timeout)     TIMEOUT="$2";     shift 2 ;;
        --help|-h)
                        cat <<'EOF'
Eigen Examples Runner (02 - Eigen)
Usage:
    ./run_all_examples.sh              # build + run all
    ./run_all_examples.sh --build-only # build only, skip running
    ./run_all_examples.sh --run-only   # run only (assumes built)
    ./run_all_examples.sh --timeout N  # per-example timeout in seconds (default: 15)
    ./run_all_examples.sh --clean      # remove .build/ and exit
    ./run_all_examples.sh --help       # show this help

Prerequisites (local WSL/Linux):
    sudo apt install build-essential libeigen3-dev
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

# ── Verify Eigen is available ─────────────────────────────────────────────────
if [[ "$RUN_ONLY" == "false" ]]; then
    if [[ ! -d /usr/include/eigen3 ]]; then
        echo -e "${RED}Eigen headers not found at /usr/include/eigen3${NC}"
        echo "Install with:  sudo apt install libeigen3-dev"
        exit 1
    fi
fi

# ── Build ─────────────────────────────────────────────────────────────────────
if [[ "$RUN_ONLY" == "false" ]]; then
    print_header "Building all examples"
    if ! make all 2>&1; then
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

for binary in .build/*; do
    [[ -f "$binary" && -x "$binary" ]] || continue
    name="$(basename "$binary")"
    total=$((total + 1))

    output=$(timeout "${TIMEOUT}s" "$binary" 2>&1)
    code=$?

    if [[ $code -eq 0 ]]; then
        print_ok "$name"
        ok=$((ok + 1))
    elif [[ $code -eq 124 ]]; then
        print_warn "$name  (timeout)"
        timeout_count=$((timeout_count + 1))
    else
        print_fail "$name  (exit $code)"
        [[ -n "$output" ]] && echo "$output" | head -5 | sed 's/^/    /'
        fail=$((fail + 1))
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
print_header "Summary"
echo -e "  ${GREEN}ok=${ok}${NC}  ${YELLOW}timeout=${timeout_count}${NC}  ${RED}fail=${fail}${NC}  total=${total}"

[[ $fail -gt 0 ]] && exit 1 || exit 0

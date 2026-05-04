#!/bin/bash
# =============================================================================
# ADIF_GUI — run script
# =============================================================================
# Builds adif_gui and launches it with PCD data.
# Defaults to ADIF synthetic data if it exists; pass custom PCDs as arguments.
#
# Usage:
#   ./run.sh                                         # use ADIF synthetic data
#   ./run.sh my_scan.pcd                             # custom scan only
#   ./run.sh my_scan.pcd my_reference.pcd            # custom scan + reference
#   ./run.sh --build-only                            # compile, don't launch
#   ./run.sh --clean                                 # clean build artefacts
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]  $1${NC}"; }
ok()    { echo -e "${GREEN}[OK]    $1${NC}"; }
warn()  { echo -e "${YELLOW}[WARN]  $1${NC}"; }
fail()  { echo -e "${RED}[FAIL]  $1${NC}"; exit 1; }

BUILD_ONLY=false
CLEAN=false
SCAN=""
REFERENCE=""

for arg in "$@"; do
  case "$arg" in
    --build-only) BUILD_ONLY=true ;;
    --clean)      CLEAN=true ;;
    --*)          warn "Unknown flag: $arg" ;;
    *)
      if   [[ -z "$SCAN"      ]]; then SCAN="$arg"
      elif [[ -z "$REFERENCE" ]]; then REFERENCE="$arg"
      else warn "Extra argument ignored: $arg"; fi
      ;;
  esac
done

# ── clean ─────────────────────────────────────────────────────────────────────
if $CLEAN; then
  info "Cleaning build artefacts..."
  rm -rf .build
  ok "Clean done."
  exit 0
fi

# ── build ─────────────────────────────────────────────────────────────────────
info "Building adif_gui..."
make -j"$(nproc)" 2>&1 | sed 's/^/  /'
ok "Build succeeded → .build/adif_gui"

$BUILD_ONLY && exit 0

# ── locate data ───────────────────────────────────────────────────────────────
ADIF_DATA="../ADIF/data"

if [[ -z "$SCAN" ]]; then
  if [[ -f "$ADIF_DATA/scanned_part.pcd" ]]; then
    SCAN="$ADIF_DATA/scanned_part.pcd"
    info "Using ADIF synthetic scan: $SCAN"
  else
    fail "No scan.pcd provided and ADIF synthetic data not found.\n" \
         "  Generate it first:  cd ../ADIF && make generate\n" \
         "  Or pass a PCD:      ./run.sh <scan.pcd> [<reference.pcd>]"
  fi
fi

if [[ -z "$REFERENCE" && -f "$ADIF_DATA/reference_part.pcd" ]]; then
  REFERENCE="$ADIF_DATA/reference_part.pcd"
  info "Using ADIF reference: $REFERENCE"
fi

# ── launch ────────────────────────────────────────────────────────────────────
CMD=(".build/adif_gui" "$SCAN")
[[ -n "$REFERENCE" ]] && CMD+=("$REFERENCE")

info "Launching: ${CMD[*]}"
"${CMD[@]}"

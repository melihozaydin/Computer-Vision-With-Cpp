#!/bin/bash

# PCL examples runner (06 - PCL)
# ============================================================
# Modes:
#   --local (default)   Build with host-installed libpcl-dev.
#   --docker            Build in Ubuntu container with libpcl-dev via apt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DOCKER_MODE=false
DOCKER_RUNTIME_IMAGE="cv-pcl-runtime:22.04"
BUILD_ONLY=false
RUN_ONLY=false
CLEAN=false
TIMEOUT=12
VERBOSE=true
GUI_MODE=false

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
    --docker-runtime-image|--image) DOCKER_RUNTIME_IMAGE="$2"; shift 2 ;;
    --build-only) BUILD_ONLY=true; shift ;;
    --run-only) RUN_ONLY=true; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --no-timeout) TIMEOUT=0; shift ;;
    --quiet) VERBOSE=false; shift ;;
    --gui|--docker-gui) GUI_MODE=true; shift ;;
    --clean) CLEAN=true; shift ;;
    --help|-h)
      cat <<'EOF'
PCL examples runner (06 - PCL)
Usage:
  ./run_all_examples.sh               # local build + run
  ./run_all_examples.sh --docker      # run in prebuilt Docker runtime image
  ./run_all_examples.sh --docker-runtime-image TAG # override runtime image tag
  ./run_all_examples.sh --build-only  # compile only
  ./run_all_examples.sh --run-only    # run existing binaries only
  ./run_all_examples.sh --timeout N   # per-example timeout (default: 12)
  ./run_all_examples.sh --no-timeout  # disable per-example timeout
  ./run_all_examples.sh --quiet       # reduce log verbosity
  ./run_all_examples.sh --gui         # enable GUI forwarding (local/docker)
  ./run_all_examples.sh --clean       # remove .build/ and exit
EOF
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$CLEAN" == "true" ]]; then
  rm -rf .build synthetic_line_cloud.pcd
  echo "Cleaned outputs"
  exit 0
fi

if [[ "$DOCKER_MODE" == "true" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  CONTAINER_NAME="cv-pcl-run-$(date +%s)-$$"
  PASS_ARGS="--local --timeout $TIMEOUT"
  [[ "$BUILD_ONLY" == "true" ]] && PASS_ARGS="$PASS_ARGS --build-only"
  [[ "$RUN_ONLY" == "true" ]] && PASS_ARGS="$PASS_ARGS --run-only"
  [[ "$VERBOSE" == "false" ]] && PASS_ARGS="$PASS_ARGS --quiet"
  [[ "$GUI_MODE" == "true" ]] && PASS_ARGS="$PASS_ARGS --gui"

  GUI_DOCKER_ARGS=""
  if [[ "$GUI_MODE" == "true" ]]; then
    # WSLg-friendly defaults in case DISPLAY vars are empty in non-interactive shells
    [[ -z "${DISPLAY:-}" ]] && [[ -d /mnt/wslg ]] && export DISPLAY=:0
    [[ -z "${WAYLAND_DISPLAY:-}" ]] && [[ -d /mnt/wslg ]] && export WAYLAND_DISPLAY=wayland-0
    [[ -z "${XDG_RUNTIME_DIR:-}" ]] && [[ -d /mnt/wslg/runtime-dir ]] && export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
    [[ -z "${XDG_SESSION_TYPE:-}" ]] && export XDG_SESSION_TYPE=x11
    [[ -z "${QT_QPA_PLATFORM:-}" ]] && export QT_QPA_PLATFORM=xcb

    GUI_DOCKER_ARGS="-e DISPLAY=${DISPLAY:-} -e WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-} -e XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-} -e XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-x11} -e QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb} -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg"
  fi

  if ! docker image inspect "$DOCKER_RUNTIME_IMAGE" >/dev/null 2>&1; then
    print_fail "Docker runtime image '$DOCKER_RUNTIME_IMAGE' not found."
    print_info "Build it once with: ./build_docker_env.sh"
    exit 1
  fi

  print_info "Image: $DOCKER_RUNTIME_IMAGE"
  print_info "Container: $CONTAINER_NAME"
  docker run --rm --name "$CONTAINER_NAME" $GUI_DOCKER_ARGS -v "$REPO_ROOT:/workspace" "$DOCKER_RUNTIME_IMAGE" bash -lc "
    set -e
    cd '/workspace/06 - PCL'
    ./run_all_examples.sh $PASS_ARGS
  "
  exit $?
fi

# Local GUI fallback for WSLg when launched from non-interactive shell
if [[ "$GUI_MODE" == "true" ]]; then
  [[ -z "${DISPLAY:-}" ]] && [[ -d /mnt/wslg ]] && export DISPLAY=:0
  [[ -z "${WAYLAND_DISPLAY:-}" ]] && [[ -d /mnt/wslg ]] && export WAYLAND_DISPLAY=wayland-0
  [[ -z "${XDG_RUNTIME_DIR:-}" ]] && [[ -d /mnt/wslg/runtime-dir ]] && export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
  [[ -z "${XDG_SESSION_TYPE:-}" ]] && export XDG_SESSION_TYPE=x11
  [[ -z "${QT_QPA_PLATFORM:-}" ]] && export QT_QPA_PLATFORM=xcb
fi

if [[ "$RUN_ONLY" == "false" ]]; then
  BUILD_CONTEXT="host"
  [[ -f /.dockerenv ]] && BUILD_CONTEXT="container"
  BUILD_ORIGIN_FILE=".build/.build_origin"

  if [[ -d .build ]] && [[ -f "$BUILD_ORIGIN_FILE" ]]; then
    PREV_CONTEXT="$(cat "$BUILD_ORIGIN_FILE" 2>/dev/null || true)"
    if [[ -n "$PREV_CONTEXT" ]] && [[ "$PREV_CONTEXT" != "$BUILD_CONTEXT" ]]; then
      print_info "Build context switched ($PREV_CONTEXT -> $BUILD_CONTEXT): cleaning stale artifacts"
      make clean >/dev/null 2>&1 || rm -rf .build
    fi
  fi

  if ! command -v make >/dev/null 2>&1; then
    print_fail "make not found. Install build-essential or use --docker."
    exit 1
  fi
  if ! command -v pkg-config >/dev/null 2>&1; then
    print_fail "pkg-config not found. Install pkg-config or use --docker."
    exit 1
  fi
  if ! pkg-config --exists pcl_common && ! pkg-config --exists pcl_common-1.13 && ! pkg-config --exists pcl_common-1.12; then
    print_fail "PCL pkg-config modules not found. Install libpcl-dev or use --docker."
    exit 1
  fi
  print_header "Building PCL examples"
  if ! make all; then
    print_fail "Build failed"
    exit 1
  fi
  mkdir -p .build
  echo "$BUILD_CONTEXT" > "$BUILD_ORIGIN_FILE"
  print_ok "Build complete → .build/"
fi

[[ "$BUILD_ONLY" == "true" ]] && exit 0

print_header "Running PCL examples (timeout: ${TIMEOUT}s each)"
ok=0; timeout_count=0; headless_skip=0; fail=0; total=0
mapfile -t targets < <(find . -maxdepth 1 -type f -name '*.cpp' -printf '%f\n' | sed 's/\.cpp$//' | sort)

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
    if timeout --signal=INT --kill-after=2s "${TIMEOUT}s" "$binary" >"$tmp_out" 2>&1; then
      code=0
    else
      code=$?
    fi
  fi
  if [[ "$VERBOSE" == "true" ]] && [[ -s "$tmp_out" ]]; then
    sed 's/^/    /' "$tmp_out" | head -40
  fi
  if grep -Eq "Visualization skipped \(headless environment\)|bad X server connection\. DISPLAY=|basic_string: construction from null is not valid" "$tmp_out"; then
    if [[ "$GUI_MODE" == "true" ]]; then
      print_warn "⚠ $name  (GUI requested but visualization was skipped in this environment)"
    else
      print_warn "⚠ $name  (headless: visualization skipped; compute output above is valid)"
    fi
    headless_skip=$((headless_skip + 1))
  elif [[ $code -eq 0 ]]; then
    print_ok "✓ $name"
    ok=$((ok + 1))
  elif [[ $code -eq 124 ]] || [[ $code -eq 139 && $(grep -c "timeout: the monitored command dumped core" "$tmp_out") -gt 0 ]]; then
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
echo -e "  ${GREEN}ok=${ok}${NC}  ${YELLOW}headless=${headless_skip}${NC}  ${YELLOW}timeout=${timeout_count}${NC}  ${RED}fail=${fail}${NC}  total=${total}"
[[ $fail -gt 0 ]] && exit 1 || exit 0

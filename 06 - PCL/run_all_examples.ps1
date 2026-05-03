# PCL examples runner (06 - PCL) — PowerShell wrapper
# ============================================================
# Delegates all build/run work to WSL via the bash runner.
# Requirements: WSL with libpcl-dev installed.
#
# Usage:
#   .\run_all_examples.ps1                      # local build + run
#   .\run_all_examples.ps1 --build-only         # compile only
#   .\run_all_examples.ps1 --run-only           # run existing binaries only
#   .\run_all_examples.ps1 --timeout 20         # per-example timeout (default: 12)
#   .\run_all_examples.ps1 --no-timeout         # disable per-example timeout
#   .\run_all_examples.ps1 --quiet              # reduce log verbosity
#   .\run_all_examples.ps1 --gui                # enable GUI (WSLg)
#   .\run_all_examples.ps1 --with-adif          # also build + smoke-test ADIF/
#   .\run_all_examples.ps1 --clean              # remove .build/ and exit
#   .\run_all_examples.ps1 --docker             # run in Docker runtime image
#   .\run_all_examples.ps1 --docker-runtime-image TAG
#   .\run_all_examples.ps1 --help               # show this help

# Accept all arguments as-is so we mirror the bash CLI (--flag style)
[CmdletBinding(PositionalBinding = $false)]
param([Parameter(ValueFromRemainingArguments)][string[]]$PassArgs)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── helpers ──────────────────────────────────────────────────
function Write-Info  { param($m) Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Fail  { param($m) Write-Host $m -ForegroundColor Red }

# ── show help locally and exit ────────────────────────────────
if ($PassArgs -contains '--help' -or $PassArgs -contains '-h') {
    Write-Host @"
PCL examples runner (06 - PCL)
Usage:
  .\run_all_examples.ps1                       # local build + run
  .\run_all_examples.ps1 --docker              # run in prebuilt Docker runtime image
  .\run_all_examples.ps1 --docker-runtime-image TAG
  .\run_all_examples.ps1 --build-only          # compile only
  .\run_all_examples.ps1 --run-only            # run existing binaries only
  .\run_all_examples.ps1 --timeout N           # per-example timeout (default: 12)
  .\run_all_examples.ps1 --no-timeout          # disable per-example timeout
  .\run_all_examples.ps1 --quiet               # reduce log verbosity
  .\run_all_examples.ps1 --gui                 # enable GUI forwarding (WSLg)
  .\run_all_examples.ps1 --with-adif           # also build + smoke-test ADIF/
  .\run_all_examples.ps1 --clean               # remove .build/ and exit
"@
    exit 0
}

# ── verify WSL is available ───────────────────────────────────
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Write-Fail "wsl.exe not found. Install WSL 2 and set up an Ubuntu distribution."
    exit 1
}

# ── resolve WSL path for this script's directory ─────────────
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Convert Windows path to WSL path  e.g.  D:\Work\... -> /mnt/d/Work/...
$wslPath = $scriptDir -replace '\\', '/'
if ($wslPath -match '^([A-Za-z]):(.*)') {
    $wslPath = "/mnt/$($Matches[1].ToLower())$($Matches[2])"
}

# ── pass all flags straight through to the bash runner ───────
$bashArgsStr = if ($PassArgs) { $PassArgs -join ' ' } else { '' }
$wslCmd = "cd '$wslPath' && chmod +x ./run_all_examples.sh && ./run_all_examples.sh $bashArgsStr"

Write-Info "WSL path : $wslPath"
Write-Info "Runner   : ./run_all_examples.sh $bashArgsStr"
Write-Host ""

# ── execute ───────────────────────────────────────────────────
wsl.exe bash -lc $wslCmd
exit $LASTEXITCODE

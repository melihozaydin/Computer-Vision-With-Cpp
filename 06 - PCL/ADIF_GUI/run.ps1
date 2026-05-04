# =============================================================================
# ADIF_GUI - run script (PowerShell / WSL wrapper)
# =============================================================================
# Delegates build and launch to WSL via run.sh.
#
# Usage:
#   .\run.ps1                                # use ADIF synthetic data
#   .\run.ps1 my_scan.pcd                    # custom scan only
#   .\run.ps1 my_scan.pcd my_ref.pcd         # custom scan + reference
#   .\run.ps1 --build-only                   # compile, don't launch
#   .\run.ps1 --clean                        # clean build artefacts
# =============================================================================

[CmdletBinding(PositionalBinding = $false)]
param([Parameter(ValueFromRemainingArguments)][string[]]$PassArgs)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO]  $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK]    $Message" -ForegroundColor Green
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[FAIL]  $Message" -ForegroundColor Red
}

$wsl = Get-Command wsl -ErrorAction SilentlyContinue
if (-not $wsl) {
    Write-Fail "WSL not found. Install WSL 2 with a Linux distro to continue."
    exit 1
}

$WslArgs = @()
foreach ($arg in $PassArgs) {
    if ($arg -match '^[A-Za-z]:\\') {
        $wslPath = (& wsl wslpath -u $arg 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($wslPath)) {
            $WslArgs += $wslPath
        } else {
            $WslArgs += $arg
        }
    } else {
        $WslArgs += $arg
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WslScriptDir = (& wsl wslpath -u $ScriptDir | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($WslScriptDir)) {
    Write-Fail "Failed to convert the script directory to a WSL path."
    exit 1
}

$dispCheck = (& wsl printenv DISPLAY 2>$null | Out-String).Trim()
$WslCommandArgs = @('--cd', $WslScriptDir, 'env')

if (-not [string]::IsNullOrWhiteSpace($dispCheck)) {
    $WslCommandArgs += "DISPLAY=$dispCheck"
    Write-Info ("WSLg display detected: " + $dispCheck)
} else {
    $WslCommandArgs += @(
        'PCL_FORCE_HEADLESS=1',
        'DISPLAY=',
        'WAYLAND_DISPLAY=',
        'XDG_SESSION_TYPE='
    )
    Write-Info "No WSLg display detected - viewer will be skipped (headless)."
}

$WslCommandArgs += @('bash', './run.sh')
$WslCommandArgs += $WslArgs

Write-Info ("Running via WSL from " + $WslScriptDir)
& wsl @WslCommandArgs
if ($LASTEXITCODE -ne 0) {
    Write-Fail ("run.sh exited with code " + $LASTEXITCODE)
    exit $LASTEXITCODE
}

Write-Ok "Done."

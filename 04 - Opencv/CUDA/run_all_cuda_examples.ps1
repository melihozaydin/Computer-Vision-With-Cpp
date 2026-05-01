<#
.SYNOPSIS
  Windows PowerShell wrapper for run_all_cuda_examples.sh

.DESCRIPTION
  Lets you launch the CUDA build/run-all workflow from Windows PowerShell
  while executing the actual script inside WSL.

.EXAMPLE
  .\run_all_cuda_examples.ps1

.EXAMPLE
  .\run_all_cuda_examples.ps1 -Mode docker -Timeout 10

.EXAMPLE
  .\run_all_cuda_examples.ps1 -Mode local -BuildOnly
#>

[CmdletBinding()]
param(
    [ValidateSet('docker', 'local')]
    [string]$Mode = 'docker',

    [int]$Timeout = 6,

    [string]$Image,

    [string]$OpenCvDir,

    [switch]$BuildOnly,
    [switch]$RunOnly,
    [switch]$Clean,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'

# Locate this script's folder (04 - Opencv/CUDA)
$scriptDirWin = Split-Path -Parent $MyInvocation.MyCommand.Path

# Convert Windows path to WSL path (stable /mnt/<drive>/... form)
if ($scriptDirWin -match '^(?<drive>[A-Za-z]):\\(?<rest>.*)$') {
    $drive = $Matches['drive'].ToLowerInvariant()
    $rest = $Matches['rest'] -replace '\\', '/'
    $scriptDirWsl = "/mnt/$drive/$rest"
}
else {
    # Fallback for non-drive paths
    $scriptDirWsl = (wsl.exe wslpath -a "$scriptDirWin").Trim()
}

if (-not $scriptDirWsl) {
    throw "Failed to convert path to WSL format: $scriptDirWin"
}

# Build argument list to forward to the bash script
$forwardArgs = New-Object System.Collections.Generic.List[string]
$forwardArgs.Add("--$Mode")
$forwardArgs.Add("--timeout")
$forwardArgs.Add([string]$Timeout)

if ($Image) {
    $forwardArgs.Add('--image')
    $forwardArgs.Add($Image)
}
if ($OpenCvDir) {
    $forwardArgs.Add('--opencv-dir')
    $forwardArgs.Add($OpenCvDir)
}
if ($BuildOnly) { $forwardArgs.Add('--build-only') }
if ($RunOnly)   { $forwardArgs.Add('--run-only') }
if ($Clean)     { $forwardArgs.Add('--clean') }
if ($Help)      { $forwardArgs.Add('--help') }

function Quote-BashDouble([string]$s) {
  # Escape for bash double-quoted context
  $e = $s.Replace('\', '\\').Replace('"', '\"').Replace('$', '\$').Replace('`', '\`')
  return '"' + $e + '"'
}

# Escape args for safe bash double-quoted usage
$quotedArgs = $forwardArgs | ForEach-Object { Quote-BashDouble $_ }
$argString = [string]::Join(' ', $quotedArgs)

# Run in WSL
# Use escaped spaces for `cd` because nested shell quoting can be finicky on Windows->WSL hops.
$cdPath = $scriptDirWsl -replace ' ', '\ '
$bashCmd = "cd $cdPath && chmod +x ./run_all_cuda_examples.sh && ./run_all_cuda_examples.sh $argString"
Write-Host "[INFO] Running in WSL: $bashCmd"

wsl.exe -e bash -lc $bashCmd
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

<#
.SYNOPSIS
    LightVC development helper — one-command GUI launch with auto DAC weights.

.DESCRIPTION
    Builds the standalone app (release), ensures the DAC weights are present
    at models/dac_44khz.safetensors (downloading ~307 MB from HuggingFace
    on first run), then launches the GUI.

    Converter weights are optional — the GUI starts without them. Load a
    trained checkpoint later via the "Load Converter" button on the
    Realtime tab once the training agent finishes.

.PARAMETER NoBuild
    Skip the cargo build step (use the existing binary).

.PARAMETER BuildOnly
    Build only, do not launch.

.PARAMETER Cuda
    Pass --cuda to the app (enables CUDA device).

.PARAMETER Metal
    Pass --metal to the app (enables Metal device on macOS).

.PARAMETER Roundtrip
    Run a DAC round-trip test on the given WAV instead of launching GUI.

.PARAMETER Input
    WAV file for -Roundtrip.

.EXAMPLE
    .\dev.ps1
    Build + download DAC weights + launch GUI.

.EXAMPLE
    .\dev.ps1 -NoBuild
    Launch GUI using the existing release binary.

.EXAMPLE
    .\dev.ps1 -Roundtrip -Input C:\audio\test.wav
    Validate DAC encode/decode on a sample.
#>
[CmdletBinding()]
param(
    [switch]$NoBuild,
    [switch]$BuildOnly,
    [switch]$Cuda,
    [switch]$Metal,
    [switch]$Roundtrip,
    [string]$Input,
    [string]$Output
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$modelsDir = Join-Path $repoRoot 'models'
$dacPath   = Join-Path $modelsDir 'dac_44khz.safetensors'
$dacUrl    = 'https://huggingface.co/descript/dac_44khz/resolve/main/model.safetensors'
$binPath   = Join-Path $repoRoot 'target\release\lightvc-app.exe'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Err($msg)  { Write-Host "    $msg" -ForegroundColor Red }

# --- 1. Build ---------------------------------------------------------------
if (-not $NoBuild) {
    Write-Step 'Building release binary (cargo build --release -p lightvc-app)'
    & cargo build --release -p lightvc-app
    if ($LASTEXITCODE -ne 0) {
        Write-Err 'Build failed.'
        exit 1
    }
    Write-Ok 'Build OK.'
}

if ($BuildOnly) {
    Write-Ok 'Build-only requested; not launching.'
    return
}

if (-not (Test-Path $binPath)) {
    Write-Err "Binary not found: $binPath"
    Write-Err 'Run again without -NoBuild, or build manually.'
    exit 1
}

# --- 2. DAC weights ---------------------------------------------------------
if (-not (Test-Path $dacPath)) {
    Write-Step "Downloading DAC weights (~307 MB) to models\dac_44khz.safetensors"
    if (-not (Test-Path $modelsDir)) {
        New-Item -ItemType Directory -Path $modelsDir | Out-Null
    }
    # Use BITS for resumable transfer on Windows; fall back to Invoke-WebRequest.
    $tmp = "$dacPath.tmp"
    try {
        Start-BitsTransfer -Source $dacUrl -Destination $tmp -DisplayName 'LightVC DAC weights'
        Move-Item -Force $tmp $dacPath
    } catch {
        Write-Host '    BITS unavailable, using Invoke-WebRequest...' -ForegroundColor Yellow
        try {
            Invoke-WebRequest -Uri $dacUrl -OutFile $tmp -UseBasicParsing
            Move-Item -Force $tmp $dacPath
        } catch {
            Write-Err "Download failed: $_"
            Write-Err "Manual download: $dacUrl"
            Write-Err "Place at: $dacPath"
            exit 1
        }
    }
    Write-Ok 'DAC weights downloaded.'
} else {
    Write-Ok "DAC weights present: $dacPath"
}

# --- 3. Launch --------------------------------------------------------------
if ($Roundtrip) {
    if (-not $Input) {
        Write-Err '-Roundtrip requires -Input <wav>'
        exit 1
    }
    $out = if ($Output) { $Output } else { 'roundtrip_output.wav' }
    Write-Step "Running DAC round-trip: $Input -> $out"
    & $binPath roundtrip --input $Input --output $out --dac-weights $dacPath
    exit $LASTEXITCODE
}

$deviceFlags = @()
if ($Cuda)  { $deviceFlags += '--cuda' }
if ($Metal) { $deviceFlags += '--metal' }

Write-Step 'Launching LightVC GUI'
Write-Host  '    (converter weights optional — load later via Realtime tab)' -ForegroundColor DarkGray
& $binPath gui --dac-weights $dacPath @deviceFlags
exit $LASTEXITCODE

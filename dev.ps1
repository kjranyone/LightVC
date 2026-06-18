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

    Demo mode (--Demo) renders with mock data and needs no DAC weights or
    model, useful for layout review. Tabs are switchable inside the app.
    Use --Snap to capture the running window to PNG on demand.

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

.PARAMETER Demo
    Launch the GUI in demo mode with mock data. No DAC weights or model
    required. Tabs are switchable inside the app.

.PARAMETER Snap
    Capture the running LightVC window to PNG (does not launch the app).
    Saves to docs/screenshots/snap-<timestamp>.png.

.EXAMPLE
    .\dev.ps1
    Build + download DAC weights + launch GUI.

.EXAMPLE
    .\dev.ps1 -NoBuild
    Launch GUI using the existing release binary.

.EXAMPLE
    .\dev.ps1 -Roundtrip -Input C:\audio\test.wav
    Validate DAC encode/decode on a sample.

.EXAMPLE
    .\dev.ps1 -Demo
    Launch GUI in demo mode (no model/DAC needed). Switch tabs in the app.

.EXAMPLE
    .\dev.ps1 -Snap
    Capture the running window to docs/screenshots/.
#>
[CmdletBinding()]
param(
    [switch]$NoBuild,
    [switch]$BuildOnly,
    [switch]$Cuda,
    [switch]$Metal,
    [switch]$Roundtrip,
    [switch]$Snap,
    [string]$Input,
    [string]$Output,
    [switch]$Demo
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
function Write-Menu($msg) { Write-Host $msg -ForegroundColor Yellow }

# --- 0. Interactive menu (when no action flags given) -----------------------
function Show-Menu {
    Write-Host ''
    Write-Host '  ╔══════════════════════════════════════════╗' -ForegroundColor Magenta
    Write-Host '  ║          LightVC Development             ║' -ForegroundColor Magenta
    Write-Host '  ╚══════════════════════════════════════════╝' -ForegroundColor Magenta
    Write-Host ''
    Write-Menu '  [1] GUI を起動（通常・DAC重みDL + ビルド）'
    Write-Menu '  [2] GUI を起動（ビルドスキップ）'
    Write-Menu '  [3] デモモードで起動（モデル/DAC不要）'
    Write-Menu '  [4] DAC ラウンドトリップテスト'
    Write-Menu '  [5] ビルドのみ'
    Write-Menu '  [6] 起動中アプリのスクショを撮る（on-demand）'
    Write-Menu '  [Q] 終了'
    Write-Host ''
    return (Read-Host '番号を選択')
}

# --- Snap: capture running window only (no build, no launch) ---------------
if ($Snap) {
    $snapScript = Join-Path $repoRoot 'tools\snap.ps1'
    $snapArgs = @()
    if ($Output) { $snapArgs += '-Out'; $snapArgs += $Output }
    & $snapScript @snapArgs
    exit $LASTEXITCODE
}

$hasAction = $BuildOnly -or $Cuda -or $Metal -or $Roundtrip -or $Demo -or $Input -or $Output
if (-not $hasAction) {
    $choice = Show-Menu
    Write-Host ''
    if ($choice -eq '1') {
        $action = 'gui'
    } elseif ($choice -eq '2') {
        $NoBuild = $true
        $action = 'gui'
    } elseif ($choice -eq '3') {
        $Demo = $true
        $action = 'demo'
    } elseif ($choice -eq '4') {
        $Roundtrip = $true
        $Input = Read-Host 'WAV ファイルパス'
        $action = 'roundtrip'
    } elseif ($choice -eq '5') {
        $BuildOnly = $true
        $action = 'buildonly'
    } elseif ($choice -eq '6') {
        $Snap = $true
        $action = 'snap'
    } elseif ($choice -eq 'q' -or $choice -eq 'Q') {
        exit 0
    } else {
        Write-Err ('無効な選択: ' + $choice)
        exit 1
    }
    if ($action -eq 'roundtrip' -and -not $Input) { exit 1 }
}

# Demo mode needs neither DAC weights nor a model.
$skipDac = $Demo

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

# --- 2. DAC weights (skipped in demo/screenshot mode) -----------------------
if (-not $skipDac) {
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
}

# --- 3. Roundtrip test ------------------------------------------------------
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

# --- 5. Launch GUI ----------------------------------------------------------
$deviceFlags = @()
if ($Cuda)  { $deviceFlags += '--cuda' }
if ($Metal) { $deviceFlags += '--metal' }

if ($Demo) {
    Write-Step 'Launching LightVC GUI in demo mode'
    Write-Host  '    (mock data, no model/DAC required; tabs switchable inside the app)' -ForegroundColor DarkGray
    & $binPath gui --demo @deviceFlags
} else {
    Write-Step 'Launching LightVC GUI'
    Write-Host  '    (converter weights optional — load later via Realtime tab)' -ForegroundColor DarkGray
    & $binPath gui --dac-weights $dacPath @deviceFlags
}
exit $LASTEXITCODE

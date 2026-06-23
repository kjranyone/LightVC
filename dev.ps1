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
    required. Tabs are switchable inside the app. The menu entry [3] implies
    -DebugBuild for fast incremental builds during UI review.

.PARAMETER Snap
    Capture the running LightVC window to PNG (does not launch the app).
    Saves to docs/screenshots/snap-<timestamp>.png.

.PARAMETER DebugBuild
    Build and run the debug binary (target\debug) instead of release.
    Incremental builds are ~3-5s vs minutes for release. Recommended for UI
    layout work where inference performance is irrelevant. (Named DebugBuild
    to avoid clashing with PowerShell's built-in -Debug common parameter.)

.PARAMETER Watch
    UI review loop: build (debug) + launch in demo mode, then poll
    crates/lightvc-app/src for .rs edits and automatically rebuild ->
    relaunch -> re-screenshot. Sub-10s iteration. Ctrl+C to exit.

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
    .\dev.ps1 -DebugBuild -Demo
    Fast UI review: debug build + demo mode. Use this for layout iteration.

.EXAMPLE
    .\dev.ps1 -Watch
    Auto-reload loop: edit .rs -> rebuild -> relaunch -> screenshot.
    Ideal for AI-assisted GUI layout review.

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
    [switch]$Demo,
    [switch]$DebugBuild,
    [switch]$Watch
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$modelsDir = Join-Path $repoRoot 'models'
$dacPath   = Join-Path $modelsDir 'dac_44khz.safetensors'
$dacUrl    = 'https://huggingface.co/descript/dac_44khz/resolve/main/model.safetensors'

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
    Write-Menu '  [1] GUI を起動（通常・release・DAC重みDL + ビルド）'
    Write-Menu '  [2] GUI を起動（release・ビルドスキップ）'
    Write-Menu '  [3] デモ＋Watch（debug・.rs編集で自動リロード・AI レビュー用）'
    Write-Menu '  [4] DAC ラウンドトリップテスト'
    Write-Menu '  [5] ビルドのみ'
    Write-Menu '  [Q] 終了'
    Write-Host ''
    return (Read-Host '番号を選択')
}

# Launch LightVC in background, wait for window, capture to live.png.
# App stays running after capture so user can re-snap via tools/snap.ps1.
function Invoke-LightvcWithCapture {
    param(
        [Parameter(Mandatory)][string[]]$LaunchArgs,
        [string]$Out = (Join-Path $repoRoot 'docs\screenshots\live.png'),
        [int]$WindowTimeoutSec = 20,
        [int]$RenderDelayMs = 3000
    )
    Write-Step 'LightVC をバックグラウンド起動中'
    $proc = Start-Process -FilePath $binPath -ArgumentList $LaunchArgs -PassThru
    if (-not $proc) {
        Write-Err '起動に失敗しました。'
        exit 1
    }
    Write-Host  "    PID: $($proc.Id)" -ForegroundColor DarkGray

    Write-Step 'ウィンドウ出現を待機中'
    $deadline = (Get-Date).AddSeconds($WindowTimeoutSec)
    $found = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 200
        $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if ($p -and $p.MainWindowHandle -ne [IntPtr]::Zero) { $found = $true; break }
    }
    if (-not $found) {
        Write-Err "ウィンドウが ${WindowTimeoutSec}s 以内に出現しませんでした。"
        exit 1
    }
    Write-Ok 'ウィンドウ検出。'

    Write-Step "初期描画待機 (${RenderDelayMs}ms)"
    Start-Sleep -Milliseconds $RenderDelayMs

    Write-Step "スクショ撮影 -> $Out"
    $snapScript = Join-Path $repoRoot 'tools\snap.ps1'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $snapScript -Out $Out
    if ($LASTEXITCODE -ne 0) {
        Write-Err 'スクショ失敗。'
        exit 1
    }
    Write-Ok "保存済み: $Out"
    Write-Host  '    アプリは起動したままです。タブ切替後に tools\snap.ps1 で再撮影できます。' -ForegroundColor DarkGray
}

# --- Snap: capture running window only (no build, no launch) ---------------
if ($Snap) {
    $snapScript = Join-Path $repoRoot 'tools\snap.ps1'
    $snapArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $snapScript)
    if ($Output) { $snapArgs += '-Out'; $snapArgs += $Output }
    & powershell @snapArgs
    exit $LASTEXITCODE
}

# --- Watch: rebuild + relaunch on .rs change (demo/debug UI review loop) ----
# Polls crates/lightvc-app/src for .rs edits, then: kill -> build -> launch
# -> snap. Optimized for AI-driven GUI layout review (sub-10s iteration).
function Invoke-WatchLoop {
    # Reject duplicate watch instances — prevent multi-launch storms.
    $lockFile = Join-Path $repoRoot 'target\.watch.lock'
    if (Test-Path $lockFile) {
        $prevPid = Get-Content $lockFile -ErrorAction SilentlyContinue
        if ($prevPid -and (Get-Process -Id $prevPid -ErrorAction SilentlyContinue)) {
            Write-Err 'Watch is already running. Stop it first (Ctrl+C in its terminal).'
            Write-Host "    PID: $prevPid" -ForegroundColor DarkGray
            exit 1
        }
    }
    $PID | Out-File -FilePath $lockFile -Encoding ascii -NoNewline
    $cleanup = {
        Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
        Get-Process -Name 'lightvc-app' -ErrorAction SilentlyContinue |
            Stop-Process -Force -ErrorAction SilentlyContinue
    }
    Register-EngineEvent PowerShell.Exiting -Action $cleanup | Out-Null

    # cargo writes progress to stderr; under 'Stop' the 2>&1 redirect turns
    # that into a terminating error. Relax for the whole loop.
    $ErrorActionPreference = 'Continue'
    $Demo = $true
    $DebugBuild = $true
    $binPath = Join-Path $repoRoot 'target\debug\lightvc-app.exe'
    $srcDir = Join-Path $repoRoot 'crates\lightvc-app\src'
    $snapScript = Join-Path $repoRoot 'tools\snap.ps1'
    $livePng = Join-Path $repoRoot 'docs\screenshots\live.png'

    function Stop-Lightvc {
        Get-Process -Name 'lightvc-app' -ErrorAction SilentlyContinue | ForEach-Object {
            try { $_.Kill(); $_.WaitForExit(3000) | Out-Null } catch {}
        }
        $deadline = (Get-Date).AddSeconds(3)
        while ((Get-Date) -lt $deadline -and (Get-Process -Name 'lightvc-app' -ErrorAction SilentlyContinue)) {
            Start-Sleep -Milliseconds 200
        }
    }

    Write-Step '初期ビルド（debug）'
    & cargo build -p lightvc-app
    if ($LASTEXITCODE -ne 0) { Write-Err '初回ビルド失敗'; & $cleanup; exit 1 }

    Stop-Lightvc
    Invoke-LightvcWithCapture -LaunchArgs @('gui', '--demo')

    # Record initial timestamps of all watched .rs files.
    $state = @{}
    Get-ChildItem -Path $srcDir -Filter '*.rs' -Recurse -ErrorAction SilentlyContinue |
        ForEach-Object { $state[$_.FullName] = $_.LastWriteTime }

    Write-Step '監視中 — .rs を編集すると再ビルド → 再起動 → スクショ更新'
    Write-Host  '    Ctrl+C で終了' -ForegroundColor DarkGray

    while ($true) {
        Start-Sleep -Milliseconds 500
        $changed = $false
        Get-ChildItem -Path $srcDir -Filter '*.rs' -Recurse -ErrorAction SilentlyContinue |
            ForEach-Object {
                $prev = $state[$_.FullName]
                if (-not $prev -or $_.LastWriteTime -gt $prev) {
                    $changed = $true
                    $state[$_.FullName] = $_.LastWriteTime
                }
            }
        if (-not $changed) { continue }

        # Debounce: collapse bursts of saves (formatter, multi-file edits).
        Start-Sleep -Milliseconds 800
        Get-ChildItem -Path $srcDir -Filter '*.rs' -Recurse -ErrorAction SilentlyContinue |
            ForEach-Object { $state[$_.FullName] = $_.LastWriteTime }

        Write-Step '変更検知 → 再ビルド'
        Stop-Lightvc

        $buildOut = & cargo build -p lightvc-app 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            Write-Err 'ビルド失敗 — .rs を修正して保存し直してください'
            $buildOut | Select-Object -Last 5 | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
            continue
        }

        $proc = Start-Process -FilePath $binPath -ArgumentList @('gui','--demo') -PassThru
        $deadline = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 200
            $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
            if ($p -and $p.MainWindowHandle -ne [IntPtr]::Zero) { break }
        }
        Start-Sleep -Milliseconds 2500
        & powershell -NoProfile -ExecutionPolicy Bypass -File $snapScript -Out $livePng *> $null
        Write-Ok "リロード完了 $(Get-Date -Format 'HH:mm:ss') — スクショ更新済み"
    }
}

$hasAction = $BuildOnly -or $Cuda -or $Metal -or $Roundtrip -or $Demo -or $Watch -or $Input -or $Output
if (-not $hasAction) {
    $choice = Show-Menu
    Write-Host ''
    if ($choice -eq '1') {
        $action = 'gui'
    } elseif ($choice -eq '2') {
        $NoBuild = $true
        $action = 'gui'
    } elseif ($choice -eq '3') {
        $Watch = $true
        $action = 'watch'
    } elseif ($choice -eq '4') {
        $Roundtrip = $true
        $Input = Read-Host 'WAV ファイルパス'
        $action = 'roundtrip'
    } elseif ($choice -eq '5') {
        $BuildOnly = $true
        $action = 'buildonly'
    } elseif ($choice -eq 'q' -or $choice -eq 'Q') {
        exit 0
    } else {
        Write-Err ('無効な選択: ' + $choice)
        exit 1
    }
    if ($action -eq 'roundtrip' -and -not $Input) { exit 1 }
}

# Watch mode: build + launch demo, then poll for .rs edits and auto-reload.
# Triggered by -Watch flag or menu [3]. Exits never (Ctrl+C to stop).
if ($Watch) {
    Invoke-WatchLoop
    exit 0
}

# Demo mode needs neither DAC weights nor a model.
$skipDac = $Demo

# Binary path depends on build profile. Computed here (after the menu may have
# set $DebugBuild) so demo mode uses the fast debug binary.
$profileDir = if ($DebugBuild) { 'debug' } else { 'release' }
$binPath    = Join-Path $repoRoot "target\$profileDir\lightvc-app.exe"

# --- 1. Build ---------------------------------------------------------------
if (-not $NoBuild) {
    $buildArgs = @('build', '-p', 'lightvc-app')
    if (-not $DebugBuild) { $buildArgs += '--release' }
    $profileLabel = if ($DebugBuild) { 'debug' } else { 'release' }
    Write-Step "Building $profileLabel binary (cargo build $buildArgs)"
    $ErrorActionPreference = 'Continue'
    & cargo @buildArgs
    $ErrorActionPreference = 'Stop'
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
    Write-Step 'デモモードで起動（自動スクショ付き）'
    Write-Host  '    (mock data, no model/DAC required; tabs switchable inside the app)' -ForegroundColor DarkGray
    Invoke-LightvcWithCapture -LaunchArgs (@('gui', '--demo') + $deviceFlags)
} else {
    Write-Step 'Launching LightVC GUI'
    Write-Host  '    (converter weights optional — load later via Realtime tab)' -ForegroundColor DarkGray
    & $binPath gui --dac-weights $dacPath @deviceFlags
}
exit $LASTEXITCODE

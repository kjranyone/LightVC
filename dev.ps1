<#
.SYNOPSIS
    LightVC development helper — build and launch the GUI.

.DESCRIPTION
    Builds the standalone app (release) and launches the GUI. Converter
    weights (front gate / K3G / rvoc3hi) are optional — the GUI starts
    without them; pick the operator WAV and the target voice on the
    Realtime tab and press Load.

    Demo mode (--Demo) renders with mock data and needs no weights or
    model, useful for layout review. Tabs are switchable inside the app.
    Use --Snap to capture the running window to PNG on demand.
#>
[CmdletBinding()]
param(
    [switch]$NoBuild,
    [switch]$BuildOnly,
    [switch]$Snap,
    [string]$Output,
    [switch]$Demo,
    [switch]$DebugBuild,
    [switch]$Release,
    [switch]$Watch,
    [switch]$Asio
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

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
    Write-Menu '  [1] GUI を起動（通常・dev-release 差分ビルド）'
    Write-Menu '  [2] GUI を起動（ビルドスキップ）'
    Write-Menu '  [3] デモ＋Watch（debug・.rs編集で自動リロード・AI レビュー用）'
    Write-Menu '  [4] ビルドのみ（dev-release・-Release で本番 LTO ビルド）'
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

$hasAction = $BuildOnly -or $Demo -or $Watch -or $Output
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
        $BuildOnly = $true
        $action = 'buildonly'
    } elseif ($choice -eq 'q' -or $choice -eq 'Q') {
        exit 0
    } else {
        Write-Err ('無効な選択: ' + $choice)
        exit 1
    }
}

# Watch mode: build + launch demo, then poll for .rs edits and auto-reload.
# Triggered by -Watch flag or menu [3]. Exits never (Ctrl+C to stop).
if ($Watch) {
    Invoke-WatchLoop
    exit 0
}

# Binary path depends on build profile.
$profileDir = if ($DebugBuild) { 'debug' } elseif ($Release) { 'release' } else { 'dev-release' }
$binPath    = Join-Path $repoRoot "target\$profileDir\lightvc-app.exe"
if ($NoBuild -and -not $DebugBuild -and -not $Release -and -not (Test-Path $binPath)) {
    $relBin = Join-Path $repoRoot 'target\release\lightvc-app.exe'
    if (Test-Path $relBin) { $binPath = $relBin }
}

# --- 1. Build ---------------------------------------------------------------
if (-not $NoBuild) {
    # gpui の Windows ビルドは fxc.exe(Windows SDK)を PATH に要求する。無ければ SDK の bin を足す。
    if (-not (Get-Command fxc.exe -ErrorAction SilentlyContinue)) {
        $sdkFxc = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\bin\*\x64\fxc.exe' -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1
        if ($sdkFxc) {
            $env:PATH = "$(Split-Path $sdkFxc.FullName);$env:PATH"
            Write-Host "    fxc: $($sdkFxc.FullName)" -ForegroundColor DarkGray
        } else {
            Write-Err 'fxc.exe が見つからない(gpui のビルドに Windows SDK が要る)。'
            exit 1
        }
    }
    $buildArgs = @('build', '-p', 'lightvc-app')
    if ($Release) {
        $buildArgs += '--release'
    } elseif (-not $DebugBuild) {
        $buildArgs += '--profile'
        $buildArgs += 'dev-release'
    }
    if ($Asio) {
        if ($env:CPAL_ASIO_DIR) {
            $buildArgs += '--features'
            $buildArgs += 'asio'
        } else {
            Write-Host '    ASIO: ネイティブ COM ASIO 経路が標準で有効です(CPAL_ASIO_DIR 不要)' -ForegroundColor DarkGray
        }
    }
    $profileLabel = if ($DebugBuild) { 'debug' } elseif ($Release) { 'release' } else { 'dev-release' }
    Write-Step "Building $profileLabel binary (cargo $($buildArgs -join ' '))"
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

# --- 2. Launch GUI ----------------------------------------------------------
if ($Demo) {
    Write-Step 'デモモードで起動（自動スクショ付き）'
    Write-Host  '    (mock data, no model required; tabs switchable inside the app)' -ForegroundColor DarkGray
    Invoke-LightvcWithCapture -LaunchArgs @('gui', '--demo')
} else {
    Write-Step 'Launching LightVC GUI'
    Write-Host  '    (converter weights optional — pick operator + target on the Realtime tab)' -ForegroundColor DarkGray
    & $binPath gui
}
exit $LASTEXITCODE

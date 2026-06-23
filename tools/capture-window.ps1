<#
.SYNOPSIS
    Capture a window screenshot for documentation / visual regression.

.DESCRIPTION
    Launches the LightVC GUI in --demo-state mode, waits for it to render,
    captures the window to PNG, then closes the app.

    Requires: the release binary built (cargo build --release -p lightvc-app)
    and Windows 10+ (uses Add-Type System.Drawing).

.PARAMETER DemoState
    Which tab/mode to capture: offline, realtime, or catalog.

.PARAMETER Out
    Output PNG path. Default: docs/screenshots/<state>.png

.PARAMETER WaitMs
    Milliseconds to wait after launch before capturing (lets the UI settle).

.EXAMPLE
    .\tools\capture-window.ps1 -DemoState realtime
    .\tools\capture-window.ps1 -DemoState offline -Out docs\screenshots\offline-full.png
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('offline', 'realtime', 'catalog')]
    [string]$DemoState,

    [string]$Out,

    [int]$WaitMs = 1500
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

if (-not $Out) {
    $Out = "docs\screenshots\$DemoState.png"
}
$outDir = Split-Path -Parent $Out
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

$bin = Join-Path $repoRoot 'target\release\lightvc-app.exe'
if (-not (Test-Path $bin)) {
    Write-Host 'Building release binary...' -ForegroundColor Cyan
    & cargo build --release -p lightvc-app
    if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
}

Write-Host "Launching lightvc-app --demo-state $DemoState ..." -ForegroundColor Cyan
$proc = Start-Process -FilePath $bin `
    -ArgumentList 'gui', '--demo-state', $DemoState `
    -PassThru

Start-Sleep -Milliseconds $WaitMs

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
if (-not ('LightvcCap2' -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class LightvcCap2 {
    [DllImport("user32.dll")]
    public static extern IntPtr FindWindow(string lpClassName, string lpWindowName);
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, int nFlags);
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@
}

# eframe 0.34 may delay setting the window title; retry for up to 5s.
$hwnd = [IntPtr]::Zero
for ($i = 0; $i -lt 10; $i++) {
    $hwnd = [LightvcCap2]::FindWindow($null, 'LightVC')
    if ($hwnd -ne [IntPtr]::Zero) { break }
    Start-Sleep -Milliseconds 500
}
if ($hwnd -eq [IntPtr]::Zero) {
    Write-Warning "Window 'LightVC' not found after 5s; saving full primary screen instead."
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
    $gfx = [System.Drawing.Graphics]::FromImage($bmp)
    $gfx.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bmp.Size)
    $gfx.Dispose()
} else {
    [LightvcCap2]::SetForegroundWindow($hwnd) | Out-Null
    Start-Sleep -Milliseconds 300
    $rect = New-Object LightvcCap2+RECT
    [LightvcCap2]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
    $w = $rect.Right - $rect.Left
    $h = $rect.Bottom - $rect.Top
    $bmp = New-Object System.Drawing.Bitmap $w, $h
    $gfx = [System.Drawing.Graphics]::FromImage($bmp)
    $hdc = $gfx.GetHdc()
    $ok = [LightvcCap2]::PrintWindow($hwnd, $hdc, 2)  # PW_RENDERFULLCONTENT
    $gfx.ReleaseHdc($hdc)
    $gfx.Dispose()
    if (-not $ok) {
        $bmp.Dispose()
        $bmp = New-Object System.Drawing.Bitmap $w, $h
        $gfx2 = [System.Drawing.Graphics]::FromImage($bmp)
        $gfx2.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bmp.Size)
        $gfx2.Dispose()
    }
}

$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()

Write-Host "Saved: $Out" -ForegroundColor Green

Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue

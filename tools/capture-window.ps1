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
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")]
    public static extern IntPtr FindWindow(string lpClassName, string lpWindowName);
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

$hwnd = [Win32]::FindWindow($null, 'LightVC')
if ($hwnd -eq [IntPtr]::Zero) {
    Write-Warning "Window 'LightVC' not found; saving full primary screen instead."
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
} else {
    [Win32]::SetForegroundWindow($hwnd) | Out-Null
    Start-Sleep -Milliseconds 200
    $rect = New-Object Win32+RECT
    [Win32]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
    $bounds = New-Object System.Drawing.Rectangle(
        $rect.Left, $rect.Top,
        $rect.Right - $rect.Left,
        $rect.Bottom - $rect.Top
    )
}

$bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$gfx.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bmp.Size)

$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$gfx.Dispose()
$bmp.Dispose()

Write-Host "Saved: $Out ($($bounds.Width)x$($bounds.Height))" -ForegroundColor Green

Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue

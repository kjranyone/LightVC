<#
.SYNOPSIS
    Capture the LightVC window screenshot (on-demand, app must be running).

.DESCRIPTION
    This script does NOT launch the app. It finds the running 'LightVC'
    window, brings it to the foreground, captures it to PNG.

    Intended workflow:
      1. User launches app:   .\dev.ps1   (or  .\dev.ps1 -Demo realtime)
      2. User interacts / positions the window
      3. AI asks for screenshot: this script (or .\dev.ps1 -Snap)
      4. User drops the PNG into the chat

.PARAMETER Out
    Output PNG path. Default: docs/screenshots/snap-<timestamp>.png

.PARAMETER WaitMs
    Milliseconds to wait after focusing the window before capture.

.EXAMPLE
    .\tools\snap.ps1
    .\tools\snap.ps1 -Out docs\screenshots/realtime-after-fix.png
#>
[CmdletBinding()]
param(
    [string]$Out,
    [int]$WaitMs = 300
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

if (-not $Out) {
    $ts = Get-Date -Format 'yyyyMMdd-HHmmss'
    $Out = "docs\screenshots\snap-$ts.png"
}
$outDir = Split-Path -Parent $Out
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
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
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

# Find the LightVC process and bring it to the foreground.
$proc = Get-Process -Name "lightvc-app" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $proc -or $proc.MainWindowHandle -eq [IntPtr]::Zero) {
    Write-Error "LightVC ウィンドウが見つかりません。アプリを起動してから実行してください。"
    exit 1
}
$hwnd = $proc.MainWindowHandle
[Win32]::ShowWindow($hwnd, 9) | Out-Null  # SW_RESTORE
[Win32]::SetForegroundWindow($hwnd) | Out-Null
Start-Sleep -Milliseconds $WaitMs

# Capture the full primary screen (window is now foregrounded).
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds

$bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$gfx.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bmp.Size)

$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$gfx.Dispose()
$bmp.Dispose()

Write-Host "保存しました: $Out ($($bounds.Width)x$($bounds.Height))" -ForegroundColor Green

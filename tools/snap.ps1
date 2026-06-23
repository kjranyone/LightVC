<#
.SYNOPSIS
    LightVC window screenshot (on-demand, app must be running).

.DESCRIPTION
    Finds the running 'lightvc-app' window, brings it to the foreground,
    and captures it via PrintWindow (hardware-acceleration-safe). BitBlt
    is used only as a fallback because it blacks out OpenGL/D3D content.

.PARAMETER Out
    Output PNG path. Default: docs/screenshots/snap-<timestamp>.png

.PARAMETER WaitMs
    Milliseconds to wait after focusing the window before capture.

.EXAMPLE
    .\tools\snap.ps1
    .\tools\snap.ps1 -Out docs\screenshots/realtime.png
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
if (-not ('LightvcCapture' -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class LightvcCapture {
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, int nFlags);
    [DllImport("dwmapi.dll")]
    public static extern int DwmGetWindowAttribute(IntPtr hwnd, int attr, out RECT pvAttribute, int cbAttribute);
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@
}

# Find the LightVC process and bring it to the foreground.
$proc = Get-Process -Name "lightvc-app" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $proc -or $proc.MainWindowHandle -eq [IntPtr]::Zero) {
    Write-Error "LightVC window not found. Launch the app first."
    exit 1
}
$hwnd = $proc.MainWindowHandle
[LightvcCapture]::ShowWindow($hwnd, 9) | Out-Null   # SW_RESTORE
[LightvcCapture]::SetForegroundWindow($hwnd) | Out-Null
Start-Sleep -Milliseconds $WaitMs

# DWM extended frame bounds (includes shadow); fall back to GetWindowRect.
$rect = New-Object LightvcCapture+RECT
$dwmAttr = 9  # DWMWA_EXTENDED_FRAME_BOUNDS
$hr = [LightvcCapture]::DwmGetWindowAttribute($hwnd, $dwmAttr, [ref]$rect, [System.Runtime.InteropServices.Marshal]::SizeOf($rect))
if ($hr -ne 0) {
    [void][LightvcCapture]::GetWindowRect($hwnd, [ref]$rect)
}

$w = $rect.Right  - $rect.Left
$h = $rect.Bottom - $rect.Top
if ($w -le 0 -or $h -le 0) {
    Write-Error "Invalid window size: ${w}x${h}"
    exit 1
}

# PrintWindow capture (PW_RENDERFULLCONTENT = 2 for Win8+, handles OpenGL/D3D).
$bmp = New-Object System.Drawing.Bitmap $w, $h
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $gfx.GetHdc()
$flags = 2  # PW_RENDERFULLCONTENT
$ok = [LightvcCapture]::PrintWindow($hwnd, $hdc, $flags)
$gfx.ReleaseHdc($hdc)
$gfx.Dispose()

if (-not $ok) {
    # Fallback: BitBlt (may black out OpenGL content).
    $bmp.Dispose()
    $bmp = New-Object System.Drawing.Bitmap $w, $h
    $gfx2 = [System.Drawing.Graphics]::FromImage($bmp)
    $gfx2.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bmp.Size)
    $gfx2.Dispose()
    Write-Host "  (PrintWindow failed, fell back to BitBlt)" -ForegroundColor Yellow
}

$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()

Write-Host "Saved: $Out (${w}x${h}, hwnd=$hwnd)" -ForegroundColor Green

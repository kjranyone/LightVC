<#
.SYNOPSIS
    Launch the demo GUI once per tab and capture docs/screenshots/<Prefix>_<tab>.png.
#>
param([string]$Prefix = 'm3e', [string[]]$Tabs = @('realtime', 'offline', 'voices', 'devices'), [string]$Theme = 'light', [string]$Bin = 'target\debug\lightvc-app.exe')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
if (Get-Process -Name 'lightvc-app' -ErrorAction SilentlyContinue) { Write-Error 'lightvc-app is already running; close it first.'; exit 1 }
foreach ($tab in $Tabs) {
    $env:LVC_TAB = $tab
    $env:LVC_THEME = $Theme
    $p = Start-Process -FilePath (Join-Path $root $Bin) -ArgumentList 'gui', '--demo' -PassThru
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 200
        $q = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
        if ($q -and $q.MainWindowHandle -ne [IntPtr]::Zero) { break }
    }
    Start-Sleep -Milliseconds 1500
    $suffix = if ($Theme -eq 'light') { '' } else { "_$Theme" }
    & (Join-Path $root 'tools\snap.ps1') -Out "docs\screenshots\${Prefix}_${tab}${suffix}.png"
    Stop-Process -Id $p.Id -Force
    Start-Sleep -Milliseconds 300
}

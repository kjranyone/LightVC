<#
.SYNOPSIS
    ASIO ビルドの事前検査(SDK は各自で Steinberg から入手・リポジトリに入れない)。

.DESCRIPTION
    cpal の asio 機能はビルド時に Steinberg の ASIO SDK(プロプライエタリ)と
    LLVM/clang(bindgen)を要求する。この脚本はその 3 つ(CPAL_ASIO_DIR・clang・fxc)
    を確かめ、足りない物に正確な手順を示す。全て揃っていれば
    cargo build --release -p lightvc-app --features asio が通る状態。

.EXAMPLE
    .\tools\setup-asio.ps1            # 検査だけ
    .\tools\setup-asio.ps1 -SetPath C:\sdk\asiosdk_2.3.3   # このセッションの CPAL_ASIO_DIR を設定
#>
[CmdletBinding()]
param(
    [string]$SetPath
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

function Ok($m)   { Write-Host "  [OK] $m" -ForegroundColor Green }
function Bad($m)  { Write-Host "  [x] $m" -ForegroundColor Red }

if ($SetPath) {
    [Environment]::SetEnvironmentVariable('CPAL_ASIO_DIR', $SetPath, 'User')
    $env:CPAL_ASIO_DIR = $SetPath
    Ok "CPAL_ASIO_DIR = $SetPath (ユーザー環境変数に保存)"
}

Write-Host "== ASIO ビルドの事前検査 ==" -ForegroundColor Cyan

# 1. ASIO SDK
$sdk = $env:CPAL_ASIO_DIR
if ($sdk -and (Test-Path (Join-Path $sdk 'common\asio.h'))) {
    Ok "ASIO SDK: $sdk"
} else {
    Bad "ASIO SDK が見つからない (CPAL_ASIO_DIR=$sdk)。"
    Write-Host "      1. https://www.steinberg.net から ASIO SDK をダウンロード(ライセンス同意が要る)" -ForegroundColor Yellow
    Write-Host "      2. 展開して .\tools\setup-asio.ps1 -SetPath <SDKの根> を実行" -ForegroundColor Yellow
    Write-Host "      3. SDK はリポジトリにコミットしない(再配布禁止)" -ForegroundColor Yellow
}

# 2. clang (bindgen)
if (Get-Command clang.exe -ErrorAction SilentlyContinue) {
    Ok "clang: $((Get-Command clang.exe).Source)"
} else {
    Bad 'clang が PATH に無い(bindgen に要る)。winget install LLVM.LLVM などで入れる。'
}

# 3. fxc (gpui の Windows ビルド)
if (Get-Command fxc.exe -ErrorAction SilentlyContinue) {
    Ok 'fxc: ある'
} else {
    $fxc = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\bin\*\x64\fxc.exe' -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
    if ($fxc) { Ok "fxc: $($fxc.FullName)(cargo には PATH 通しが要る・dev.ps1 は自動)" }
    else { Bad 'fxc.exe が無い(Windows SDK を入れる)' }
}

if (($sdk -and (Test-Path (Join-Path $sdk 'common\asio.h'))) -and (Get-Command clang.exe -ErrorAction SilentlyContinue)) {
    Write-Host ''
    Write-Host '  全て揃った。次で ASIO 入りアプリがビルドできる:' -ForegroundColor Green
    Write-Host '    .\dev.ps1 -Asio          # ビルドして起動(GUI の Devices で asio を選ぶ)' -ForegroundColor Green
    Write-Host '    .\target\release\lightvc-app.exe realtime-snv --host asio --buffer 240 ...' -ForegroundColor Green
}

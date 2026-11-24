<#
.SYNOPSIS
    同梱のボイスプリセット(voices/<id>.flac と voices/manifest.json)を Irodori-TTS で作る。

.DESCRIPTION
    声の元(キャラクターの caption・seed・読ませる文)は voice_presets.json に記録してある。
    各声は 2 つの発話を 0.6 秒の間でつないだ 1 本の FLAC(48 kHz・モノラル・20 秒以上)になる。
      1 本目 = caption と seed だけで作った素の声(実在の人物の参照は使わない)
      2 本目 = その 1 本目を参照音声にして別の文を読ませたもの(同じ声にするため)
    目標の参照は計 20 秒以上が目安なので、1 本だけで 20 秒を超えるようにする。

    要るもの: sttts-say(kjranyone/sttts-gui)と ffmpeg。GPU を使うので sttts-gui は閉じておく。
    sttts-say には一時フォルダをアプリのルート(STTTS_ROOT)として渡すので、利用者の設定や声バンクには触らない。
    声を替えたら NOTICE の「Bundled voice presets」も直す。

.EXAMPLE
    .\tools\make_voice_presets.ps1                       # 無いものだけ作る
    .\tools\make_voice_presets.ps1 -Only imouto -Force   # imouto だけ作り直す
#>
param([string[]]$Only, [switch]$Force)
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$outDir = Join-Path $root 'voices'
$say = Get-Command sttts-say -ErrorAction SilentlyContinue
if (-not $say) { throw 'sttts-say が見つかりません(kjranyone/sttts-gui の .\dev.ps1 say でビルドして PATH へ)' }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw 'ffmpeg が見つかりません' }
$work = Join-Path ([IO.Path]::GetTempPath()) 'lightvc-voice-presets'
New-Item -ItemType Directory -Force -Path $work, $outDir | Out-Null
$utf8 = [Text.UTF8Encoding]::new($false)
$spec = Get-Content (Join-Path $PSScriptRoot 'voice_presets.json') -Raw -Encoding utf8 | ConvertFrom-Json
$voices = @($spec.voices)
if ($Only) { $voices = @($voices | Where-Object { $_.id -in $Only }) }
if (-not $voices.Count) { throw '該当する声がありません' }

function Render($script, $dir, $what, $id) {
  & $say.Source render $script --out-dir $dir 2> (Join-Path $dir "$what.log") | Out-Null
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $dir "$what.wav"))) {
    throw "sttts-say failed: $id / $what ($((Get-Content (Join-Path $dir "$what.log") -Raw)))"
  }
}

# 呼び出し元の環境(アプリのルート)を汚さないよう、終わったら戻す
$savedRoot = [Environment]::GetEnvironmentVariable('STTTS_ROOT', 'Process')
$env:STTTS_ROOT = $work
try {
  # モデルの切替(読み込み直し)を最少にするため、モデルごとにまとめる
  foreach ($group in ($voices | Group-Object model)) {
    & $say.Source model use $group.Name | Out-Null
    foreach ($v in $group.Group) {
      $flac = Join-Path $outDir "$($v.id).flac"
      if ((Test-Path $flac) -and -not $Force) { "skip (exists): $($v.id)"; continue }
      $dir = Join-Path (Join-Path $work 'takes') $v.id
      New-Item -ItemType Directory -Force -Path $dir | Out-Null
      # 指定は台本(JSONL)で渡す(引数に日本語を並べない)
      $first = [ordered]@{ id = 'first'; text = $v.clips[0].text; caption = $v.caption; seed = [long]$v.clips[0].seed }
      [IO.File]::WriteAllText((Join-Path $dir 'first.jsonl'), (($first | ConvertTo-Json -Compress) + "`n"), $utf8)
      Render (Join-Path $dir 'first.jsonl') $dir 'first' $v.id
      # 1 本目を sttts-say の声として登録し、2 本目はそれを参照に読ませる
      $voice = "ref_$($v.id)"
      if (-not (Test-Path (Join-Path $work "data\voices\$voice.wav"))) {
        & $say.Source voice save $voice --from (Join-Path $dir 'first.wav') 2> (Join-Path $dir 'save.log') | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "sttts-say voice save failed: $v.id" }
      }
      $second = [ordered]@{ id = 'second'; text = $v.clips[1].text; voice = $voice; seed = [long]$v.clips[1].seed }
      [IO.File]::WriteAllText((Join-Path $dir 'second.jsonl'), (($second | ConvertTo-Json -Compress) + "`n"), $utf8)
      Render (Join-Path $dir 'second.jsonl') $dir 'second' $v.id
      # 2 本を 0.6 秒の無音ではさんで 1 本にする
      & ffmpeg -loglevel error -y -i (Join-Path $dir 'first.wav') -i (Join-Path $dir 'second.wav') `
        -filter_complex '[0:a]apad=pad_dur=0.6[a0];[a0][1:a]concat=n=2:v=0:a=1[out]' -map '[out]' `
        -ar 48000 -ac 1 -c:a flac -compression_level 12 -sample_fmt s16 $flac
      if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed: $($v.id)" }
      "$($v.id): $($v.model)"
    }
  }
} finally {
  [Environment]::SetEnvironmentVariable('STTTS_ROOT', $savedRoot, 'Process')
  & $say.Source stop | Out-Null
}

# 一覧は常に仕様から書き出す(声と一覧がずれないように)
$manifest = [ordered]@{
  version = 1
  voices = @($spec.voices | ForEach-Object { [ordered]@{ id = $_.id; name = $_.name; gender = $_.gender; file = "$($_.id).flac"; model = $_.model } })
}
[IO.File]::WriteAllText((Join-Path $outDir 'manifest.json'), ((($manifest | ConvertTo-Json -Depth 5) -replace "`r`n", "`n") + "`n"), $utf8)
"manifest: $(@($spec.voices).Count) voices -> $(Join-Path $outDir 'manifest.json')"

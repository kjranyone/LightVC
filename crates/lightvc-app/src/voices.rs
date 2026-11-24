//! 同梱のボイスプリセット(目標の声の参照)。Irodori-TTS で作った架空の声で、実在の人物の録音ではない。
//!
//! - 場所は `voices/`(`LVC_VOICES` → カレントフォルダ → 実行ファイルの隣 → 開発中はリポジトリ)。
//!   `voices/manifest.json` に一覧(並び = 表示の並び)。
//! - 中身は `tools/make_voice_presets.ps1` が `tools/voice_presets.json` から作る(各声 1 本の FLAC・48 kHz・20 秒以上)。
//! - 声を替えたら `NOTICE` の「Bundled voice presets」も直す。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// 声の区分。同梱のプリセットは女性・男性、自分で追加した声は「自分」。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceGroup {
    Female,
    Male,
    Own,
}

impl VoiceGroup {
    pub fn label(self) -> &'static str {
        match self {
            Self::Female => "女性",
            Self::Male => "男性",
            Self::Own => "自分",
        }
    }

    fn from_manifest(s: &str) -> Result<Self> {
        match s {
            "female" => Ok(Self::Female),
            "male" => Ok(Self::Male),
            other => anyhow::bail!("voices/manifest.json: 未知の区分 {other:?}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VoiceEntry {
    pub name: String,
    /// 参照の音声(48 kHz の WAV / FLAC)
    pub path: PathBuf,
    pub group: VoiceGroup,
}

/// プリセットのあるフォルダ(`manifest.json` を持つ最初の候補)。
pub fn voices_dir() -> Option<PathBuf> {
    let has = |p: PathBuf| p.join("manifest.json").is_file().then_some(p);
    if let Some(p) = std::env::var_os("LVC_VOICES") {
        return has(PathBuf::from(p));
    }
    has(PathBuf::from("voices"))
        .or_else(|| std::env::current_exe().ok().and_then(|e| e.parent().map(|d| has(d.join("voices")))).flatten())
        .or_else(|| has(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../voices")))
}

/// 一覧(`voices/manifest.json`の順)。ファイルが欠けている声はエラーにせず除き、標準エラーへ知らせる。
fn read_manifest(dir: &Path) -> Result<Vec<VoiceEntry>> {
    let path = dir.join("manifest.json");
    let root: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).with_context(|| path.display().to_string())?)
        .with_context(|| path.display().to_string())?;
    let list = root.get("voices").and_then(|v| v.as_array()).context("voices/manifest.json: voices が無い")?;
    let field = |v: &serde_json::Value, k: &str| {
        v.get(k).and_then(|x| x.as_str()).filter(|s| !s.is_empty()).map(str::to_string).with_context(|| format!("voices/manifest.json: {k} が無い"))
    };
    let mut out = Vec::new();
    for v in list {
        let audio = dir.join(field(v, "file")?);
        if !audio.is_file() {
            eprintln!("voices: 参照の音声が無い: {}", audio.display());
            continue;
        }
        out.push(VoiceEntry { name: field(v, "name")?, path: audio, group: VoiceGroup::from_manifest(&field(v, "gender")?)? });
    }
    Ok(out)
}

/// 同梱のプリセットの一覧。場所が無ければ空(利用者の声だけで動く)。
pub fn bundled_voices() -> Vec<VoiceEntry> {
    let Some(dir) = voices_dir() else { return Vec::new() };
    match read_manifest(&dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("voices: {e:#}");
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 声と一覧がそろっていて、どの声も 48 kHz・20 秒以上で音があるか(`voices/` が無い環境では読み飛ばす)。
    #[test]
    fn bundled_voices_are_complete_and_audible() {
        let Some(dir) = voices_dir() else {
            eprintln!("skip: voices/ が無い");
            return;
        };
        let voices = read_manifest(&dir).unwrap();
        assert_eq!(voices.iter().filter(|v| v.group == VoiceGroup::Female).count(), 15);
        assert_eq!(voices.iter().filter(|v| v.group == VoiceGroup::Male).count(), 15);
        let mut names: Vec<&str> = voices.iter().map(|v| v.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), voices.len(), "表示名が重複している");
        for v in &voices {
            let (x, sr) = crate::audio_playback::load_audio_mono(&v.path).unwrap_or_else(|e| panic!("{}: {e:#}", v.path.display()));
            assert_eq!(sr, 48_000, "{}", v.name);
            assert!(x.len() >= 20 * 48_000, "{}: {:.1} 秒", v.name, x.len() as f32 / 48_000.0);
            let peak = x.iter().fold(0f32, |a, s| a.max(s.abs()));
            assert!((0.05..=1.0).contains(&peak), "{}: peak {peak}", v.name);
        }
    }
}

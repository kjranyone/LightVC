//! 話者登録: 操作者(あなた)の普通の話し声を録り、変換の基準にする声の特徴の登録に加える。
//!
//! 窓(モーダル)が開いている間は入力装置を開いたまま見張り(波形・レベル)、録音の合図で SECONDS 秒を録って
//! 48 kHz の WAV に保存し、`settings.operator` に加える。窓を閉じると(`Session` を落とすと)装置を閉じる。

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use lightvc_audio::{AudioEndpoint, AudioEngine};

use crate::state::AppState;

/// 1 回の録音の長さ(秒)。
pub const SECONDS: u32 = 30;
/// 波形の 1 列の長さ(秒)。
pub const COL_S: f32 = 0.02;
/// 見張りの波形に出す列の数(3 秒)。
pub const SCOPE_COLS: usize = 150;
/// 声とみなす 1 列の RMS(dBFS)。
pub const VOICE_DB: f32 = -45.0;
/// レベルの数値の窓(300 ms)。
const LEVEL_COLS: usize = 15;
/// レベルの判定の窓(見張りの波形と同じ 3 秒の中で最も大きい列・話の切れ目で判定が揺れないように)。
const JUDGE_COLS: usize = SCOPE_COLS;
/// 音割れの表示を保つ列の数(1 秒)。
const CLIP_HOLD: usize = 50;

#[derive(Clone, PartialEq)]
pub enum Phase {
    Opening,
    Monitoring,
    Recording,
    Saving,
    /// 装置を開けない・切断された(糸は終わる)。
    Failed(String),
}

/// 入力の様子(録音の糸が書き、画面が読む)。
#[derive(Clone)]
pub struct Live {
    pub phase: Phase,
    pub device: String,
    pub host: String,
    pub sample_rate: u32,
    /// 直近 3 秒の列ごとの (最小, 最大)。
    pub scope: VecDeque<(f32, f32)>,
    /// 録音中の列ごとの (最小, 最大)。
    pub take: Vec<(f32, f32)>,
    /// 直近 300 ms の RMS とピーク(dBFS)。
    pub rms_db: f32,
    pub peak_db: f32,
    /// 直近 3 秒で最も大きい列の RMS(dBFS)。大きさの判定に使う。
    pub voice_db: f32,
    /// 直近 1 秒に音割れ(|x| ≥ 0.98)があった。
    pub clip: bool,
    /// 録音した秒数と、そのうち声が入っていた秒数。
    pub recorded: f32,
    pub voiced: f32,
    /// 直前の録音の結果(保存した場所・声が入っていた秒数)。
    pub saved: Option<(PathBuf, f32)>,
    pub save_error: Option<String>,
}

impl Default for Live {
    fn default() -> Self {
        Self {
            phase: Phase::Opening,
            device: String::new(),
            host: String::new(),
            sample_rate: 0,
            scope: VecDeque::with_capacity(SCOPE_COLS),
            take: Vec::new(),
            rms_db: -120.0,
            peak_db: -120.0,
            voice_db: -120.0,
            clip: false,
            recorded: 0.0,
            voiced: 0.0,
            saved: None,
            save_error: None,
        }
    }
}

/// 開いている話者登録。落とすと入力装置を閉じる。
pub struct Session {
    live: Arc<Mutex<Live>>,
    record: Arc<AtomicBool>,
    abort: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl Session {
    pub fn live(&self) -> Live {
        self.live.lock().unwrap().clone()
    }

    pub fn start_take(&self) {
        self.abort.store(false, Ordering::Relaxed);
        self.record.store(true, Ordering::Relaxed);
    }

    pub fn abort_take(&self) {
        self.record.store(false, Ordering::Relaxed);
        self.abort.store(true, Ordering::Relaxed);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// 選んでいる入力装置(未選択なら既定)を開いて見張りを始める。
pub fn open(state: Arc<Mutex<AppState>>) -> Session {
    let ep = state.lock().unwrap().input_endpoint();
    let live = Arc::new(Mutex::new(Live::default()));
    let (record, abort, stop) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let session = Session { live: live.clone(), record: record.clone(), abort: abort.clone(), stop: stop.clone() };
    std::thread::spawn(move || run(state, ep, live, record, abort, stop));
    session
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-6).log10()
}

fn run(state: Arc<Mutex<AppState>>, ep: AudioEndpoint, live: Arc<Mutex<Live>>, record: Arc<AtomicBool>, abort: Arc<AtomicBool>, stop: Arc<AtomicBool>) {
    let fail = |msg: String| live.lock().unwrap().phase = Phase::Failed(msg);
    let mut device = lightvc_audio::engine::input_device_name(&ep).unwrap_or_else(|_| "既定の入力装置".to_string());
    if let Some(ch) = lightvc_audio::engine::input_channel_label(&ep) {
        device = format!("{device}・{ch}");
    }
    let (engine, mut rx) = match AudioEngine::start_capture_only(ep, 48_000, 240) {
        Ok(v) => v,
        Err(e) => return fail(format!("入力装置「{device}」を開けません: {e}")),
    };
    let sr = engine.capture_sample_rate as usize;
    {
        let mut l = live.lock().unwrap();
        l.device = device;
        l.host = engine.host_label.to_uppercase();
        l.sample_rate = sr as u32;
        l.phase = Phase::Monitoring;
    }
    let col_n = ((sr as f32 * COL_S) as usize).max(1);
    let want = SECONDS as usize * sr;
    let mut col: Vec<f32> = Vec::with_capacity(col_n);
    let mut take: Vec<f32> = Vec::new();
    let mut levels: VecDeque<(f32, f32)> = VecDeque::with_capacity(JUDGE_COLS);
    let mut clip_hold = 0usize;
    let mut rec = false;
    loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        if engine.is_disconnected() {
            return fail("入力装置が切断されました".to_string());
        }
        while let Ok(x) = rx.pop() {
            col.push(x);
            if col.len() < col_n {
                continue;
            }
            let (mn, mx) = col.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let ms = col.iter().map(|v| v * v).sum::<f32>() / col.len() as f32;
            let pk = mn.abs().max(mx.abs());
            levels.push_back((ms, pk));
            if levels.len() > JUDGE_COLS {
                levels.pop_front();
            }
            clip_hold = if pk >= 0.98 { CLIP_HOLD } else { clip_hold.saturating_sub(1) };
            if record.swap(false, Ordering::Relaxed) && !rec {
                rec = true;
                take.clear();
                take.reserve(want);
                let mut l = live.lock().unwrap();
                l.take.clear();
                (l.recorded, l.voiced, l.saved, l.save_error) = (0.0, 0.0, None, None);
                l.phase = Phase::Recording;
            }
            if abort.swap(false, Ordering::Relaxed) && rec {
                rec = false;
                take.clear();
                let mut l = live.lock().unwrap();
                l.take.clear();
                l.phase = Phase::Monitoring;
            }
            let recent = levels.iter().rev().take(LEVEL_COLS);
            let (sum, peak, n) = recent.fold((0.0f32, 0.0f32, 0usize), |(s, p, n), &(m, k)| (s + m, p.max(k), n + 1));
            let judge = levels.iter().map(|&(m, _)| m).fold(0.0f32, f32::max);
            {
                let mut l = live.lock().unwrap();
                l.scope.push_back((mn, mx));
                if l.scope.len() > SCOPE_COLS {
                    l.scope.pop_front();
                }
                l.rms_db = db((sum / n.max(1) as f32).sqrt());
                l.peak_db = db(peak);
                l.voice_db = db(judge.sqrt());
                l.clip = clip_hold > 0;
                if rec {
                    take.extend_from_slice(&col);
                    l.take.push((mn, mx));
                    l.recorded = take.len() as f32 / sr as f32;
                    if db(ms.sqrt()) > VOICE_DB {
                        l.voiced += COL_S;
                    }
                }
            }
            col.clear();
            if rec && take.len() >= want {
                rec = false;
                take.truncate(want);
                let voiced = {
                    let mut l = live.lock().unwrap();
                    l.phase = Phase::Saving;
                    l.voiced
                };
                let r = save(&take, sr as u32);
                if let Ok(ref p) = r {
                    state.lock().unwrap().settings.operator.push(p.clone());
                }
                let mut l = live.lock().unwrap();
                match r {
                    Ok(p) => l.saved = Some((p, voiced)),
                    Err(e) => l.save_error = Some(format!("保存に失敗しました: {e}")),
                }
                l.phase = Phase::Monitoring;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

fn save(x: &[f32], sr: u32) -> anyhow::Result<PathBuf> {
    let dir = std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_default().join("LightVC").join("recordings");
    std::fs::create_dir_all(&dir)?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let path = dir.join(format!("operator_{stamp}.wav"));
    let x48 = crate::cli::resample_to_48k(x, sr)?;
    crate::cli::save_wav_mono(&path, &x48, 48_000)?;
    Ok(path)
}

/// 登録した音声の長さ(秒)。WAV は見出しだけを読み、ほかは復号して測る。
pub fn duration_s(path: &Path) -> Option<f32> {
    if let Ok(r) = hound::WavReader::open(path) {
        let spec = r.spec();
        return Some(r.duration() as f32 / spec.sample_rate.max(1) as f32);
    }
    crate::audio_playback::load_audio_mono(path).ok().map(|(x, sr)| x.len() as f32 / sr.max(1) as f32)
}

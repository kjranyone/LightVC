//! アプリ共有の状態と制御(UI 非依存・GPUI と CLI の両方から使う)。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{unbounded, Receiver, Sender};
use lightvc_audio::{AudioEndpoint, HostPref};
use lightvc_core::snv_stream::SnvStream;

use crate::voices::VoiceEntry;

/// Real-time metrics from inference thread.
#[derive(Clone, Debug, Default)]
pub struct RtMetrics {
    pub input_rms: f32,
    pub output_rms: f32,
    pub latency_ms: f32,
    pub rtf: f32,
    /// True when the audio device was lost or failed to open.
    pub disconnected: bool,
    pub error: Option<String>,
    pub overrun: u64,
    pub underrun: u64,
    /// 直前の 5 秒の窓の 1 ブロックの計算時間 ms(realtime-snv と同じ定義)。
    pub compute_p50: f32,
    pub compute_p95: f32,
    pub compute_p99: f32,
    /// 窓の中のブロック数。
    pub blocks: u32,
    /// 直前のブロックで門が開いていたか。
    pub gate_open: bool,
    /// 使っているホストの名前(wasapi / asio / wasapi+asio)。
    pub host: String,
    pub capture_buffer_frames: u32,
    pub playback_buffer_frames: u32,
    pub capture_sample_rate: u32,
    pub playback_sample_rate: u32,
}

/// Control messages from UI to the real-time inference thread.
pub enum RtControl {
    StartWithDevices {
        input: AudioEndpoint,
        output: AudioEndpoint,
    },
    /// 読み込みと登録を済ませた変換器(重い処理は別の糸で・音声の経路にロックを置かない)。
    Stream(Box<SnvStream>),
    Stop,
    Bypass(bool),
    /// Mute the output (silence) while keeping the stream armed.
    Mute(bool),
    /// Fixed capture/playback buffer size in frames. Applied to the next
    /// engine Start; the UI re-arms a running stream so it takes effect.
    SetBufferSize(u32),
    /// 固定の半音(−12〜+12)。
    SetSemitones(f32),
    /// 入力のノイズ除去(変換の前)。
    SetDenoise(crate::denoise::Denoise),
}

/// SnvStream の部品の書き出し(training/export_{pfront,snv_model,rvoc}.py)と登録。
#[derive(Clone)]
pub struct SnvSettings {
    pub front: String,
    pub model: String,
    pub voc: String,
    /// 操作者の登録(自分の普通の発話の WAV・複数可)
    pub operator: Vec<PathBuf>,
    pub threads: usize,
    pub semitones: f32,
}

/// 製品の書き出し(配布物の models/ の名前, 作業ツリーの results/ の名前)。
pub const EXPORTS: [(&str, &str); 3] = [
    ("front_gate", "snv_p126_gate_export"),
    ("k3g", "snv_p126_k3g16kv_export"),
    ("rvoc3hi", "snv_p123_s/export_20k"),
];

/// 書き出しの既定の場所: 実行形式の隣(と 3 つ上まで)の models/ → %APPDATA%\LightVC\models → カレントの results/ → 実行形式の上の results/。見つからなければ results/ の相対パス。
pub fn resolve_export(sub: &str, dev: &str) -> String {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let mut d = exe.parent().map(Path::to_path_buf);
        for _ in 0..4 {
            let Some(p) = d else { break };
            d = p.parent().map(Path::to_path_buf);
            roots.push(p);
        }
    }
    let app = std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("LightVC"));
    let cwd = std::env::current_dir().unwrap_or_default();
    let models = roots.iter().chain(app.iter()).map(|r| r.join("models").join(sub));
    let results = std::iter::once(&cwd).chain(roots.iter()).map(|r| r.join("results").join(dev));
    models
        .chain(results)
        .find(|d| d.join("manifest.json").is_file())
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| format!("results/{dev}"))
}

impl Default for SnvSettings {
    fn default() -> Self {
        let [f, m, v] = EXPORTS.map(|(sub, dev)| resolve_export(sub, dev));
        Self {
            front: f,
            model: m,
            voc: v,
            operator: Vec::new(),
            threads: 4,
            semitones: 0.0,
        }
    }
}

/// Application-wide shared state.
pub struct AppState {
    pub model_loaded: bool,
    pub loading: bool,
    pub settings: SnvSettings,
    /// 読み込んだ経路のアルゴリズム遅延(標本・48 kHz。未読み込みは製品の既定)。
    pub algo_latency_samples: usize,
    pub voices: Vec<VoiceEntry>,
    pub selected_voice: Option<usize>,
    pub error: Option<String>,
    pub status: String,
    pub offline_result: Option<Vec<f32>>,
    /// ファイル変換の糸が終わった(成功・中止・失敗)。画面が拾って戻す。
    pub offline_done: bool,
    /// ファイル変換の速さの覚え(残り時間の見積もり)。
    pub offline_rates: crate::offline_tab::Rates,
    pub rt_control_tx: Option<Sender<RtControl>>,
    pub rt_metrics_rx: Option<Receiver<RtMetrics>>,
    pub rt_initialized: bool,
    /// App-wide audio device selection (None = system default).
    pub selected_input_host: HostPref,
    pub selected_input: Option<usize>,
    pub selected_input_name: Option<String>,
    /// 入力・出力のチャンネル(0 始まり・空 = 既定・AudioEndpoint::channels)。装置を替えると空に戻す。
    pub selected_input_channels: Vec<u16>,
    pub selected_output_channels: Vec<u16>,
    /// 入力のノイズ除去(リアルタイムの変換の前)。
    pub denoise: crate::denoise::Denoise,
    pub selected_output_host: HostPref,
    pub selected_output: Option<usize>,
    pub selected_output_name: Option<String>,
    /// Devices tab filter (Default = all hosts, Wasapi, Asio).
    pub selected_host: HostPref,
}

impl AppState {
    /// 選んでいる入力(装置とチャンネル)。
    pub fn input_endpoint(&self) -> AudioEndpoint {
        AudioEndpoint {
            host: self.selected_input_host,
            device_index: self.selected_input,
            device_name: self.selected_input_name.clone(),
            channels: self.selected_input_channels.clone(),
        }
    }

    /// 選んでいる出力(装置とチャンネル)。
    pub fn output_endpoint(&self) -> AudioEndpoint {
        AudioEndpoint {
            host: self.selected_output_host,
            device_index: self.selected_output,
            device_name: self.selected_output_name.clone(),
            channels: self.selected_output_channels.clone(),
        }
    }

    pub fn new() -> Self {
        // 同梱のプリセットを最初から選べるようにして、先頭の声を選んでおく
        let voices = crate::voices::bundled_voices();
        let selected_voice = voices.first().map(|_| 0);
        Self {
            model_loaded: false,
            loading: false,
            settings: SnvSettings::default(),
            algo_latency_samples: SnvStream::LATENCY_SAMPLES,
            voices,
            selected_voice,
            error: None,
            status: "準備完了".to_string(),
            offline_result: None,
            offline_done: false,
            offline_rates: crate::offline_tab::Rates::default(),
            rt_control_tx: None,
            rt_metrics_rx: None,
            rt_initialized: false,
            selected_input_host: HostPref::Default,
            selected_input: None,
            selected_input_name: None,
            selected_input_channels: Vec::new(),
            selected_output_channels: Vec::new(),
            denoise: crate::denoise::Denoise::Off,
            selected_output_host: HostPref::Default,
            selected_output: None,
            selected_output_name: None,
            selected_host: HostPref::Default,
        }
    }
}

/// WAV / FLAC を読み、48 kHz のモノラルにする。
pub fn load_48k(path: &Path) -> anyhow::Result<Vec<f32>> {
    let (x, sr) = crate::audio_playback::load_audio_mono(path)?;
    crate::cli::resample_to_48k(&x, sr)
}

/// リアルタイムの変換器: 話者登録(cfg.operator)と目標の参照から SnvStream を組む(読み込み + 登録 + ウォームアップ)。
pub fn build_stream(cfg: &SnvSettings, target: &[PathBuf]) -> anyhow::Result<SnvStream> {
    anyhow::ensure!(!cfg.operator.is_empty(), "話者登録がありません(リアルタイムの画面の「話者登録」で録音します)");
    let op = cfg.operator.iter().map(|p| load_48k(p)).collect::<anyhow::Result<Vec<_>>>()?;
    build_stream_with(cfg, &op, target)
}

/// 話し手の音声(48 kHz)を直に渡して組む。ファイル変換は変換元そのものを渡す(話者登録は使わない)。
pub fn build_stream_with(cfg: &SnvSettings, operator: &[Vec<f32>], target: &[PathBuf]) -> anyhow::Result<SnvStream> {
    anyhow::ensure!(!target.is_empty(), "目標の声がありません(Voices で選ぶ)");
    let mut s = SnvStream::load(Path::new(&cfg.front), Path::new(&cfg.model), Path::new(&cfg.voc))?;
    if cfg.threads > 1 {
        s.enable_parallel(cfg.threads);
    }
    s.set_semitones(cfg.semitones);
    s.enroll_operator(&operator.iter().map(|x| x.as_slice()).collect::<Vec<_>>())?;
    let tg = target.iter().map(|p| load_48k(p)).collect::<anyhow::Result<Vec<_>>>()?;
    s.enroll_target(&tg.iter().map(|x| x.as_slice()).collect::<Vec<_>>())?;
    {
        // 重みのページインと常駐スレッドの起動を済ませる(開始直後の無音の窓を防ぐ・状態は 5 ms の無音ぶんだけ先行する)。
        let warm = vec![0f32; SnvStream::BLOCK];
        let mut sink = vec![0f32; SnvStream::BLOCK];
        s.process_block(&warm, &mut sink);
    }
    Ok(s)
}

pub fn ensure_rt_thread(state: &Arc<Mutex<AppState>>) {
    let mut s = state.lock().unwrap();
    if s.rt_initialized {
        return;
    }
    let (control_tx, control_rx) = unbounded();
    let (metrics_tx, metrics_rx) = unbounded();
    let _ = control_tx.send(RtControl::SetDenoise(s.denoise));
    s.rt_control_tx = Some(control_tx);
    s.rt_metrics_rx = Some(metrics_rx);
    s.rt_initialized = true;
    drop(s);
    // The inference thread runs even without an engine (bypass), so the
    // audio path can be tested before the model is loaded.
    std::thread::spawn(move || {
        crate::realtime_tab::inference_loop(control_rx, metrics_tx);
    });
}

/// 選んだ目標の声で SnvStream を組み直し、音声の糸へ渡す(重い処理は別の糸で・音は止めない・ロックは音声の経路に置かない)。
pub fn load_engine(state: &Arc<Mutex<AppState>>) {
    ensure_rt_thread(state);
    let (cfg, target) = {
        let mut s = state.lock().unwrap();
        if s.loading {
            return;
        }
        let target: Vec<PathBuf> = s.selected_voice.and_then(|i| s.voices.get(i)).map(|v| vec![v.path.clone()]).unwrap_or_default();
        s.loading = true;
        s.status = "モデルを読み込み、登録しています…".to_string();
        s.error = None;
        (s.settings.clone(), target)
    };
    let st = state.clone();
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        let r = build_stream(&cfg, &target);
        let mut s = st.lock().unwrap();
        s.loading = false;
        match r {
            Ok(engine) => {
                let lat = engine.latency_samples();
                let sent = s.rt_control_tx.as_ref().map(|tx| tx.send(RtControl::Stream(Box::new(engine))).is_ok()).unwrap_or(false);
                if sent {
                    s.model_loaded = true;
                    s.algo_latency_samples = lat;
                    s.status = format!("モデルを読み込みました(登録 {:.1} 秒)", t0.elapsed().as_secs_f64());
                } else {
                    s.error = Some("読み込みに失敗: リアルタイムの処理が止まっています".to_string());
                }
            }
            Err(e) => s.error = Some(format!("読み込みに失敗: {e}")),
        }
    });
}

pub fn send_control(state: &Arc<Mutex<AppState>>, ctrl: RtControl) {
    let s = state.lock().unwrap();
    if let Some(ref tx) = s.rt_control_tx {
        let _ = tx.send(ctrl);
    }
}

pub fn start_msg(state: &Arc<Mutex<AppState>>) -> RtControl {
    let s = state.lock().unwrap();
    RtControl::StartWithDevices {
        input: s.input_endpoint(),
        output: s.output_endpoint(),
    }
}

//! アプリ共有の状態と制御(UI 非依存・GPUI と CLI の両方から使う)。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{unbounded, Receiver, Sender};
use lightvc_audio::HostPref;
use lightvc_core::snv_stream::SnvStream;

/// Real-time metrics from inference thread.
#[derive(Clone, Debug, Default)]
pub struct RtMetrics {
    pub input_rms: f32,
    pub output_rms: f32,
    pub latency_ms: f32,
    pub rtf: f32,
    /// True when the audio device was lost. The UI should stop and return to
    /// the device-selection screen.
    pub disconnected: bool,
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
    /// 使っているホストの名前(default / wasapi / asio)。
    pub host: String,
}

/// Control messages from UI to the real-time inference thread.
pub enum RtControl {
    StartWithDevices {
        host: HostPref,
        input_idx: Option<usize>,
        output_idx: Option<usize>,
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
    /// 固定の半音(運用 = 0 / 5 / 7 / 12)。
    SetSemitones(f32),
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

impl Default for SnvSettings {
    fn default() -> Self {
        Self {
            front: "results/snv_p45_gate_t80_export".into(),
            model: "results/snv_k3g16kv_export".into(),
            voc: "results/rvoc3hi/export_330k".into(),
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
    pub voices: Vec<VoiceEntry>,
    pub selected_voice: Option<usize>,
    pub error: Option<String>,
    pub status: String,
    pub offline_result: Option<Vec<f32>>,
    pub rt_control_tx: Option<Sender<RtControl>>,
    pub rt_metrics_rx: Option<Receiver<RtMetrics>>,
    pub rt_initialized: bool,
    /// 進行中の操作者の録音(無し = 通常)。
    pub operator_rec: Option<OperatorRecording>,
    /// App-wide audio device selection (None = system default).
    pub selected_input: Option<usize>,
    pub selected_output: Option<usize>,
    /// App-wide audio host selection (default / wasapi / asio).
    pub selected_host: HostPref,
}

#[derive(Clone)]
pub struct VoiceEntry {
    pub name: String,
    pub path: std::path::PathBuf,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            model_loaded: false,
            loading: false,
            settings: SnvSettings::default(),
            voices: Vec::new(),
            selected_voice: None,
            error: None,
            status: "Ready".to_string(),
            offline_result: None,
            rt_control_tx: None,
            rt_metrics_rx: None,
            rt_initialized: false,
            operator_rec: None,
            selected_input: None,
            selected_output: None,
            selected_host: HostPref::Default,
        }
    }
}

/// 進行中の操作者の録音(GUI からマイク録音・30 秒ほど)。
pub struct OperatorRecording {
    pub started: std::time::Instant,
    pub seconds: u32,
    pub path: PathBuf,
    /// キャンセルの合図(録音の糸が每周確認する)。
    pub cancel: Arc<std::sync::atomic::AtomicBool>,
}

/// WAV を読み、48 kHz のモノラルにする。
pub fn load_48k(path: &Path) -> anyhow::Result<Vec<f32>> {
    let (x, sr) = crate::audio_playback::load_wav_mono(path)?;
    crate::cli::resample_to_48k(&x, sr)
}

/// 設定と目標の参照から SnvStream を組む(読み込み + 登録 + ウォームアップ)。
pub fn build_stream(cfg: &SnvSettings, target: &[PathBuf]) -> anyhow::Result<SnvStream> {
    anyhow::ensure!(!cfg.operator.is_empty(), "操作者の登録の WAV がありません(Realtime タブの Operator)");
    anyhow::ensure!(!target.is_empty(), "目標の声がありません(Voices で選ぶ)");
    let mut s = SnvStream::load(Path::new(&cfg.front), Path::new(&cfg.model), Path::new(&cfg.voc))?;
    if cfg.threads > 1 {
        s.enable_parallel(cfg.threads);
    }
    s.set_semitones(cfg.semitones);
    let op = cfg.operator.iter().map(|p| load_48k(p)).collect::<anyhow::Result<Vec<_>>>()?;
    s.enroll_operator(&op.iter().map(|x| x.as_slice()).collect::<Vec<_>>())?;
    let tg = target.iter().map(|p| load_48k(p)).collect::<anyhow::Result<Vec<_>>>()?;
    s.enroll_target(&tg.iter().map(|x| x.as_slice()).collect::<Vec<_>>())?;
    {
        // 重みのページインと常駐スレッドの起動を済ませる(開始直後の無音の窓を防ぐ・状態は 5 ms の無音ぶんだけ先行する)。
        let mut warm = vec![0f32; SnvStream::BLOCK];
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
        s.status = "Loading model and enrolling...".to_string();
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
                let sent = s.rt_control_tx.as_ref().map(|tx| tx.send(RtControl::Stream(Box::new(engine))).is_ok()).unwrap_or(false);
                if sent {
                    s.model_loaded = true;
                    s.status = format!("Model loaded・enrolled in {:.1} s", t0.elapsed().as_secs_f64());
                } else {
                    s.error = Some("Load failed: realtime thread is gone".to_string());
                }
            }
            Err(e) => s.error = Some(format!("Load failed: {e}")),
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
    RtControl::StartWithDevices { host: s.selected_host, input_idx: s.selected_input, output_idx: s.selected_output }
}

/// 操作者の発話を既定のマイクから seconds 秒録って 48 kHz の WAV に納め、登録の列に加える。
pub fn record_operator(state: &Arc<Mutex<AppState>>, seconds: u32) {
    use std::sync::atomic::AtomicBool;
    {
        let mut s = state.lock().unwrap();
        if s.operator_rec.is_some() || s.loading {
            return;
        }
        let dir = std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_default().join("LightVC").join("recordings");
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let path = if dir.is_dir() || std::fs::create_dir_all(&dir).is_ok() {
            dir.join(format!("operator_{stamp}.wav"))
        } else {
            PathBuf::from(format!("operator_{stamp}.wav"))
        };
        s.operator_rec = Some(OperatorRecording { started: std::time::Instant::now(), seconds, path, cancel: Arc::new(AtomicBool::new(false)) });
    }
    let st = state.clone();
    std::thread::spawn(move || {
        let cancel = st.lock().unwrap().operator_rec.as_ref().map(|r| r.cancel.clone());
        let r = (|| -> anyhow::Result<PathBuf> {
            let (engine, mut io) = lightvc_audio::AudioEngine::start_default_buffered(cpal::BufferSize::Fixed(240))?;
            let dev_sr = engine.capture_sample_rate as usize;
            let want = seconds as usize * dev_sr;
            let mut cap = Vec::with_capacity(want + 4800);
            while cap.len() < want {
                if engine.is_disconnected() {
                    anyhow::bail!("オーディオ装置が切断された");
                }
                if cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                    anyhow::bail!("cancelled");
                }
                while let Ok(s) = io.capture.pop() {
                    cap.push(s);
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            drop(engine);
            cap.truncate(want);
            let x48 = crate::cli::resample_to_48k(&cap, dev_sr as u32)?;
            let path = st.lock().unwrap().operator_rec.as_ref().map(|r| r.path.clone()).unwrap_or_default();
            crate::cli::save_wav_mono(&path, &x48, 48_000)?;
            Ok(path)
        })();
        let mut s = st.lock().unwrap();
        let cancelled = r.as_ref().err().is_some_and(|e| e.to_string().contains("cancelled"));
        s.operator_rec = None;
        match r {
            Ok(path) => {
                s.settings.operator.push(path.clone());
                s.status = format!("操作者の録音を保存した: {}", path.display());
            }
            Err(_e) if cancelled => s.status = "操作者の録音を中止した".to_string(),
            Err(e) => s.error = Some(format!("録音の失敗: {e}")),
        }
    });
}

//! LightVC VST3 Plugin
//!
//! Real-time voice conversion as a VST3 audio effect.
//! 変換は paper_snv の製品経路 `SnvStream`(門と重みつきの前段 → K3G = 符号と回路の 3 組の平均 → rvoc3hi・48 kHz・240 サンプル・4 スレッドで実時間)。
//! 登録 = 操作者の普通の発話の WAV と目標の声の参照の WAV(計 20 秒以上)。
//! Uses nice-plug + nice-plug-egui.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossbeam_channel::{unbounded, Receiver, Sender};
use lightvc_core::snv_stream::SnvStream;
use nice_plug::prelude::*;
use nice_plug_egui::{create_egui_editor, EguiState};

const SR: usize = 48_000;
const HOP: usize = SnvStream::BLOCK;
/// 固定の半音(運用 = 0 / 5 / 7 / 12)。Pitch パラメータはこの添字。
const PITCH: [i32; 4] = [0, 5, 7, 12];

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

#[derive(Params)]
struct LightVcParams {
    #[id = "bypass"]
    pub bypass: BoolParam,

    #[id = "pitch"]
    pub pitch: IntParam,

    #[id = "mix"]
    pub mix: FloatParam,

    #[id = "gain"]
    pub output_gain: FloatParam,

    /// SnvStream の部品の書き出し(training/export_{pfront,snv_model,rvoc}.py)
    #[persist = "front-dir"]
    pub front_dir: Arc<Mutex<String>>,

    #[persist = "model-dir"]
    pub model_dir: Arc<Mutex<String>>,

    #[persist = "voc-dir"]
    pub voc_dir: Arc<Mutex<String>>,

    /// 操作者の登録の WAV(1 行に 1 本)
    #[persist = "operator-wavs"]
    pub operator_wavs: Arc<Mutex<String>>,

    /// 目標の声の参照の WAV(1 行に 1 本)
    #[persist = "target-wavs"]
    pub target_wavs: Arc<Mutex<String>>,

    #[persist = "threads"]
    pub threads: Arc<Mutex<usize>>,

    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
}

impl Default for LightVcParams {
    fn default() -> Self {
        Self {
            bypass: BoolParam::new("Bypass", false),
            pitch: IntParam::new("Pitch", 0, IntRange::Linear { min: 0, max: 3 })
                .with_value_to_string(Arc::new(|v| format!("+{}", PITCH[v.clamp(0, 3) as usize]))),
            mix: FloatParam::new(
                "Mix",
                100.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(50.0))
            .with_unit("%"),
            output_gain: FloatParam::new(
                "Output",
                0.0,
                FloatRange::Skewed {
                    min: -24.0,
                    max: 24.0,
                    factor: FloatRange::gain_skew_factor(-24.0, 24.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(20.0))
            .with_unit(" dB"),
            front_dir: Arc::new(Mutex::new(String::new())),
            model_dir: Arc::new(Mutex::new(String::new())),
            voc_dir: Arc::new(Mutex::new(String::new())),
            operator_wavs: Arc::new(Mutex::new(String::new())),
            target_wavs: Arc::new(Mutex::new(String::new())),
            threads: Arc::new(Mutex::new(4)),
            editor_state: EguiState::from_size(420, 560),
        }
    }
}

impl LightVcParams {
    fn load_cfg(&self) -> LoadCfg {
        let lines = |m: &Mutex<String>| -> Vec<PathBuf> {
            m.lock().unwrap().lines().map(str::trim).filter(|l| !l.is_empty()).map(PathBuf::from).collect()
        };
        LoadCfg {
            front: self.front_dir.lock().unwrap().trim().to_string(),
            model: self.model_dir.lock().unwrap().trim().to_string(),
            voc: self.voc_dir.lock().unwrap().trim().to_string(),
            operator: lines(&self.operator_wavs),
            target: lines(&self.target_wavs),
            threads: *self.threads.lock().unwrap(),
            semitones: PITCH[self.pitch.value().clamp(0, 3) as usize] as f32,
        }
    }
}

// ---------------------------------------------------------------------------
// Communication
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct Metrics {
    input_rms: f32,
    output_rms: f32,
    rtf: f32,
    /// Reported plugin latency (ms).
    latency_ms: f32,
    status: String,
}

#[derive(Default)]
struct Shared {
    ready: AtomicBool,
    loading: AtomicBool,
    semitones: AtomicI32,
    metrics: Mutex<Metrics>,
}

struct LoadCfg {
    front: String,
    model: String,
    voc: String,
    operator: Vec<PathBuf>,
    target: Vec<PathBuf>,
    threads: usize,
    semitones: f32,
}

impl LoadCfg {
    fn complete(&self) -> bool {
        !self.front.is_empty() && !self.model.is_empty() && !self.voc.is_empty() && !self.operator.is_empty() && !self.target.is_empty()
    }

    fn build(&self) -> anyhow::Result<SnvStream> {
        anyhow::ensure!(self.complete(), "export dirs, operator WAV and target WAV are required");
        let mut s = SnvStream::load(Path::new(&self.front), Path::new(&self.model), Path::new(&self.voc))?;
        if self.threads > 1 {
            s.enable_parallel(self.threads);
        }
        s.set_semitones(self.semitones);
        let op = self.operator.iter().map(|p| load_48k(p)).collect::<anyhow::Result<Vec<_>>>()?;
        s.enroll_operator(&op.iter().map(|x| x.as_slice()).collect::<Vec<_>>())?;
        let tg = self.target.iter().map(|p| load_48k(p)).collect::<anyhow::Result<Vec<_>>>()?;
        s.enroll_target(&tg.iter().map(|x| x.as_slice()).collect::<Vec<_>>())?;
        Ok(s)
    }
}

enum Task {
    /// Host-rate rings and converters, built in `initialize()`.
    Init(Box<Io>),
    Load(LoadCfg),
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

struct LightVcPlugin {
    params: Arc<LightVcParams>,
    task_tx: Sender<Task>,
    shared: Arc<Shared>,
    // Ring buffer handles for process() (host audio thread)
    capture_tx: Option<rtrb::Producer<f32>>,
    playback_rx: Option<rtrb::Consumer<f32>>,
    /// Dry signal delayed by the reported latency (bypass / mix stay aligned).
    dry: VecDeque<f32>,
    /// Wet samples substituted with silence on underrun; dropped once they arrive.
    debt: usize,
}

impl Default for LightVcPlugin {
    fn default() -> Self {
        let (task_tx, task_rx) = unbounded();
        let shared = Arc::new(Shared::default());
        let sh = shared.clone();
        std::thread::spawn(move || {
            inference_thread(task_rx, sh);
        });
        Self {
            params: Arc::new(LightVcParams::default()),
            task_tx,
            shared,
            capture_tx: None,
            playback_rx: None,
            dry: VecDeque::new(),
            debt: 0,
        }
    }
}

impl Plugin for LightVcPlugin {
    const NAME: &'static str = "LightVC";
    const VENDOR: &'static str = "LightVC";
    const URL: &'static str = "https://github.com/kjranyone/LightVC";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = "0.1.0";

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(1),
        main_output_channels: NonZeroU32::new(1),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames {
            layout: Some("LightVC Mono"),
            main_input: Some("Input"),
            main_output: Some("Output"),
            aux_inputs: &[],
            aux_outputs: &[],
        },
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        struct EditorUserState {
            shared: Arc<Shared>,
            params: Arc<LightVcParams>,
            task_tx: Sender<Task>,
        }

        // Kawaii color constants for the plugin editor
        const BG: egui::Color32 = egui::Color32::from_rgb(28, 22, 38);
        const PANEL: egui::Color32 = egui::Color32::from_rgb(42, 32, 56);
        const PINK: egui::Color32 = egui::Color32::from_rgb(255, 130, 190);
        const PINK_BRIGHT: egui::Color32 = egui::Color32::from_rgb(255, 160, 210);
        const LAVENDER: egui::Color32 = egui::Color32::from_rgb(170, 140, 255);
        const CYAN: egui::Color32 = egui::Color32::from_rgb(120, 230, 255);
        const MINT: egui::Color32 = egui::Color32::from_rgb(130, 255, 200);
        const YELLOW: egui::Color32 = egui::Color32::from_rgb(255, 220, 130);
        const TEXT: egui::Color32 = egui::Color32::from_rgb(240, 235, 250);
        const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(160, 150, 180);

        create_egui_editor(
            self.params.editor_state.clone(),
            EditorUserState {
                shared: self.shared.clone(),
                params: self.params.clone(),
                task_tx: self.task_tx.clone(),
            },
            nice_plug_egui::EguiSettings::default(),
            |ctx, _queue, _state| {
                // Apply dark kawaii background
                let mut style = (*ctx.global_style()).clone();
                style.visuals.dark_mode = true;
                style.visuals.panel_fill = BG;
                style.visuals.widgets.inactive.bg_fill = PANEL;
                style.visuals.widgets.hovered.bg_fill = PINK;
                style.visuals.widgets.active.bg_fill = LAVENDER;
                style.spacing.item_spacing = egui::vec2(8.0, 6.0);
                ctx.set_global_style(style);
            },
            move |ui, setter, _queue, state| {
                let m = state.shared.metrics.lock().unwrap().clone();
                let ready = state.shared.ready.load(Ordering::Relaxed);
                let loading = state.shared.loading.load(Ordering::Relaxed);
                let params = state.params.clone();

                // Header
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("✦ LightVC")
                            .size(16.0)
                            .strong()
                            .color(PINK_BRIGHT),
                    );
                });

                ui.add_space(6.0);

                // Status
                let (dot_color, status_text) = if loading {
                    (YELLOW, "● LOADING")
                } else if ready {
                    (MINT, "● READY")
                } else {
                    (TEXT_DIM, "● NO MODEL")
                };
                ui.label(egui::RichText::new(status_text).size(13.0).color(dot_color));
                if !m.status.is_empty() {
                    ui.label(egui::RichText::new(&m.status).size(10.0).color(TEXT_DIM));
                }

                ui.add_space(8.0);

                // Metrics
                if ready {
                    for (name, v, color) in [("In", m.input_rms, MINT), ("Out", m.output_rms, CYAN)] {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(name).size(11.0).color(TEXT_DIM));
                            let (rect, _) =
                                ui.allocate_exact_size(egui::vec2(80.0, 10.0), egui::Sense::hover());
                            let level = (v * 10.0).clamp(0.0, 1.0);
                            ui.painter().rect_filled(rect, 4.0, BG);
                            ui.painter().rect_filled(
                                egui::Rect::from_min_size(
                                    rect.min,
                                    egui::vec2(rect.width() * level, rect.height()),
                                ),
                                4.0,
                                if level > 0.8 {
                                    PINK
                                } else if level > 0.5 {
                                    YELLOW
                                } else {
                                    color
                                },
                            );
                        });
                    }
                    ui.label(
                        egui::RichText::new(format!("RTF: {:.2}", m.rtf))
                            .size(10.0)
                            .color(TEXT_DIM),
                    );
                }
                ui.label(
                    egui::RichText::new(format!("Latency: {:.0} ms", m.latency_ms))
                        .size(10.0)
                        .color(TEXT_DIM),
                );

                ui.add_space(10.0);

                // Bypass button
                let bypassed = params.bypass.value();
                let btn_text = if bypassed { "▶ ENGAGE" } else { "■ BYPASS" };
                let btn = egui::Button::new(
                    egui::RichText::new(btn_text)
                        .size(13.0)
                        .strong()
                        .color(TEXT),
                )
                .fill(if bypassed { LAVENDER } else { PINK })
                .stroke(egui::Stroke::new(2.0, CYAN))
                .min_size(egui::vec2(120.0, 30.0));
                if ui.add(btn).clicked() {
                    setter.set_parameter(&params.bypass, !bypassed);
                }

                ui.add_space(6.0);

                // Pitch buttons (fixed semitones)
                ui.label(egui::RichText::new("Pitch").size(11.0).color(TEXT_DIM));
                ui.horizontal(|ui| {
                    let cur = params.pitch.value();
                    for (i, k) in PITCH.iter().enumerate() {
                        let selected = cur == i as i32;
                        let btn = egui::Button::new(
                            egui::RichText::new(format!("+{k}")).size(11.0).color(if selected {
                                TEXT
                            } else {
                                TEXT_DIM
                            }),
                        )
                        .fill(if selected { PINK } else { PANEL })
                        .stroke(egui::Stroke::new(
                            1.0,
                            if selected { PINK_BRIGHT } else { LAVENDER },
                        ));
                        if ui.add(btn).clicked() {
                            setter.set_parameter(&params.pitch, i as i32);
                        }
                    }
                });

                ui.add_space(8.0);

                // Mix + Gain knobs (pure egui, no image assets)
                ui.horizontal(|ui| {
                    // Mix knob
                    let mix_val = params.mix.value();
                    let mix_norm = mix_val / 100.0;
                    if let Some(new) =
                        egui_knob(ui, "mix_knob", mix_norm, "Mix", &format!("{:.0}%", mix_val))
                    {
                        setter.set_parameter(&params.mix, new * 100.0);
                    }

                    ui.add_space(12.0);

                    // Gain knob
                    let gain_val = params.output_gain.value();
                    let gain_norm = (gain_val + 24.0) / 48.0;
                    if let Some(new) = egui_knob(
                        ui,
                        "gain_knob",
                        gain_norm,
                        "Gain",
                        &format!("{:+.1}dB", gain_val),
                    ) {
                        setter.set_parameter(&params.output_gain, new * 48.0 - 24.0);
                    }
                });

                ui.add_space(8.0);

                // ---- Model + enrollment ----
                egui::CollapsingHeader::new("Model / Enrollment")
                    .default_open(!ready)
                    .show(ui, |ui| {
                        for (label, field) in [
                            ("Front export dir", &params.front_dir),
                            ("E / C export dir", &params.model_dir),
                            ("Vocoder export dir", &params.voc_dir),
                        ] {
                            ui.label(egui::RichText::new(label).size(9.0).color(TEXT_DIM));
                            let mut v = field.lock().unwrap();
                            ui.add(
                                egui::TextEdit::singleline(&mut *v)
                                    .desired_width(380.0)
                                    .text_color(TEXT),
                            );
                        }
                        for (label, field) in [
                            ("Operator WAV (one per line)", &params.operator_wavs),
                            ("Target voice WAV, 20 s+ (one per line)", &params.target_wavs),
                        ] {
                            ui.label(egui::RichText::new(label).size(9.0).color(TEXT_DIM));
                            let mut v = field.lock().unwrap();
                            ui.add(
                                egui::TextEdit::multiline(&mut *v)
                                    .desired_width(380.0)
                                    .desired_rows(2)
                                    .text_color(TEXT),
                            );
                        }
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Threads").size(9.0).color(TEXT_DIM));
                            let mut n = params.threads.lock().unwrap();
                            for k in [1usize, 2, 4] {
                                ui.selectable_value(&mut *n, k, k.to_string());
                            }
                        });
                        let cfg = params.load_cfg();
                        let can = cfg.complete() && !loading;
                        if ui.add_enabled(can, egui::Button::new("Load")).clicked() {
                            let _ = state.task_tx.send(Task::Load(cfg));
                        }
                    });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let h = buffer_config.sample_rate as usize;
        let (up, down) = if h == SR {
            (None, None)
        } else {
            match (Conv::new(h, SR, HOP), Conv::new(SR, h, HOP)) {
                (Ok(u), Ok(d)) => (Some(u), Some(d)),
                (Err(e), _) | (_, Err(e)) => {
                    nice_log!("Resampler init failed: {e}");
                    return false;
                }
            }
        };

        // Latency = pre-filled wet ring (host buffer + block / chunk granularity + one block of
        // scheduling margin) + resampler delays + algorithmic latency of SnvStream.
        let to_host = |n48: usize| (n48 as f64 * h as f64 / SR as f64).ceil() as usize;
        let up_chunk = up.as_ref().map_or(0, |u| u.input_frames_next());
        let prefill = buffer_config.max_buffer_size as usize + up_chunk + to_host(2 * HOP);
        let delay = up.as_ref().map_or(0, |u| to_host(u.output_delay()))
            + to_host(SnvStream::LATENCY_SAMPLES)
            + down.as_ref().map_or(0, |d| d.output_delay());
        let latency = prefill + delay;

        let cap = (h / 5).max(16384).max(2 * latency);
        let (capture_tx, capture_rx) = rtrb::RingBuffer::new(cap);
        let (mut playback_tx, playback_rx) = rtrb::RingBuffer::new(cap);
        for _ in 0..prefill {
            let _ = playback_tx.push(0.0);
        }
        self.capture_tx = Some(capture_tx);
        self.playback_rx = Some(playback_rx);
        self.dry = VecDeque::with_capacity(latency + 1);
        self.dry.resize(latency, 0.0);
        self.debt = 0;
        let _ = self.task_tx.send(Task::Init(Box::new(Io {
            capture_rx,
            playback_tx,
            up,
            down,
            cap: Vec::new(),
            in48: Vec::new(),
            out48: Vec::new(),
            block: vec![0.0; HOP],
        })));
        context.set_latency_samples(latency as u32);
        self.shared.metrics.lock().unwrap().latency_ms = latency as f32 * 1000.0 / h as f32;

        let cfg = self.params.load_cfg();
        if cfg.complete() && !self.shared.ready.load(Ordering::Relaxed) && !self.shared.loading.load(Ordering::Relaxed) {
            let _ = self.task_tx.send(Task::Load(cfg));
        }

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let bypass = self.params.bypass.value() || !self.shared.ready.load(Ordering::Relaxed);
        self.shared
            .semitones
            .store(PITCH[self.params.pitch.value().clamp(0, 3) as usize], Ordering::Relaxed);

        let (Some(tx), Some(rx)) = (self.capture_tx.as_mut(), self.playback_rx.as_mut()) else {
            return ProcessStatus::Normal;
        };
        while self.debt > 0 && rx.pop().is_ok() {
            self.debt -= 1;
        }

        // Everything flows through the inference thread so the latency stays constant;
        // bypass / no model output the dry signal delayed by the same latency.
        for channel_samples in buffer.iter_samples() {
            let mix = self.params.mix.smoothed.next() / 100.0;
            let gain = 10.0f32.powf(self.params.output_gain.smoothed.next() / 20.0);
            for sample in channel_samples {
                let x = *sample;
                let _ = tx.push(x);
                self.dry.push_back(x);
                let d = self.dry.pop_front().unwrap_or(0.0);
                let w = rx.pop().unwrap_or_else(|_| {
                    self.debt += 1;
                    0.0
                });
                *sample = if bypass { d } else { d * (1.0 - mix) + w * mix } * gain;
            }
        }

        ProcessStatus::Normal
    }
}

// ---------------------------------------------------------------------------
// Resampler (host SR ↔ 48 kHz)
// ---------------------------------------------------------------------------

use rubato::{
    audioadapter_buffers::direct::SequentialSlice, Async, FixedAsync, Resampler as _,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Rate converter with a fixed input chunk.
struct Conv {
    r: Async<f32>,
    out: Vec<f32>,
}

impl Conv {
    fn new(from_sr: usize, to_sr: usize, chunk_size: usize) -> anyhow::Result<Self> {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let r = Async::<f32>::new_sinc(to_sr as f64 / from_sr as f64, 2.0, &params, chunk_size, 1, FixedAsync::Input)?;
        Ok(Self {
            out: vec![0.0; r.output_frames_max()],
            r,
        })
    }

    fn input_frames_next(&self) -> usize {
        self.r.input_frames_next()
    }

    fn output_delay(&self) -> usize {
        self.r.output_delay()
    }

    fn process(&mut self, input: &[f32]) -> anyhow::Result<&[f32]> {
        let needed = self.r.input_frames_next();
        anyhow::ensure!(input.len() >= needed, "resampler: need {needed}, got {}", input.len());
        let frames_out = self.r.output_frames_next();
        let in_adapter = SequentialSlice::new(input, 1, needed)?;
        let mut out_adapter = SequentialSlice::new_mut(&mut self.out, 1, frames_out)?;
        let (_used, written) = self.r.process_into_buffer(&in_adapter, &mut out_adapter, None)?;
        Ok(&self.out[..written])
    }
}

/// WAV を読み、48 kHz のモノラルにする(登録用)。
fn load_48k(path: &Path) -> anyhow::Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let ch = spec.channels as usize;
    let x: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    let x: Vec<f32> = x.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    let sr = spec.sample_rate as usize;
    if sr == SR {
        return Ok(x);
    }
    let mut conv = Conv::new(sr, SR, 1024)?;
    let want = (x.len() as f64 * SR as f64 / sr as f64).round() as usize;
    let delay = conv.output_delay();
    let mut out = Vec::with_capacity(want + delay + 2048);
    let mut pos = 0;
    while out.len() < want + delay {
        let k = conv.input_frames_next();
        let mut chunk = vec![0.0f32; k];
        let n = x.len().saturating_sub(pos).min(k);
        chunk[..n].copy_from_slice(&x[pos..pos + n]);
        out.extend_from_slice(conv.process(&chunk)?);
        pos += k;
    }
    out.drain(..delay);
    out.truncate(want);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Inference thread
// ---------------------------------------------------------------------------

/// Host-rate capture → 48 kHz → SnvStream (240-sample blocks) → host-rate playback.
/// `up` / `down` are None when the host runs at 48 kHz.
struct Io {
    capture_rx: rtrb::Consumer<f32>,
    playback_tx: rtrb::Producer<f32>,
    up: Option<Conv>,
    down: Option<Conv>,
    cap: Vec<f32>,
    in48: Vec<f32>,
    out48: Vec<f32>,
    block: Vec<f32>,
}

impl Io {
    fn pump(&mut self, mut stream: Option<&mut SnvStream>, shared: &Shared) -> bool {
        let mut did = false;
        while let Ok(s) = self.capture_rx.pop() {
            self.cap.push(s);
            did = true;
        }
        match self.up.as_mut() {
            Some(u) => {
                while self.cap.len() >= u.input_frames_next() {
                    let k = u.input_frames_next();
                    match u.process(&self.cap[..k]) {
                        Ok(y) => self.in48.extend_from_slice(y),
                        Err(e) => nice_log!("resample up: {e}"),
                    }
                    self.cap.drain(..k);
                }
            }
            None => self.in48.append(&mut self.cap),
        }

        let mut stats = None;
        while self.in48.len() >= HOP {
            let t0 = Instant::now();
            let blk = &self.in48[..HOP];
            match stream.as_deref_mut() {
                Some(s) => s.process_block(blk, &mut self.block),
                None => self.block.copy_from_slice(blk),
            }
            stats = Some((rms(blk), rms(&self.block), t0.elapsed().as_secs_f32() / (HOP as f32 / SR as f32)));
            self.out48.extend(self.block.iter().map(|v| v.clamp(-1.0, 1.0)));
            self.in48.drain(..HOP);
            did = true;
        }

        match self.down.as_mut() {
            Some(d) => {
                while self.out48.len() >= d.input_frames_next() {
                    let k = d.input_frames_next();
                    match d.process(&self.out48[..k]) {
                        Ok(y) => {
                            for v in y {
                                let _ = self.playback_tx.push(*v);
                            }
                        }
                        Err(e) => nice_log!("resample down: {e}"),
                    }
                    self.out48.drain(..k);
                }
            }
            None => {
                for v in self.out48.drain(..) {
                    let _ = self.playback_tx.push(v);
                }
            }
        }

        if let Some((i, o, rtf)) = stats {
            let mut m = shared.metrics.lock().unwrap();
            m.input_rms = i;
            m.output_rms = o;
            m.rtf = rtf;
        }
        did
    }
}

fn inference_thread(task_rx: Receiver<Task>, shared: Arc<Shared>) {
    let (load_tx, load_rx) = unbounded::<anyhow::Result<SnvStream>>();
    let mut stream: Option<SnvStream> = None;
    let mut io: Option<Io> = None;
    let mut semitones = i32::MIN;

    loop {
        while let Ok(task) = task_rx.try_recv() {
            match task {
                Task::Init(v) => io = Some(*v),
                Task::Load(cfg) => {
                    if shared.loading.swap(true, Ordering::Relaxed) {
                        continue;
                    }
                    shared.metrics.lock().unwrap().status = "Loading model and enrolling...".into();
                    let tx = load_tx.clone();
                    std::thread::spawn(move || {
                        let _ = tx.send(cfg.build());
                    });
                }
            }
        }

        while let Ok(r) = load_rx.try_recv() {
            shared.loading.store(false, Ordering::Relaxed);
            match r {
                Ok(s) => {
                    stream = Some(s);
                    semitones = i32::MIN;
                    shared.ready.store(true, Ordering::Relaxed);
                    shared.metrics.lock().unwrap().status = "Loaded".into();
                }
                Err(e) => {
                    nice_log!("Model load failed: {e}");
                    shared.metrics.lock().unwrap().status = format!("Load failed: {e}");
                }
            }
        }

        let k = shared.semitones.load(Ordering::Relaxed);
        if k != semitones && let Some(s) = stream.as_mut() {
            s.set_semitones(k as f32);
            semitones = k;
        }

        let Some(io) = io.as_mut() else {
            std::thread::sleep(std::time::Duration::from_millis(20));
            continue;
        };
        if !io.pump(stream.as_mut(), &shared) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

// ---------------------------------------------------------------------------
// Pure-egui knob widget (no image assets needed)
// ---------------------------------------------------------------------------

/// Draw an interactive knob using egui primitives only.
/// `value` is normalized 0.0..=1.0; returns the new normalized value if dragged.
fn egui_knob(
    ui: &mut egui::Ui,
    _id_str: &str,
    value: f32,
    label: &str,
    display_text: &str,
) -> Option<f32> {
    let knob_size = 52.0;
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(knob_size + 20.0, knob_size + 30.0),
        egui::Sense::drag(),
    );

    let center = egui::pos2(rect.center().x, rect.min.y + knob_size / 2.0 + 4.0);
    let radius = knob_size / 2.0;

    let mut new_val = None;

    if response.dragged() {
        let drag = response.drag_delta().y;
        new_val = Some((value - drag * 0.005).clamp(0.0, 1.0));
    }
    if response.double_clicked() {
        new_val = Some(0.5);
    }

    let shown = new_val.unwrap_or(value);
    let painter = ui.painter();

    // Background circle
    painter.circle_filled(center, radius + 2.0, egui::Color32::from_rgb(28, 22, 38));

    // Arc background (270° span from -135° to +135°)
    let start_angle = -135.0_f32.to_radians();
    let end_angle = 135.0_f32.to_radians();
    let bg_points = (0..40)
        .map(|i| {
            let t = i as f32 / 39.0;
            let a = start_angle + (end_angle - start_angle) * t;
            center + egui::vec2(a.cos() * radius, a.sin() * radius)
        })
        .collect::<Vec<_>>();
    painter.add(egui::Shape::line(
        bg_points,
        egui::Stroke::new(3.0, egui::Color32::from_rgb(52, 40, 68)),
    ));

    // Value arc
    let val_end = start_angle + (end_angle - start_angle) * shown;
    let val_points = (0..40)
        .take(((shown * 39.0).round() as usize).max(1))
        .map(|i| {
            let t = i as f32 / 39.0;
            let a = start_angle + (val_end - start_angle) * t.min(1.0);
            center + egui::vec2(a.cos() * radius, a.sin() * radius)
        })
        .collect::<Vec<_>>();

    let arc_color = if response.dragged() {
        egui::Color32::from_rgb(255, 160, 210)
    } else if response.hovered() {
        egui::Color32::from_rgb(170, 140, 255)
    } else {
        egui::Color32::from_rgb(255, 130, 190)
    };
    if val_points.len() >= 2 {
        painter.add(egui::Shape::line(
            val_points,
            egui::Stroke::new(3.0, arc_color),
        ));
    }

    // Indicator line
    let ind_a = start_angle + (end_angle - start_angle) * shown;
    let ind_start = center + egui::vec2(ind_a.cos() * (radius - 8.0), ind_a.sin() * (radius - 8.0));
    let ind_end = center + egui::vec2(ind_a.cos() * radius, ind_a.sin() * radius);
    painter.line_segment([ind_start, ind_end], egui::Stroke::new(2.5, arc_color));

    // Inner circle
    painter.circle_filled(center, radius - 6.0, egui::Color32::from_rgb(42, 32, 56));

    // Glow when active
    if response.dragged() {
        painter.circle_stroke(
            center,
            radius + 4.0,
            egui::Stroke::new(
                2.0,
                egui::Color32::from_rgba_premultiplied(255, 130, 190, 60),
            ),
        );
    }

    // Value text in center
    painter.text(
        center,
        egui::Align2::CENTER_CENTER,
        display_text,
        egui::FontId::proportional(10.0),
        egui::Color32::from_rgb(240, 235, 250),
    );

    // Label below
    painter.text(
        egui::pos2(center.x, rect.min.y + knob_size + 12.0),
        egui::Align2::CENTER_TOP,
        label,
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(160, 150, 180),
    );

    new_val
}

// ---------------------------------------------------------------------------
// CLAP export
// ---------------------------------------------------------------------------

impl ClapPlugin for LightVcPlugin {
    const CLAP_ID: &'static str = "com.lightvc.lightvc";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Real-time voice conversion");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

nice_export_clap!(LightVcPlugin);

// ---------------------------------------------------------------------------
// VST3 wrapper export (via clap-wrapper, Steinberg VST3 SDK is MIT as of 2025)
// ---------------------------------------------------------------------------

#[cfg(feature = "vst3")]
clap_wrapper::export_vst3!();

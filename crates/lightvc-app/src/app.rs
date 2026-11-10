//! LightVC GUI — 3-tab egui application.
//!
//! Tab 1: Offline conversion (file → convert → save)
//! Tab 2: Real-time conversion (mic → VC → speaker)
//! Tab 3: Voice catalog (target voice references)
//!
//! 変換は paper_snv の製品経路 `lightvc_core::snv_stream::SnvStream`(前段 v3 → E / C → rvoc3hi・48 kHz・240 サンプル)。
//! 登録 = 操作者の普通の発話(計 30 秒程度)と目標の声の参照(計 20 秒以上)。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
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
}

/// Control messages from UI to real-time inference thread.
pub enum RtControl {
    StartWithDevices {
        input_idx: Option<usize>,
        output_idx: Option<usize>,
    },
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

/// Hot-swappable conversion engine shared with the inference thread.
pub type EngineSlot = Arc<Mutex<Option<SnvStream>>>;

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
            front: "results/pfront_export".into(),
            model: "results/p_model_export".into(),
            voc: "results/rvoc3hi/export_330k".into(),
            operator: Vec::new(),
            threads: 4,
            semitones: 0.0,
        }
    }
}

/// Application-wide shared state.
pub struct AppState {
    pub engine: EngineSlot,
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
    /// App-wide audio device selection (None = system default).
    /// Indices match DuplexStream::list_input_devices() /
    /// list_output_devices() ordering.
    pub selected_input: Option<usize>,
    pub selected_output: Option<usize>,
}

#[derive(Clone)]
pub struct VoiceEntry {
    pub name: String,
    pub path: std::path::PathBuf,
}

/// WAV を読み、48 kHz のモノラルにする。
pub fn load_48k(path: &Path) -> anyhow::Result<Vec<f32>> {
    let (x, sr) = crate::audio_playback::load_wav_mono(path)?;
    crate::cli::resample_to_48k(&x, sr)
}

/// 設定と目標の参照から SnvStream を組む(読み込み + 登録)。
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
    Ok(s)
}

#[derive(PartialEq, Copy, Clone)]
enum Tab {
    Offline,
    Realtime,
    Catalog,
}

pub struct LightVcApp {
    state: Arc<Mutex<AppState>>,
    current_tab: Tab,
    offline: crate::offline_tab::OfflineState,
    catalog: crate::voice_catalog::CatalogState,
    rt_running: bool,
    rt_bypass: bool,
    /// Demo mode for screenshot capture (None = normal operation).
    demo: bool,
    rt_metrics: RtMetrics,
    /// File picker for the operator enrollment WAVs.
    operator_pick: crate::file_pick::FilePick,
    /// File pickers for the Catalog add/import actions.
    catalog_add_pick: crate::file_pick::FilePick,
    catalog_import_pick: crate::file_pick::FilePick,
    asset_cache: crate::assets::AssetCache,
    splash_frames: u32, // 0 = showing splash, >0 = finished
    settings_open: bool,
    /// Output mute (Transport). Silences the wet signal without unarming.
    rt_muted: bool,
    /// Selected fixed buffer size in frames (128/256/512/1024).
    rt_buffer_frames: u32,
}

impl LightVcApp {
    pub fn new() -> Self {
        let state = Arc::new(Mutex::new(AppState {
            engine: Arc::new(Mutex::new(None)),
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
            selected_input: None,
            selected_output: None,
        }));

        Self {
            state,
            current_tab: Tab::Offline,
            offline: Default::default(),
            catalog: Default::default(),
            rt_running: false,
            rt_bypass: false,
            demo: false,
            rt_metrics: RtMetrics::default(),
            operator_pick: Default::default(),
            catalog_add_pick: Default::default(),
            catalog_import_pick: Default::default(),
            asset_cache: Default::default(),
            splash_frames: 0,
            settings_open: false,
            rt_muted: false,
            rt_buffer_frames: 256,
        }
    }

    /// Enable demo mode for screenshot capture. Injects mock data so all
    /// three tabs render fully without a model or audio devices.
    pub fn enable_demo(&mut self) {
        self.demo = true;
        self.splash_frames = 60;
        {
            let mut s = self.state.lock().unwrap();
            s.model_loaded = true;
            s.settings.operator = vec!["samples/operator.wav".into()];
            s.voices = vec![
                VoiceEntry { name: "Venus (warm)".into(), path: "samples/venus.wav".into() },
                VoiceEntry { name: "Mars (bright)".into(), path: "samples/mars.wav".into() },
                VoiceEntry { name: "Lyra (soft)".into(), path: "samples/lyra.wav".into() },
            ];
            s.selected_voice = Some(0);
            s.status = "Demo mode".into();
        }
        self.offline.source_path = "samples/source_male.wav".into();
        self.offline.reference_path = "samples/venus.wav".into();
        self.offline.converted_samples = Some(vec![0.0; 48_000]);
        self.rt_running = true;
        self.rt_bypass = false;
        self.rt_metrics = RtMetrics {
            input_rms: 0.18,
            output_rms: 0.12,
            latency_ms: 31.0,
            rtf: 0.34,
            disconnected: false,
            overrun: 0,
            underrun: 2,
        };
        self.current_tab = Tab::Realtime;
    }

    fn ensure_rt_thread_static(state: &Arc<Mutex<AppState>>) {
        let mut s = state.lock().unwrap();
        if s.rt_initialized {
            return;
        }
        let (control_tx, control_rx) = unbounded();
        let (metrics_tx, metrics_rx) = unbounded();
        let engine = s.engine.clone();
        s.rt_control_tx = Some(control_tx);
        s.rt_metrics_rx = Some(metrics_rx);
        s.rt_initialized = true;
        drop(s);
        // The inference thread runs even without an engine (bypass), so the
        // audio path can be tested before the model is loaded.
        std::thread::spawn(move || {
            crate::realtime_tab::inference_loop(engine, control_rx, metrics_tx);
        });
    }

    /// 選んだ目標の声で SnvStream を組み直し、組めたら差し替える(重い処理は別の糸で・音は止めない)。
    pub fn load_engine_static(state: &Arc<Mutex<AppState>>) {
        let (cfg, target, slot) = {
            let mut s = state.lock().unwrap();
            if s.loading {
                return;
            }
            let target: Vec<PathBuf> = s.selected_voice.and_then(|i| s.voices.get(i)).map(|v| vec![v.path.clone()]).unwrap_or_default();
            s.loading = true;
            s.status = "Loading model and enrolling...".to_string();
            s.error = None;
            (s.settings.clone(), target, s.engine.clone())
        };
        let st = state.clone();
        std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            let r = build_stream(&cfg, &target);
            let mut s = st.lock().unwrap();
            s.loading = false;
            match r {
                Ok(engine) => {
                    let old = slot.lock().unwrap().replace(engine);
                    drop(old);
                    s.model_loaded = true;
                    s.status = format!("Model loaded・enrolled in {:.1} s", t0.elapsed().as_secs_f64());
                }
                Err(e) => s.error = Some(format!("Load failed: {e}")),
            }
        });
    }

    fn send_control(state: &Arc<Mutex<AppState>>, ctrl: RtControl) {
        let s = state.lock().unwrap();
        if let Some(ref tx) = s.rt_control_tx {
            let _ = tx.send(ctrl);
        }
    }
}

impl Default for LightVcApp {
    fn default() -> Self {
        Self::new()
    }
}

impl eframe::App for LightVcApp {
    /// Root UI entry point (eframe 0.34 required method). Wraps the
    /// context-level render in a CentralPanel so that nested panels work.
    #[allow(deprecated)]
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // We render at the ctx level (top-level panels), so we don't use
        // the passed `ui` directly. Add an empty area to satisfy the API.
        ui.allocate_space(ui.available_size());
        self.render(&ctx);
    }
}

impl LightVcApp {
    // egui 0.34 deprecated CentralPanel::show/Panel::show in favor of
    // show_inside(), but migrating requires restructuring the entire app
    // (each panel must nest inside a parent Ui). Kept as top-level show()
    // until a full Ui-tree migration is done.
    #[allow(deprecated)]
    pub fn render(&mut self, ctx: &egui::Context) {
        // Apply kawaii theme
        crate::theme::apply_theme(ctx);

        // Splash screen — show for ~30 frames (~0.5s at 60fps)
        if self.splash_frames < 30 {
            self.splash_frames += 1;
            let alpha = if self.splash_frames < 20 {
                1.0
            } else {
                1.0 - (self.splash_frames - 20) as f32 / 10.0
            };

            let splash = self.asset_cache.splash(ctx);
            let screen = ctx.content_rect();
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(0xF0, 0xE0, 0xEC)))
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(screen.height() * 0.25);
                        let size = egui::Vec2::new(300.0, 150.0);
                        ui.add(
                            egui::Image::from_texture(splash)
                                .fit_to_exact_size(size)
                                .tint(egui::Color32::from_rgba_premultiplied(
                                    255,
                                    255,
                                    255,
                                    (alpha * 255.0) as u8,
                                )),
                        );
                        ui.add_space(8.0);
                        ui.spinner();
                    });
                });
            ctx.request_repaint();
            return;
        }

        // Draw kawaii neon-glow background (procedural gradient + blooms).
        // Replaces the opaque PNG texture — translucent cards now float on
        // the neon gradient and read as genuine glassmorphism.
        {
            let screen = ctx.content_rect();
            crate::theme::paint_background(&ctx.layer_painter(egui::LayerId::background()), screen);
        }

        // Top bar with logo image + kawaii tabs
        egui::Panel::top("tabs")
            .frame(
                egui::Frame::NONE
                    .fill(crate::theme::with_alpha(
                        crate::theme::colors::CARD_GLASS,
                        180,
                    ))
                    .stroke(egui::Stroke::new(1.0, crate::theme::colors::BORDER))
                    .inner_margin(egui::Margin::symmetric(10, 4)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(2.0);
                    // Logo image — crop source margins, readable header height
                    let logo = self.asset_cache.logo(ctx);
                    let logo_h = 24.0;
                    let aspect = 171.0 / 41.0;
                    let logo_w = logo_h * aspect;
                    let uv = crate::assets::AssetCache::logo_crop_uv();
                    ui.add(
                        egui::Image::from_texture(logo)
                            .fit_to_exact_size(egui::Vec2::new(logo_w, logo_h))
                            .uv(egui::Rect::from_min_max(
                                egui::pos2(uv.min.x, 1.0 - uv.max.y),
                                egui::pos2(uv.max.x, 1.0 - uv.min.y),
                            )),
                    );
                    ui.add_space(crate::theme::space::SMALL);

                    // Tabs — distribute remaining width
                    let tab_labels = [
                        (Tab::Offline, "Offline"),
                        (Tab::Realtime, "Realtime"),
                        (Tab::Catalog, "Voices"),
                    ];
                    for (tab, label) in &tab_labels {
                        let selected = self.current_tab == *tab;
                        if crate::theme::tab_button(ui, label, selected) {
                            self.current_tab = *tab;
                        }
                    }

                    // Settings button — pinned to right edge via right-to-left layout
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("=").clicked() {
                            self.settings_open = !self.settings_open;
                        }
                    });
                });
            });

        // Settings dialog — app-wide device selection
        if self.settings_open {
            let inputs = lightvc_audio::DuplexStream::list_input_devices().unwrap_or_default();
            let outputs = lightvc_audio::DuplexStream::list_output_devices().unwrap_or_default();
            let (mut sel_in, mut sel_out) = {
                let s = self.state.lock().unwrap();
                (s.selected_input, s.selected_output)
            };
            egui::Window::new("Audio Devices")
                .open(&mut self.settings_open)
                .resizable(true)
                .default_width(360.0)
                .default_height(420.0)
                .min_width(280.0)
                .min_height(240.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(360.0)
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("Input")
                                    .size(12.0)
                                    .color(crate::theme::colors::TEXT_DIM),
                            );
                            ui.selectable_value(&mut sel_in, None, "(default)");
                            for (i, d) in inputs.iter().enumerate() {
                                let lbl =
                                    format!("{} ({}Hz, {}ch)", d.name, d.sample_rate, d.channels);
                                ui.selectable_value(&mut sel_in, Some(i), &lbl);
                            }
                            ui.add_space(8.0);
                            ui.separator();
                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new("Output")
                                    .size(12.0)
                                    .color(crate::theme::colors::TEXT_DIM),
                            );
                            ui.selectable_value(&mut sel_out, None, "(default)");
                            for (i, d) in outputs.iter().enumerate() {
                                let lbl =
                                    format!("{} ({}Hz, {}ch)", d.name, d.sample_rate, d.channels);
                                ui.selectable_value(&mut sel_out, Some(i), &lbl);
                            }
                        });
                });
            let mut s = self.state.lock().unwrap();
            s.selected_input = sel_in;
            s.selected_output = sel_out;
        }

        // Status bar
        {
            let st = self.state.lock().unwrap();
            egui::Panel::bottom("status")
                .frame(
                    egui::Frame::NONE
                        .fill(crate::theme::with_alpha(
                            crate::theme::colors::CARD_GLASS,
                            180,
                        ))
                        .stroke(egui::Stroke::new(1.0, crate::theme::colors::BORDER))
                        .inner_margin(egui::Margin::same(8)),
                )
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        let (dot_color, msg) = if let Some(ref err) = st.error {
                            (crate::theme::colors::ERROR, err.clone())
                        } else {
                            (crate::theme::colors::MINT, st.status.clone())
                        };
                        crate::theme::status_dot(ui, true, dot_color);
                        ui.label(
                            egui::RichText::new(&msg)
                                .size(12.0)
                                .color(crate::theme::colors::TEXT_DIM),
                        );

                        // Output level meter (right-aligned)
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                egui::RichText::new("Out")
                                    .size(11.0)
                                    .color(crate::theme::colors::TEXT_DIM),
                            );
                            crate::theme::level_meter_kind_compact(
                                ui,
                                self.rt_metrics.output_rms,
                                crate::theme::MeterKind::Output,
                            );
                        });
                    });
                });
        }

        // Tab content
        match self.current_tab {
            Tab::Offline => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            let folder = self.asset_cache.icon_folder(ctx).clone();
                            let play = self.asset_cache.icon_play(ctx).clone();
                            let convert = self.asset_cache.icon_convert(ctx).clone();
                            let speaker = self.asset_cache.icon_speaker(ctx).clone();
                            let mic = self.asset_cache.icon_mic(ctx).clone();
                            crate::offline_tab::render(
                                ui,
                                ctx,
                                &self.state,
                                &mut self.offline,
                                &folder,
                                &play,
                                &convert,
                                &speaker,
                                &mic,
                            );
                        });
                });
            }
            Tab::Realtime => {
                let mut disconnected = false;
                {
                    let s = self.state.lock().unwrap();
                    if let Some(ref rx) = s.rt_metrics_rx {
                        while let Ok(m) = rx.try_recv() {
                            disconnected |= m.disconnected;
                            self.rt_metrics = m;
                        }
                    }
                }
                if disconnected {
                    self.rt_running = false;
                    self.state.lock().unwrap().status = "Audio device disconnected".to_string();
                }

                let state = self.state.clone();
                let mut rt_running = self.rt_running;
                let mut rt_bypass = self.rt_bypass;
                let mut rt_muted = self.rt_muted;
                let mut rt_buffer_frames = self.rt_buffer_frames;
                let metrics = self.rt_metrics.clone();
                let operator_pick = self.operator_pick.clone();

                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            crate::realtime_tab::render(
                                ui,
                                ctx,
                                &operator_pick,
                                &state,
                                &mut rt_running,
                                &mut rt_bypass,
                                &mut rt_muted,
                                &mut rt_buffer_frames,
                                &metrics,
                                || Self::load_engine_static(&state),
                                || Self::ensure_rt_thread_static(&state),
                                |ctrl| Self::send_control(&state, ctrl),
                                &mut self.asset_cache,
                            );
                        });
                });

                self.rt_running = rt_running;
                self.rt_bypass = rt_bypass;
                self.rt_muted = rt_muted;
                self.rt_buffer_frames = rt_buffer_frames;
            }
            Tab::Catalog => {
                let mut catalog = std::mem::take(&mut self.catalog);
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            let folder = self.asset_cache.icon_folder(ctx).clone();
                            let play = self.asset_cache.icon_play(ctx).clone();
                            let trash = self.asset_cache.icon_trash(ctx).clone();
                            let empty = self.asset_cache.empty_stars(ctx).clone();
                            let state = self.state.clone();
                            let add_pick = self.catalog_add_pick.clone();
                            let import_pick = self.catalog_import_pick.clone();
                            crate::voice_catalog::render(
                                ui,
                                ctx,
                                &add_pick,
                                &import_pick,
                                &state,
                                &mut catalog,
                                &folder,
                                &play,
                                &trash,
                                &empty,
                                |idx| {
                                    // Use the selected voice as the target. If a model is
                                    // already loaded, rebuild the engine with the new target.
                                    let reload = {
                                        let mut s = state.lock().unwrap();
                                        if s.voices.get(idx).is_some() {
                                            s.selected_voice = Some(idx);
                                        }
                                        s.model_loaded && !s.settings.operator.is_empty()
                                    };
                                    if reload {
                                        Self::load_engine_static(&state);
                                    }
                                },
                            );
                        });
                });
                self.catalog = catalog;
            }
        }

        // Only request continuous repaint during realtime mode, while loading,
        // or while a Catalog preview is playing.
        let loading = self.state.lock().unwrap().loading;
        if self.current_tab == Tab::Realtime
            || loading
            || (self.current_tab == Tab::Catalog && self.catalog.playing_voice.is_some())
        {
            ctx.request_repaint();
        }
    }
}

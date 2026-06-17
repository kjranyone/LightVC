//! LightVC GUI — 3-tab egui application.
//!
//! Tab 1: Offline conversion (file → convert → save)
//! Tab 2: Real-time conversion (mic → VC → speaker)
//! Tab 3: Voice catalog (zero-shot reference management)

use std::sync::{Arc, Mutex};

use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;

/// Real-time metrics from inference thread.
#[derive(Clone, Debug, Default)]
pub struct RtMetrics {
    pub input_rms: f32,
    pub output_rms: f32,
    pub latency_ms: f32,
    pub rtf: f32,
    /// True when the audio device was lost ([07-4]). The UI should stop
    /// and return to the device-selection screen.
    pub disconnected: bool,
    /// Capture overrun count since start ([07-4]).
    pub overrun: u64,
    /// Playback underrun count since start ([07-4]).
    pub underrun: u64,
}

/// Control messages from UI to real-time inference thread.
pub enum RtControl {
    /// Start with explicit device selection ([05-6]).
    /// `input_idx` / `output_idx` are indices into the cpal device list
    /// (same order as `DuplexStream::list_input_devices()`). `None` = default.
    StartWithDevices {
        input_idx: Option<usize>,
        output_idx: Option<usize>,
    },
    Stop,
    SetMode(lightvc_core::converter::LatencyMode),
    Bypass(bool),
    LoadReference(Vec<f32>), // 44.1kHz mono PCM
}

/// Application-wide shared state.
pub struct AppState {
    pub dac_weights: std::path::PathBuf,
    pub converter_weights: Option<std::path::PathBuf>,
    pub converter_config: Option<std::path::PathBuf>,
    pub pipeline: Option<Arc<Mutex<lightvc_core::pipeline::VcPipeline>>>,
    pub voices: Vec<VoiceEntry>,
    pub selected_voice: Option<usize>,
    pub error: Option<String>,
    pub status: String,
    // Offline conversion result
    pub offline_result: Option<Vec<f32>>,
    // Real-time communication
    pub rt_control_tx: Option<Sender<RtControl>>,
    pub rt_metrics_rx: Option<Receiver<RtMetrics>>,
    pub rt_initialized: bool,
}

#[derive(Clone)]
pub struct VoiceEntry {
    pub name: String,
    pub path: std::path::PathBuf,
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
    file_dialog: egui_file_dialog::FileDialog,
    offline: crate::offline_tab::OfflineState,
    rt_running: bool,
    rt_bypass: bool,
    rt_mode: lightvc_core::converter::LatencyMode,
    rt_metrics: RtMetrics,
    /// Selected input device index (None = default) ([05-6]).
    rt_selected_input: Option<usize>,
    /// Selected output device index (None = default) ([05-6]).
    rt_selected_output: Option<usize>,
    conv_path_buf: String,
    conv_cfg_buf: String,
    asset_cache: crate::assets::AssetCache,
    splash_frames: u32, // 0 = showing splash, >0 = finished
}

impl LightVcApp {
    pub fn new(dac_weights: std::path::PathBuf) -> Self {
        let state = Arc::new(Mutex::new(AppState {
            dac_weights,
            converter_weights: None,
            converter_config: None,
            pipeline: None,
            voices: Vec::new(),
            selected_voice: None,
            error: None,
            status: "Ready".to_string(),
            offline_result: None,
            rt_control_tx: None,
            rt_metrics_rx: None,
            rt_initialized: false,
        }));

        Self {
            state,
            current_tab: Tab::Offline,
            file_dialog: egui_file_dialog::FileDialog::default(),
            offline: Default::default(),
            rt_running: false,
            rt_bypass: false,
            rt_mode: lightvc_core::converter::LatencyMode::Balanced,
            rt_metrics: RtMetrics::default(),
            rt_selected_input: None,
            rt_selected_output: None,
            conv_path_buf: String::new(),
            conv_cfg_buf: String::new(),
            asset_cache: Default::default(),
            splash_frames: 0,
        }
    }

    fn ensure_rt_thread_static(state: &Arc<Mutex<AppState>>) {
        let mut s = state.lock().unwrap();
        if s.rt_initialized {
            return;
        }

        let (control_tx, control_rx) = unbounded();
        let (metrics_tx, metrics_rx) = unbounded();
        let pipeline = s.pipeline.clone();

        s.rt_control_tx = Some(control_tx);
        s.rt_metrics_rx = Some(metrics_rx);
        s.rt_initialized = true;

        drop(s);

        // Spawn the inference thread even without a converter — it will
        // run in bypass mode, allowing the audio path to be tested.
        std::thread::spawn(move || {
            crate::realtime_tab::inference_loop(pipeline, control_rx, metrics_tx);
        });
    }

    fn load_converter_static(state: &Arc<Mutex<AppState>>, conv_path: &str, cfg_path: &str) {
        let dac_path = state.lock().unwrap().dac_weights.clone();
        let result = (|| -> anyhow::Result<()> {
            let device = candle_core::Device::Cpu;
            let dac_config = lightvc_core::DacConfig::default();

            let conv_config = if !cfg_path.is_empty() {
                let cfg_str = std::fs::read_to_string(cfg_path)?;
                serde_json::from_str(&cfg_str)?
            } else {
                lightvc_core::converter::ConverterConfig::default()
            };

            let vb = lightvc_core::weights::load_varbuilder(
                std::path::Path::new(conv_path),
                candle_core::DType::F32,
                &device,
            )?;
            let converter = lightvc_core::converter::AnyConverter::new(conv_config, vb)?;

            let pipeline = lightvc_core::pipeline::VcPipeline::new(
                &dac_path,
                &dac_config,
                converter,
                lightvc_core::converter::LatencyMode::Balanced,
                device,
            )?;

            let mut s = state.lock().unwrap();
            s.pipeline = Some(Arc::new(Mutex::new(pipeline)));
            s.converter_weights = Some(std::path::PathBuf::from(conv_path));
            s.converter_config = if cfg_path.is_empty() {
                None
            } else {
                Some(std::path::PathBuf::from(cfg_path))
            };
            s.status = "Converter loaded".to_string();
            s.error = None;
            Ok(())
        })();

        if let Err(e) = result {
            let mut s = state.lock().unwrap();
            s.error = Some(format!("Load failed: {e}"));
        }
    }

    fn send_control(state: &Arc<Mutex<AppState>>, ctrl: RtControl) {
        let s = state.lock().unwrap();
        if let Some(ref tx) = s.rt_control_tx {
            let _ = tx.send(ctrl);
        }
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
                .frame(
                    egui::Frame::NONE.fill(egui::Color32::from_rgba_premultiplied(28, 22, 38, 255)),
                )
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

        // Draw background texture
        {
            let bg = self.asset_cache.bg(ctx);
            let screen = ctx.content_rect();
            ctx.layer_painter(egui::LayerId::background()).image(
                bg.id(),
                screen,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }

        // Top bar with logo image + kawaii tabs
        egui::Panel::top("tabs")
            .frame(
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgba_premultiplied(28, 22, 38, 220))
                    .inner_margin(egui::Margin::same(12)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    // Logo image — responsive: max 140px, scales down on narrow
                    let logo = self.asset_cache.logo(ctx);
                    let avail = ui.available_width();
                    let logo_w = avail.min(140.0).max(80.0);
                    let logo_h = logo_w * (30.0 / 320.0); // maintain aspect ratio
                    ui.add(
                        egui::Image::from_texture(logo)
                            .fit_to_exact_size(egui::Vec2::new(logo_w, logo_h)),
                    );
                    ui.add_space(8.0);

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
                });
            });

        // Status bar
        {
            let st = self.state.lock().unwrap();
            egui::Panel::bottom("status")
                .frame(
                    egui::Frame::NONE
                        .fill(egui::Color32::from_rgba_premultiplied(42, 32, 56, 180))
                        .inner_margin(egui::Margin::same(8)),
                )
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        let (dot_color, msg) = if let Some(ref err) = st.error {
                            (egui::Color32::from_rgb(255, 100, 100), err.clone())
                        } else {
                            (crate::theme::colors::MINT, st.status.clone())
                        };
                        crate::theme::status_dot(ui, true, dot_color);
                        ui.label(
                            egui::RichText::new(&msg)
                                .size(12.0)
                                .color(crate::theme::colors::TEXT_DIM),
                        );
                    });
                });
        }

        // Tab content
        match self.current_tab {
            Tab::Offline => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let folder = self.asset_cache.icon_folder(ctx).clone();
                    let play = self.asset_cache.icon_play(ctx).clone();
                    let convert = self.asset_cache.icon_convert(ctx).clone();
                    let speaker = self.asset_cache.icon_speaker(ctx).clone();
                    let mic = self.asset_cache.icon_mic(ctx).clone();
                    crate::offline_tab::render(
                        ui,
                        ctx,
                        &mut self.file_dialog,
                        &self.state,
                        &mut self.offline,
                        &folder,
                        &play,
                        &convert,
                        &speaker,
                        &mic,
                    );
                });
            }
            Tab::Realtime => {
                // Receive metrics
                {
                    let s = self.state.lock().unwrap();
                    if let Some(ref rx) = s.rt_metrics_rx {
                        while let Ok(m) = rx.try_recv() {
                            // [07-4] device disconnection: the inference thread
                            // already tore down its streams; reflect that in the UI.
                            if m.disconnected {
                                self.rt_running = false;
                                self.state.lock().unwrap().status =
                                    "Audio device disconnected".to_string();
                            }
                            self.rt_metrics = m;
                        }
                    }
                }

                let state = self.state.clone();
                let mut conv_path = std::mem::take(&mut self.conv_path_buf);
                let mut conv_cfg = std::mem::take(&mut self.conv_cfg_buf);
                let mut rt_running = self.rt_running;
                let mut rt_bypass = self.rt_bypass;
                let mut rt_mode = self.rt_mode;
                let mut rt_sel_in = self.rt_selected_input;
                let mut rt_sel_out = self.rt_selected_output;
                let metrics = self.rt_metrics.clone();
                let file_dialog = &mut self.file_dialog;
                let knob_tex = self.asset_cache.knob(ctx);
                let knob_tex_ref = knob_tex.clone();
                let icon_stop_tex = self.asset_cache.icon_stop(ctx);
                let icon_stop_tex_ref = icon_stop_tex.clone();

                egui::CentralPanel::default().show(ctx, |ui| {
                    crate::realtime_tab::render(
                        ui,
                        ctx,
                        file_dialog,
                        &state,
                        &mut conv_path,
                        &mut conv_cfg,
                        &mut rt_running,
                        &mut rt_bypass,
                        &mut rt_mode,
                        &mut rt_sel_in,
                        &mut rt_sel_out,
                        &metrics,
                        Some(&knob_tex_ref),
                        Some(&icon_stop_tex_ref),
                        |c, cfg| Self::load_converter_static(&state, c, cfg),
                        || Self::ensure_rt_thread_static(&state),
                        |ctrl| Self::send_control(&state, ctrl),
                    );
                });

                self.conv_path_buf = conv_path;
                self.conv_cfg_buf = conv_cfg;
                self.rt_running = rt_running;
                self.rt_bypass = rt_bypass;
                self.rt_mode = rt_mode;
                self.rt_selected_input = rt_sel_in;
                self.rt_selected_output = rt_sel_out;
            }
            Tab::Catalog => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let folder = self.asset_cache.icon_folder(ctx).clone();
                    let play = self.asset_cache.icon_play(ctx).clone();
                    let trash = self.asset_cache.icon_trash(ctx).clone();
                    let empty = self.asset_cache.empty_stars(ctx).clone();
                    let state = self.state.clone();
                    crate::voice_catalog::render(
                        ui,
                        ctx,
                        &mut self.file_dialog,
                        &state,
                        &folder,
                        &play,
                        &trash,
                        &empty,
                        |idx| {
                            // Load the selected voice as the Realtime reference.
                            let mut s = state.lock().unwrap();
                            if let Some(voice) = s.voices.get(idx).cloned() {
                                s.selected_voice = Some(idx);
                                if let Ok((wav, sr)) =
                                    crate::audio_playback::load_wav_mono(&voice.path)
                                {
                                    let wav44 = crate::audio_playback::resample_linear(&wav, sr);
                                    Self::send_control(&state, RtControl::LoadReference(wav44));
                                }
                            }
                        },
                    );
                });
            }
        }

        // Only request continuous repaint during realtime mode
        if self.current_tab == Tab::Realtime {
            ctx.request_repaint();
        }
    }
}

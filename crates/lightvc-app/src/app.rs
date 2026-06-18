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
    /// Current effective mode ([F5]). May differ from the user-selected
    /// mode when auto-degradation kicked in.
    pub current_mode: lightvc_core::converter::LatencyMode,
    /// True when the mode was auto-downgraded due to underruns ([F5]).
    pub auto_degraded: bool,
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
    SetProsody {
        mode: lightvc_core::converter::ProsodyMode,
        blend: f64,
    },
    SetVelocityScale(f64),
    Bypass(bool),
    LoadReference(Vec<f32>), // 44.1kHz mono PCM
}

/// Type alias for the shared pipeline slot.
pub type PipelineSlot = Arc<Mutex<Option<Arc<Mutex<lightvc_core::pipeline::VcPipeline>>>>>;

/// Application-wide shared state.
pub struct AppState {
    pub dac_weights: std::path::PathBuf,
    pub converter_weights: Option<std::path::PathBuf>,
    pub converter_config: Option<std::path::PathBuf>,
    pub pipeline: Option<Arc<Mutex<lightvc_core::pipeline::VcPipeline>>>,
    /// Shared hot-swappable pipeline slot. The inference thread reads this
    /// every loop iteration so a converter loaded after thread start is
    /// picked up without restarting the thread ([F2]).
    pub pipeline_slot: PipelineSlot,
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
    offline: crate::offline_tab::OfflineState,
    catalog: crate::voice_catalog::CatalogState,
    rt_running: bool,
    rt_bypass: bool,
    rt_mode: lightvc_core::converter::LatencyMode,
    rt_prosody_mode: lightvc_core::converter::ProsodyMode,
    rt_prosody_blend: f32,
    rt_velocity_scale: f32,
    /// Demo mode for screenshot capture (None = normal operation).
    demo: Option<crate::cli::DemoState>,
    rt_metrics: RtMetrics,
    /// Selected input device index (None = default) ([05-6]).
    rt_selected_input: Option<usize>,
    /// Selected output device index (None = default) ([05-6]).
    rt_selected_output: Option<usize>,
    conv_path_buf: String,
    conv_cfg_buf: String,
    /// File pickers for the Realtime converter/config fields.
    rt_converter_pick: crate::file_pick::FilePick,
    rt_config_pick: crate::file_pick::FilePick,
    /// File pickers for the Catalog add/import actions.
    catalog_add_pick: crate::file_pick::FilePick,
    catalog_import_pick: crate::file_pick::FilePick,
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
            pipeline_slot: Arc::new(Mutex::new(None)),
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
            offline: crate::offline_tab::OfflineState {
                prosody_blend: 0.5,
                velocity_scale: 1.0,
                ..Default::default()
            },
            catalog: Default::default(),
            rt_running: false,
            rt_bypass: false,
            rt_mode: lightvc_core::converter::LatencyMode::Balanced,
            rt_prosody_mode: lightvc_core::converter::ProsodyMode::default(),
            rt_prosody_blend: 0.5,
            rt_velocity_scale: 1.0,
            rt_metrics: RtMetrics::default(),
            rt_selected_input: None,
            rt_selected_output: None,
            conv_path_buf: String::new(),
            conv_cfg_buf: String::new(),
            rt_converter_pick: Default::default(),
            rt_config_pick: Default::default(),
            catalog_add_pick: Default::default(),
            catalog_import_pick: Default::default(),
            asset_cache: Default::default(),
            splash_frames: 0,
            demo: None,
        }
    }

    /// Enable demo mode for screenshot capture. Injects mock data so the
    /// GUI renders fully without a model or audio devices.
    pub fn enable_demo(&mut self, demo: crate::cli::DemoState) {
        self.demo = Some(demo);
        // Skip the splash animation so the screenshot is immediate.
        self.splash_frames = 60;
        // Pretend a converter is loaded so tabs show their full UI.
        {
            let mut s = self.state.lock().unwrap();
            s.converter_weights = Some(std::path::PathBuf::from("models/converter.safetensors"));
            s.converter_config = Some(std::path::PathBuf::from("configs/phase_c.yaml"));
            // Register sample voices for the Catalog tab.
            s.voices = vec![
                crate::app::VoiceEntry {
                    name: "Venus (warm)".into(),
                    path: "samples/venus.wav".into(),
                },
                crate::app::VoiceEntry {
                    name: "Mars (bright)".into(),
                    path: "samples/mars.wav".into(),
                },
                crate::app::VoiceEntry {
                    name: "Lyra (soft)".into(),
                    path: "samples/lyra.wav".into(),
                },
            ];
            s.selected_voice = Some(0);
            s.status = "Demo mode".into();
        }
        match demo {
            crate::cli::DemoState::Offline => {
                self.current_tab = Tab::Offline;
                self.offline.source_path = "samples/source_male.wav".into();
                self.offline.reference_path = "samples/venus.wav".into();
                self.offline.prosody_mode = lightvc_core::converter::ProsodyMode::Blend;
                self.offline.prosody_blend = 0.4;
                self.offline.velocity_scale = 1.0;
                self.offline.converted_samples = Some(vec![0.0; 44_100]);
            }
            crate::cli::DemoState::Realtime => {
                self.current_tab = Tab::Realtime;
                self.rt_running = true;
                self.rt_bypass = false;
                self.rt_mode = lightvc_core::converter::LatencyMode::Balanced;
                self.rt_prosody_mode = lightvc_core::converter::ProsodyMode::ImitateTarget;
                self.rt_prosody_blend = 0.5;
                self.rt_velocity_scale = 1.0;
                self.rt_metrics = RtMetrics {
                    input_rms: 0.18,
                    output_rms: 0.12,
                    latency_ms: 46.0,
                    rtf: 0.34,
                    disconnected: false,
                    overrun: 0,
                    underrun: 2,
                    current_mode: lightvc_core::converter::LatencyMode::Balanced,
                    auto_degraded: false,
                };
            }
            crate::cli::DemoState::Catalog => {
                self.current_tab = Tab::Catalog;
            }
        }
    }

    fn ensure_rt_thread_static(state: &Arc<Mutex<AppState>>) {
        let mut s = state.lock().unwrap();
        if s.rt_initialized {
            return;
        }

        let (control_tx, control_rx) = unbounded();
        let (metrics_tx, metrics_rx) = unbounded();
        // Share the pipeline slot itself (Arc<Mutex<Option<Arc<Mutex<…>>>>>)
        // so that load_converter_static can swap it in after the thread is
        // already running. Previously the thread captured a clone of the
        // Option at spawn time, so a converter loaded later was invisible.
        let pipeline_slot = s.pipeline_slot.clone();

        s.rt_control_tx = Some(control_tx);
        s.rt_metrics_rx = Some(metrics_rx);
        s.rt_initialized = true;

        drop(s);

        // Spawn the inference thread even without a converter — it will
        // run in bypass mode, allowing the audio path to be tested.
        std::thread::spawn(move || {
            crate::realtime_tab::inference_loop(pipeline_slot, control_rx, metrics_tx);
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
            let pipeline_arc = Arc::new(Mutex::new(pipeline));
            s.pipeline = Some(pipeline_arc.clone());
            *s.pipeline_slot.lock().unwrap() = Some(pipeline_arc);
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
                            // [F5] sync auto-degraded mode to the knob.
                            self.rt_mode = m.current_mode;
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
                let mut rt_prosody_mode = self.rt_prosody_mode;
                let mut rt_prosody_blend = self.rt_prosody_blend;
                let mut rt_velocity_scale = self.rt_velocity_scale;
                let mut rt_sel_in = self.rt_selected_input;
                let mut rt_sel_out = self.rt_selected_output;
                let metrics = self.rt_metrics.clone();
                let converter_pick = self.rt_converter_pick.clone();
                let config_pick = self.rt_config_pick.clone();
                let knob_tex = self.asset_cache.knob(ctx);
                let knob_tex_ref = knob_tex.clone();
                let icon_stop_tex = self.asset_cache.icon_stop(ctx);
                let icon_stop_tex_ref = icon_stop_tex.clone();

                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            crate::realtime_tab::render(
                                ui,
                                ctx,
                                &converter_pick,
                                &config_pick,
                                &state,
                                &mut conv_path,
                                &mut conv_cfg,
                                &mut rt_running,
                                &mut rt_bypass,
                                &mut rt_mode,
                                &mut rt_prosody_mode,
                                &mut rt_prosody_blend,
                                &mut rt_velocity_scale,
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
                });

                self.conv_path_buf = conv_path;
                self.conv_cfg_buf = conv_cfg;
                self.rt_running = rt_running;
                self.rt_bypass = rt_bypass;
                self.rt_mode = rt_mode;
                self.rt_prosody_mode = rt_prosody_mode;
                self.rt_prosody_blend = rt_prosody_blend;
                self.rt_velocity_scale = rt_velocity_scale;
                self.rt_selected_input = rt_sel_in;
                self.rt_selected_output = rt_sel_out;
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
                                    // Load the selected voice as the Realtime reference.
                                    let mut s = state.lock().unwrap();
                                    if let Some(voice) = s.voices.get(idx).cloned() {
                                        s.selected_voice = Some(idx);
                                        if let Ok((wav, sr)) =
                                            crate::audio_playback::load_wav_mono(&voice.path)
                                        {
                                            let wav44 =
                                                crate::audio_playback::resample_linear(&wav, sr);
                                            Self::send_control(
                                                &state,
                                                RtControl::LoadReference(wav44),
                                            );
                                        }
                                    }
                                },
                            );
                        });
                });
                self.catalog = catalog;
            }
        }

        // Only request continuous repaint during realtime mode, or while a
        // Catalog preview is playing (so the Stop button reverts to Play on
        // natural finish).
        if self.current_tab == Tab::Realtime
            || (self.current_tab == Tab::Catalog && self.catalog.playing_voice.is_some())
        {
            ctx.request_repaint();
        }
    }
}

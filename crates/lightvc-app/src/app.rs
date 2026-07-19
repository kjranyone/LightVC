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
    /// Algorithmic latency of the current chunk (ms). Exposed separately so
    /// the Latency card can render the E2E breakdown; the fixed buffer/resample
    /// terms are derived UI-side from the selected buffer size.
    pub algo_ms: f32,
}

/// Control messages from UI to real-time inference thread.
pub enum RtControl {
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
    LoadReference(Vec<f32>),
    /// Not yet sent from UI; the receiver in `inference_loop` calls
    /// `B1Streaming::set_timbre`. Pending a timbre-file load button.
    #[allow(dead_code)]
    SetB1Timbre(candle_core::Tensor),
    SetB1Tau(f64),
    SetWetDry(f32),
    /// Mute the output (silence) while keeping the stream armed. Distinct from
    /// Bypass (which passes the dry signal through).
    Mute(bool),
    /// Fixed capture/playback buffer size in frames. Applied to the next
    /// engine Start; the UI re-arms a running stream so it takes effect.
    SetBufferSize(u32),
}

/// Type alias for the shared pipeline slot.
pub type PipelineSlot = Arc<Mutex<Option<Arc<Mutex<lightvc_core::Backend>>>>>;

/// Application-wide shared state.
pub struct AppState {
    pub dac_weights: std::path::PathBuf,
    pub converter_weights: Option<std::path::PathBuf>,
    pub converter_config: Option<std::path::PathBuf>,
    pub pipeline: Option<Arc<Mutex<lightvc_core::Backend>>>,
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
    demo: bool,
    rt_metrics: RtMetrics,
    conv_path_buf: String,
    /// Inline-editable converter config (field-by-field, no JSON file).
    rt_config: lightvc_core::converter::ConverterConfig,
    /// File picker for the Realtime converter field.
    rt_converter_pick: crate::file_pick::FilePick,
    /// File pickers for the Catalog add/import actions.
    catalog_add_pick: crate::file_pick::FilePick,
    catalog_import_pick: crate::file_pick::FilePick,
    asset_cache: crate::assets::AssetCache,
    splash_frames: u32, // 0 = showing splash, >0 = finished
    settings_open: bool,
    // B1 adapter UI state
    b1_adapter_path: String,
    b1_quantizer_path: String,
    b1_timbre_path: String,
    b1_tau: f32,
    wet_dry: f32,
    /// Output mute (Transport). Silences the wet signal without unarming.
    rt_muted: bool,
    /// Selected fixed buffer size in frames (128/256/512/1024). 256 ≈ paravo.
    rt_buffer_frames: u32,
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
            selected_input: None,
            selected_output: None,
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
            conv_path_buf: String::new(),
            rt_config: lightvc_core::converter::ConverterConfig::default(),
            rt_converter_pick: Default::default(),
            catalog_add_pick: Default::default(),
            catalog_import_pick: Default::default(),
            asset_cache: Default::default(),
            splash_frames: 0,
            settings_open: false,
            demo: false,
            b1_adapter_path: "models/utte_adapter_b1.safetensors".into(),
            b1_quantizer_path: "models/dac_quantizer.safetensors".into(),
            b1_timbre_path: String::new(),
            b1_tau: 5.0,
            wet_dry: 1.0,
            rt_muted: false,
            rt_buffer_frames: 256,
        }
    }

    /// Enable demo mode for screenshot capture. Injects mock data so all
    /// three tabs render fully without a model or audio devices. The user
    /// switches tabs inside the app.
    pub fn enable_demo(&mut self) {
        self.demo = true;
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
        // Offline tab mock data.
        self.offline.source_path = "samples/source_male.wav".into();
        self.offline.reference_path = "samples/venus.wav".into();
        self.offline.prosody_mode = lightvc_core::converter::ProsodyMode::Blend;
        self.offline.prosody_blend = 0.4;
        self.offline.velocity_scale = 1.0;
        self.offline.converted_samples = Some(vec![0.0; 44_100]);
        // Realtime tab mock data.
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
            algo_ms: 20.0,
        };
        // Initial tab: Realtime (most informative for screenshots).
        self.current_tab = Tab::Realtime;
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

    fn load_converter_static(
        state: &Arc<Mutex<AppState>>,
        conv_path: &str,
        conv_config: lightvc_core::converter::ConverterConfig,
    ) {
        let dac_path = state.lock().unwrap().dac_weights.clone();
        let result = (|| -> anyhow::Result<()> {
            let device = candle_core::Device::Cpu;
            let dac_config = lightvc_core::DacConfig::default();

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
            let pipeline_arc = Arc::new(Mutex::new(lightvc_core::Backend::Legacy(pipeline)));
            s.pipeline = Some(pipeline_arc.clone());
            *s.pipeline_slot.lock().unwrap() = Some(pipeline_arc);
            s.converter_weights = Some(std::path::PathBuf::from(conv_path));
            s.status = "Converter loaded".to_string();
            s.error = None;
            Ok(())
        })();

        if let Err(e) = result {
            let mut s = state.lock().unwrap();
            s.error = Some(format!("Load failed: {e}"));
        }
    }

    fn load_b1_static(
        state: &Arc<Mutex<AppState>>,
        dac_path: &str,
        quantizer_path: &str,
        adapter_path: &str,
        timbre_path: &str,
    ) {
        let result = (|| -> anyhow::Result<()> {
            let device = candle_core::Device::Cpu;

            let timbre = {
                let vb = lightvc_core::weights::load_varbuilder(
                    std::path::Path::new(timbre_path),
                    candle_core::DType::F32,
                    &device,
                )?;
                vb.get((1, 192), "timbre")?
            };

            let mut b1 = lightvc_core::b1_pipeline::B1Streaming::new(
                std::path::Path::new(dac_path),
                std::path::Path::new(quantizer_path),
                std::path::Path::new(adapter_path),
                lightvc_core::streaming::ChunkMode::Balanced,
                device,
            )?;
            b1.set_timbre(timbre);

            let mut s = state.lock().unwrap();
            let arc = Arc::new(Mutex::new(lightvc_core::Backend::B1(b1)));
            s.pipeline = Some(arc.clone());
            *s.pipeline_slot.lock().unwrap() = Some(arc);
            s.status = "B1 adapter loaded".to_string();
            s.error = None;
            Ok(())
        })();

        if let Err(e) = result {
            let mut s = state.lock().unwrap();
            s.error = Some(format!("B1 load failed: {e}"));
        }
    }

    /// Load the FreeVocoder resynthesis backend (mic → mel → freeC vocoder →
    /// out). `voc_path` = freeC vocoder weights, `mel_basis_path` = librosa
    /// slaney mel filterbank (`mel_basis` key). `k` = mel frames per chunk.
    #[allow(dead_code)]
    fn load_freevoc_static(
        state: &Arc<Mutex<AppState>>,
        voc_path: &str,
        mel_basis_path: &str,
        k: usize,
    ) {
        let result = (|| -> anyhow::Result<()> {
            let device = candle_core::Device::Cpu;
            let resynth = lightvc_core::free_resynth::FreeResynth::new(
                std::path::Path::new(voc_path),
                std::path::Path::new(mel_basis_path),
                k,
                device,
            )?;

            let mut s = state.lock().unwrap();
            let arc = Arc::new(Mutex::new(lightvc_core::Backend::FreeVoc(resynth)));
            s.pipeline = Some(arc.clone());
            *s.pipeline_slot.lock().unwrap() = Some(arc);
            s.status = "FreeVocoder resynthesis loaded".to_string();
            s.error = None;
            Ok(())
        })();

        if let Err(e) = result {
            let mut s = state.lock().unwrap();
            s.error = Some(format!("FreeVoc load failed: {e}"));
        }
    }

    fn send_control(state: &Arc<Mutex<AppState>>, ctrl: RtControl) {
        let s = state.lock().unwrap();
        if let Some(ref tx) = s.rt_control_tx {
            let _ = tx.send(ctrl);
        }
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
                let mut rt_config = self.rt_config.clone();
                let mut rt_running = self.rt_running;
                let mut rt_bypass = self.rt_bypass;
                let mut rt_mode = self.rt_mode;
                let mut rt_prosody_mode = self.rt_prosody_mode;
                let mut rt_prosody_blend = self.rt_prosody_blend;
                let mut rt_velocity_scale = self.rt_velocity_scale;
                let mut b1_adapter_path = self.b1_adapter_path.clone();
                let mut b1_quantizer_path = self.b1_quantizer_path.clone();
                let mut b1_timbre_path = self.b1_timbre_path.clone();
                let mut b1_tau = self.b1_tau;
                let mut wet_dry = self.wet_dry;
                let mut rt_muted = self.rt_muted;
                let mut rt_buffer_frames = self.rt_buffer_frames;
                let metrics = self.rt_metrics.clone();
                let converter_pick = self.rt_converter_pick.clone();

                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            crate::realtime_tab::render(
                                ui,
                                ctx,
                                &converter_pick,
                                &state,
                                &mut conv_path,
                                &mut rt_config,
                                &mut rt_running,
                                &mut rt_bypass,
                                &mut rt_mode,
                                &mut rt_prosody_mode,
                                &mut rt_prosody_blend,
                                &mut rt_velocity_scale,
                                &mut rt_muted,
                                &mut rt_buffer_frames,
                                &metrics,
                                |c, cfg| Self::load_converter_static(&state, c, cfg.clone()),
                                || Self::ensure_rt_thread_static(&state),
                                |ctrl| Self::send_control(&state, ctrl),
                                &mut self.asset_cache,
                            );

                            ui.separator();
                            ui.collapsing("B1 Adapter (UTTE)", |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Adapter:");
                                    ui.text_edit_singleline(&mut b1_adapter_path);
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Quantizer:");
                                    ui.text_edit_singleline(&mut b1_quantizer_path);
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Timbre:");
                                    ui.text_edit_singleline(&mut b1_timbre_path);
                                });

                                let dac_path = state
                                    .lock()
                                    .unwrap()
                                    .dac_weights
                                    .to_string_lossy()
                                    .to_string();
                                let load_enabled = !b1_timbre_path.is_empty();
                                ui.add_enabled_ui(load_enabled, |ui| {
                                    if ui.button("Load B1 Adapter").clicked() {
                                        Self::load_b1_static(
                                            &state,
                                            &dac_path,
                                            &b1_quantizer_path,
                                            &b1_adapter_path,
                                            &b1_timbre_path,
                                        );
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Tau:");
                                    if ui
                                        .add(egui::Slider::new(&mut b1_tau, 0.1..=10.0).text(""))
                                        .changed()
                                    {
                                        Self::send_control(
                                            &state,
                                            RtControl::SetB1Tau(b1_tau as f64),
                                        );
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Wet/Dry:");
                                    if ui
                                        .add(egui::Slider::new(&mut wet_dry, 0.0..=1.0).text(""))
                                        .changed()
                                    {
                                        Self::send_control(&state, RtControl::SetWetDry(wet_dry));
                                    }
                                });

                                let is_b1 = state
                                    .lock()
                                    .unwrap()
                                    .pipeline
                                    .as_ref()
                                    .map(|p| p.lock().map(|p| p.is_b1()).unwrap_or(false))
                                    .unwrap_or(false);
                                if is_b1 {
                                    ui.colored_label(egui::Color32::GREEN, "● B1 adapter active");
                                }
                            });
                        });
                });

                self.conv_path_buf = conv_path;
                self.rt_config = rt_config;
                self.rt_running = rt_running;
                self.rt_bypass = rt_bypass;
                self.rt_mode = rt_mode;
                self.rt_prosody_mode = rt_prosody_mode;
                self.rt_prosody_blend = rt_prosody_blend;
                self.rt_velocity_scale = rt_velocity_scale;
                self.b1_adapter_path = b1_adapter_path;
                self.b1_quantizer_path = b1_quantizer_path;
                self.b1_timbre_path = b1_timbre_path;
                self.b1_tau = b1_tau;
                self.wet_dry = wet_dry;
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

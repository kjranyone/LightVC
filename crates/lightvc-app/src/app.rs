//! LightVC-X GUI — 3-tab egui application.
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
}

/// Control messages from UI to real-time inference thread.
pub enum RtControl {
    Start,
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

#[derive(PartialEq)]
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
    // Converter load UI fields
    conv_path_buf: String,
    conv_cfg_buf: String,
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
            conv_path_buf: String::new(),
            conv_cfg_buf: String::new(),
        }
    }

    /// Spawn the real-time inference thread (once).
    fn ensure_rt_thread(&mut self) {
        Self::ensure_rt_thread_static(&self.state);
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

        if let Some(pipeline) = pipeline {
            std::thread::spawn(move || {
                crate::realtime_tab::inference_loop(pipeline, control_rx, metrics_tx);
            });
        }
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
    pub fn render(&mut self, ctx: &egui::Context) {
        // Apply kawaii theme every frame (cheap, ensures consistency)
        crate::theme::apply_theme(ctx);
        crate::theme::gradient_background(ctx);

        // Top bar with kawaii tabs
        egui::TopBottomPanel::top("tabs")
            .frame(
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgba_premultiplied(42, 32, 56, 200))
                    .inner_margin(egui::Margin::same(12)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    // Logo
                    ui.label(
                        egui::RichText::new("✦ LightVC-X")
                            .size(18.0)
                            .strong()
                            .color(crate::theme::colors::PINK_BRIGHT),
                    );
                    ui.add_space(24.0);

                    if crate::theme::tab_button(ui, "Offline", self.current_tab == Tab::Offline) {
                        self.current_tab = Tab::Offline;
                    }
                    if crate::theme::tab_button(ui, "Realtime", self.current_tab == Tab::Realtime) {
                        self.current_tab = Tab::Realtime;
                    }
                    if crate::theme::tab_button(ui, "Voices", self.current_tab == Tab::Catalog) {
                        self.current_tab = Tab::Catalog;
                    }
                });
            });

        // Status bar
        {
            let st = self.state.lock().unwrap();
            egui::TopBottomPanel::bottom("status")
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
                    crate::offline_tab::render(
                        ui,
                        ctx,
                        &mut self.file_dialog,
                        &self.state,
                        &mut self.offline,
                    );
                });
            }
            Tab::Realtime => {
                // Receive metrics
                {
                    let s = self.state.lock().unwrap();
                    if let Some(ref rx) = s.rt_metrics_rx {
                        while let Ok(m) = rx.try_recv() {
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
                let metrics = self.rt_metrics.clone();
                let file_dialog = &mut self.file_dialog;

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
                        &metrics,
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
            }
            Tab::Catalog => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    crate::voice_catalog::render(ui, ctx, &mut self.file_dialog, &self.state);
                });
            }
        }

        ctx.request_repaint();
    }
}

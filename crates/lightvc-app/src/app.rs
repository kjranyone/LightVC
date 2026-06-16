//! LightVC-X GUI — 3-tab egui application.
//!
//! Tab 1: Offline conversion (file → convert → save)
//! Tab 2: Real-time conversion (mic → VC → speaker)
//! Tab 3: Voice catalog (zero-shot reference management)

use std::sync::{Arc, Mutex};

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;

/// Application-wide shared state.
pub struct AppState {
    /// Model weights path (DAC + converter).
    pub dac_weights: std::path::PathBuf,
    pub converter_weights: Option<std::path::PathBuf>,
    pub converter_config: Option<std::path::PathBuf>,
    /// Loaded pipeline (lazy-initialized when converter is set).
    pub pipeline: Option<Arc<Mutex<lightvc_core::pipeline::VcPipeline>>>,
    /// Voice catalog: name → reference WAV path.
    pub voices: Vec<VoiceEntry>,
    /// Currently selected voice index.
    pub selected_voice: Option<usize>,
    /// Error message to display.
    pub error: Option<String>,
    /// Status message.
    pub status: String,
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
        }));

        Self {
            state,
            current_tab: Tab::Offline,
            file_dialog: egui_file_dialog::FileDialog::default(),
            offline: Default::default(),
        }
    }
}

impl LightVcApp {
    pub fn render(&mut self, ctx: &egui::Context) {
        // Top bar with tabs
        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.current_tab, Tab::Offline, "Offline Convert");
                ui.selectable_value(&mut self.current_tab, Tab::Realtime, "Realtime");
                ui.selectable_value(&mut self.current_tab, Tab::Catalog, "Voice Catalog");
            });
        });

        // Status bar
        {
            let st = self.state.lock().unwrap();
            egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let status_color = if st.error.is_some() {
                        egui::Color32::from_rgb(220, 80, 80)
                    } else {
                        egui::Color32::from_rgb(120, 180, 120)
                    };
                    let msg = st.error.clone().unwrap_or_else(|| st.status.clone());
                    ui.colored_label(status_color, &msg);
                });
            });
        }

        // Tab content
        egui::CentralPanel::default().show(ctx, |ui| match self.current_tab {
            Tab::Offline => {
                crate::offline_tab::render(
                    ui,
                    ctx,
                    &mut self.file_dialog,
                    &self.state,
                    &mut self.offline,
                );
            }
            Tab::Realtime => {
                crate::realtime_tab::render(ui, ctx, &self.state);
            }
            Tab::Catalog => {
                crate::voice_catalog::render(ui, ctx, &mut self.file_dialog, &self.state);
            }
        });

        ctx.request_repaint();
    }
}

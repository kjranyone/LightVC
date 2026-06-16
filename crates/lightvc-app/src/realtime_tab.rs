//! Tab 2: Real-time voice conversion.
//! Mic input → DAC encode → converter → DAC decode → speaker output.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;

use crate::widgets;
use crate::app::AppState;

#[derive(Clone, Debug, Default)]
struct Metrics {
    input_rms: f32,
    output_rms: f32,
    latency_ms: f32,
    rtf: f32,
}

enum ControlMsg {
    Start,
    Stop,
    SetMode(lightvc_core::converter::LatencyMode),
    Bypass(bool),
}

pub fn render(ui: &mut egui::Ui, ctx: &egui::Context, state: &Arc<Mutex<AppState>>) {
    let metrics_rx = ctx.data_mut(|d| {
        d.get_temp_mut_or_insert_with::<Option<Receiver<Metrics>>>("rt_metrics_rx".into(), || None)
            .clone()
    });

    let control_tx = ctx.data_mut(|d| {
        d.get_temp_mut_or_insert_with::<Option<Sender<ControlMsg>>>("rt_control_tx".into(), || None)
            .clone()
    });

    let running = ctx.data_mut(|d| {
        d.get_temp_mut_or_insert_with("rt_running".into(), || false)
            .clone()
    });

    let bypass = ctx.data_mut(|d| {
        d.get_temp_mut_or_insert_with("rt_bypass".into(), || false)
            .clone()
    });

    let selected_mode = ctx.data_mut(|d| {
        d.get_temp_mut_or_insert_with("rt_mode".into(), || {
            lightvc_core::converter::LatencyMode::Balanced
        })
        .clone()
    });

    let mut metrics = Metrics::default();
    if let Some(ref rx) = metrics_rx {
        while let Ok(m) = rx.try_recv() {
            metrics = m;
        }
    }

    ui.heading("Real-time Voice Conversion");
    ui.add_space(8.0);

    // Model setup
    let (has_converter, has_pipeline) = {
        let s = state.lock().unwrap();
        (s.converter_weights.is_some(), s.pipeline.is_some())
    };

    if !has_converter {
        ui.horizontal(|ui| {
            ui.label("Converter weights:");
            if ui.button("Browse...").clicked() {
                // TODO: file dialog for converter weights
            }
        });
        ui.colored_label(
            egui::Color32::from_rgb(200, 180, 80),
            "Load converter weights to enable VC.",
        );
    } else {
        // Status
        ui.horizontal(|ui| {
            widgets::status_dot(ui, running);
            let label = if running {
                if bypass {
                    "BYPASS"
                } else {
                    "LIVE"
                }
            } else {
                "STOPPED"
            };
            let color = if running {
                if bypass {
                    egui::Color32::from_rgb(200, 200, 80)
                } else {
                    egui::Color32::from_rgb(80, 200, 80)
                }
            } else {
                egui::Color32::from_rgb(160, 160, 160)
            };
            ui.colored_label(color, label);
        });

        ui.add_space(8.0);

        // Level meters
        widgets::level_meter(ui, metrics.input_rms, "Input");
        widgets::level_meter(ui, metrics.output_rms, "Output");

        ui.add_space(4.0);
        ui.label(format!(
            "Latency: {:.0} ms | RTF: {:.2}",
            metrics.latency_ms, metrics.rtf
        ));

        ui.add_space(8.0);

        // Quality mode
        ui.horizontal(|ui| {
            ui.label("Mode:");
            let mut mode = selected_mode;
            ui.radio_value(
                &mut mode,
                lightvc_core::converter::LatencyMode::Strict,
                "Strict",
            );
            ui.radio_value(
                &mut mode,
                lightvc_core::converter::LatencyMode::Balanced,
                "Balanced",
            );
            ui.radio_value(
                &mut mode,
                lightvc_core::converter::LatencyMode::Quality,
                "Quality",
            );
            if mode != selected_mode {
                let _ = control_tx
                    .as_ref()
                    .map(|tx| tx.send(ControlMsg::SetMode(mode)));
            }
        });

        ui.add_space(8.0);

        // Bypass + Start/Stop
        let mut bp = bypass;
        ui.checkbox(&mut bp, "Bypass (monitor only)");
        if bp != bypass {
            let _ = control_tx
                .as_ref()
                .map(|tx| tx.send(ControlMsg::Bypass(bp)));
        }

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if !running {
                ui.add_enabled_ui(has_pipeline, |ui| {
                    if ui.button("▶ Start").clicked() {
                        let _ = control_tx.as_ref().map(|tx| tx.send(ControlMsg::Start));
                    }
                });
            } else {
                if ui.button("■ Stop").clicked() {
                    let _ = control_tx.as_ref().map(|tx| tx.send(ControlMsg::Stop));
                }
            }
        });

        // Audio devices
        ui.add_space(12.0);
        ui.collapsing("Audio Devices", |ui| {
            let inputs = lightvc_audio::DuplexStream::list_input_devices().unwrap_or_default();
            let outputs = lightvc_audio::DuplexStream::list_output_devices().unwrap_or_default();
            ui.label("Inputs:");
            for d in &inputs {
                ui.label(format!(
                    "  {} ({}Hz, {}ch)",
                    d.name, d.sample_rate, d.channels
                ));
            }
            ui.label("Outputs:");
            for d in &outputs {
                ui.label(format!(
                    "  {} ({}Hz, {}ch)",
                    d.name, d.sample_rate, d.channels
                ));
            }
            // Show ASIO availability
            let host = cpal::default_host();
            ui.label(format!("Host: {}", host.id().name()));
        });
    }

    // Persist state
    ctx.data_mut(|d| {
        d.insert_temp("rt_running".into(), running);
        d.insert_temp("rt_bypass".into(), bypass);
        d.insert_temp("rt_mode".into(), selected_mode);
    });
}

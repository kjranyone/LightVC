//! Tab 2: Real-time voice conversion.
//! Mic input → DAC encode → converter → DAC decode → speaker output.
//!
//! Contains both UI rendering and the inference thread loop.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use egui_file_dialog::FileDialog;

use crate::app::AppState;
use crate::app::{RtControl, RtMetrics};
use crate::widgets;

/// Render the realtime tab.
#[allow(clippy::too_many_arguments)]
pub fn render(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    file_dialog: &mut FileDialog,
    state: &Arc<Mutex<AppState>>,
    conv_path: &mut String,
    conv_cfg: &mut String,
    running: &mut bool,
    bypass: &mut bool,
    mode: &mut lightvc_core::converter::LatencyMode,
    metrics: &RtMetrics,
    knob_tex: Option<&egui::TextureHandle>,
    mut on_load: impl FnMut(&str, &str),
    mut on_ensure_thread: impl FnMut(),
    on_control: impl Fn(RtControl),
) {
    ui.heading("Real-time Voice Conversion");
    ui.add_space(8.0);

    let has_pipeline = state.lock().unwrap().pipeline.is_some();

    // --- Model Setup Section ---
    if !has_pipeline {
        crate::theme::info_card(ui, |ui| {
            crate::theme::heading(ui, "Load Model");

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Converter")
                        .size(13.0)
                        .color(crate::theme::colors::TEXT_DIM),
                );
                ui.text_edit_singleline(conv_path);
                if crate::theme::pill_button(ui, "Browse", false) {
                    file_dialog.pick_file();
                    ctx.data_mut(|d| d.insert_temp("rt_pick".into(), "converter"));
                }
            });

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Config")
                        .size(13.0)
                        .color(crate::theme::colors::TEXT_DIM),
                );
                ui.text_edit_singleline(conv_cfg);
                if crate::theme::pill_button(ui, "Browse", false) {
                    file_dialog.pick_file();
                    ctx.data_mut(|d| d.insert_temp("rt_pick".into(), "config"));
                }
            });

            ui.add_space(4.0);
            if crate::theme::pill_button(ui, "Load Converter", !conv_path.is_empty())
                && !conv_path.is_empty()
            {
                on_load(conv_path, conv_cfg);
            }
        });

        // Handle file dialog
        if let Some(path) = file_dialog.take_picked() {
            let pick = ctx.data_mut(|d| d.get_temp::<String>("rt_pick".into()).unwrap_or_default());
            match pick.as_str() {
                "converter" => *conv_path = path.to_string_lossy().into_owned(),
                "config" => *conv_cfg = path.to_string_lossy().into_owned(),
                _ => {}
            }
        }
        return;
    }

    // --- Pipeline loaded: realtime UI ---

    on_ensure_thread();

    // Status
    ui.horizontal(|ui| {
        let (dot_color, label) = if *running {
            if *bypass {
                (crate::theme::colors::YELLOW, "BYPASS")
            } else {
                (crate::theme::colors::MINT, "● LIVE")
            }
        } else {
            (crate::theme::colors::TEXT_MUTED, "STOPPED")
        };
        crate::theme::status_dot(ui, *running, dot_color);
        ui.label(
            egui::RichText::new(label)
                .size(16.0)
                .strong()
                .color(if *running {
                    crate::theme::colors::TEXT
                } else {
                    crate::theme::colors::TEXT_MUTED
                }),
        );
    });

    ui.add_space(12.0);

    // Level meters
    crate::theme::info_card(ui, |ui| {
        crate::theme::level_meter(ui, metrics.input_rms, "Input");
        crate::theme::level_meter(ui, metrics.output_rms, "Output");

        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(format!(
                "Latency: {:.0} ms  |  RTF: {:.2}",
                metrics.latency_ms, metrics.rtf
            ))
            .size(12.0)
            .color(crate::theme::colors::TEXT_DIM),
        );
    });

    ui.add_space(12.0);

    // Quality mode — knob or pill buttons
    ui.horizontal(|ui| {
        if let Some(tex) = knob_tex {
            // Knob: Strict=0.0, Balanced=0.5, Quality=1.0
            let mode_val = match *mode {
                lightvc_core::converter::LatencyMode::Strict => 0.0,
                lightvc_core::converter::LatencyMode::Balanced => 0.5,
                lightvc_core::converter::LatencyMode::Quality => 1.0,
            };
            let mode_name = match *mode {
                lightvc_core::converter::LatencyMode::Strict => "Strict",
                lightvc_core::converter::LatencyMode::Balanced => "Balanced",
                lightvc_core::converter::LatencyMode::Quality => "Quality",
            };

            let id = ui.make_persistent_id("rt_mode_knob");
            if let Some(new_val) = crate::theme::knob(ui, tex, id, mode_val, "Mode") {
                let old_mode = *mode;
                *mode = if new_val < 0.33 {
                    lightvc_core::converter::LatencyMode::Strict
                } else if new_val < 0.67 {
                    lightvc_core::converter::LatencyMode::Balanced
                } else {
                    lightvc_core::converter::LatencyMode::Quality
                };
                if *mode != old_mode {
                    on_control(RtControl::SetMode(*mode));
                }
            }

            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new("Mode")
                        .size(13.0)
                        .color(crate::theme::colors::TEXT_DIM),
                );
                ui.label(
                    egui::RichText::new(mode_name)
                        .size(16.0)
                        .strong()
                        .color(crate::theme::colors::PINK_BRIGHT),
                );
                ui.label(
                    egui::RichText::new(match *mode {
                        lightvc_core::converter::LatencyMode::Strict => "0ms lookahead",
                        lightvc_core::converter::LatencyMode::Balanced => "~40ms lookahead",
                        lightvc_core::converter::LatencyMode::Quality => "~80ms lookahead",
                    })
                    .size(10.0)
                    .color(crate::theme::colors::TEXT_MUTED),
                );
            });
        } else {
            // Fallback: pill buttons
            ui.label(
                egui::RichText::new("Mode")
                    .size(13.0)
                    .color(crate::theme::colors::TEXT_DIM),
            );
            let old = *mode;
            for (m, name) in [
                (lightvc_core::converter::LatencyMode::Strict, "Strict"),
                (lightvc_core::converter::LatencyMode::Balanced, "Balanced"),
                (lightvc_core::converter::LatencyMode::Quality, "Quality"),
            ] {
                if crate::theme::pill_button(ui, name, *mode == m) {
                    *mode = m;
                }
            }
            if *mode != old {
                on_control(RtControl::SetMode(*mode));
            }
        }
    });

    ui.add_space(8.0);

    // Bypass
    let old_bp = *bypass;
    ui.checkbox(bypass, "Bypass");
    if *bypass != old_bp {
        on_control(RtControl::Bypass(*bypass));
    }

    ui.add_space(12.0);

    // Start/Stop
    ui.horizontal(|ui| {
        if !*running {
            if crate::theme::pill_button(ui, "▶ Start", true) {
                on_control(RtControl::Start);
                *running = true;
            }
        } else {
            if crate::theme::pill_button(ui, "■ Stop", true) {
                on_control(RtControl::Stop);
                *running = false;
            }
        }
    });

    ui.add_space(12.0);

    // Audio devices
    ui.collapsing(
        egui::RichText::new("Audio Devices")
            .size(13.0)
            .color(crate::theme::colors::CYAN),
        |ui| {
            let inputs = lightvc_audio::DuplexStream::list_input_devices().unwrap_or_default();
            let outputs = lightvc_audio::DuplexStream::list_output_devices().unwrap_or_default();
            ui.label(
                egui::RichText::new("Inputs")
                    .size(12.0)
                    .color(crate::theme::colors::TEXT_DIM),
            );
            for d in &inputs {
                ui.label(
                    egui::RichText::new(format!(
                        "  {} ({}Hz, {}ch)",
                        d.name, d.sample_rate, d.channels
                    ))
                    .size(11.0)
                    .color(crate::theme::colors::TEXT_MUTED),
                );
            }
            ui.label(
                egui::RichText::new("Outputs")
                    .size(12.0)
                    .color(crate::theme::colors::TEXT_DIM),
            );
            for d in &outputs {
                ui.label(
                    egui::RichText::new(format!(
                        "  {} ({}Hz, {}ch)",
                        d.name, d.sample_rate, d.channels
                    ))
                    .size(11.0)
                    .color(crate::theme::colors::TEXT_MUTED),
                );
            }
        },
    );
}

// =========================================================================
// Inference thread
// =========================================================================

pub fn inference_loop(
    pipeline: Arc<Mutex<lightvc_core::pipeline::VcPipeline>>,
    control_rx: Receiver<RtControl>,
    metrics_tx: Sender<RtMetrics>,
) {
    let mut running = false;
    let mut bypass = false;
    let mut duplex: Option<lightvc_audio::DuplexStream> = None;
    let mut capture_consumer: Option<rtrb::Consumer<f32>> = None;
    let mut playback_producer: Option<rtrb::Producer<f32>> = None;
    let mut resampler_up: Option<lightvc_audio::Resampler> = None;
    let mut resampler_down: Option<lightvc_audio::Resampler> = None;
    let mut device_sr: u32 = 44_100;

    loop {
        while let Ok(msg) = control_rx.try_recv() {
            match msg {
                RtControl::Start => {
                    if running {
                        continue;
                    }
                    match start_audio(
                        &mut capture_consumer,
                        &mut playback_producer,
                        &mut device_sr,
                    ) {
                        Ok(d) => {
                            if let Ok(r1) = lightvc_audio::Resampler::new(device_sr as usize, 4096)
                            {
                                resampler_up = Some(r1);
                            }
                            if let Ok(r2) = lightvc_audio::Resampler::new(device_sr as usize, 4096)
                            {
                                resampler_down = Some(r2);
                            }
                            duplex = Some(d);
                            running = true;
                        }
                        Err(e) => eprintln!("Audio: {e}"),
                    }
                }
                RtControl::Stop => {
                    running = false;
                    duplex = None;
                    capture_consumer = None;
                    playback_producer = None;
                    resampler_up = None;
                    resampler_down = None;
                }
                RtControl::SetMode(mode) => {
                    if let Ok(mut p) = pipeline.lock() {
                        p.codec_mut().set_chunk_mode(match mode {
                            lightvc_core::converter::LatencyMode::Strict => {
                                lightvc_core::streaming::ChunkMode::Strict
                            }
                            lightvc_core::converter::LatencyMode::Balanced => {
                                lightvc_core::streaming::ChunkMode::Balanced
                            }
                            lightvc_core::converter::LatencyMode::Quality => {
                                lightvc_core::streaming::ChunkMode::Quality
                            }
                        });
                    }
                }
                RtControl::Bypass(b) => bypass = b,
                RtControl::LoadReference(pcm) => {
                    if let Ok(mut p) = pipeline.lock() {
                        let _ = p.set_target(&pcm);
                    }
                }
            }
        }

        if !running {
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }

        let Some(cap_rx) = capture_consumer.as_mut() else {
            continue;
        };
        let Some(pb_tx) = playback_producer.as_mut() else {
            continue;
        };

        let chunk_sz = pipeline.lock().map(|p| p.chunk_samples()).unwrap_or(2048);
        let needed = resampler_up
            .as_ref()
            .map(|r| r.input_frames_needed_up())
            .unwrap_or(chunk_sz);

        let mut cap = Vec::with_capacity(needed);
        while cap.len() < needed {
            match cap_rx.pop() {
                Ok(s) => cap.push(s),
                Err(_) => break,
            }
        }
        if cap.len() < needed.min(512) {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }
        if cap.len() < needed {
            cap.resize(needed, 0.0);
        }

        let t0 = Instant::now();

        let pcm_44k = if device_sr != 44_100 {
            resampler_up
                .as_mut()
                .and_then(|r| r.process_up(&cap).ok())
                .unwrap_or_else(|| cap.clone())
        } else {
            cap.clone()
        };

        let mut chunk = pcm_44k;
        if chunk.len() < chunk_sz {
            chunk.resize(chunk_sz, 0.0);
        } else if chunk.len() > chunk_sz {
            chunk.truncate(chunk_sz);
        }

        let out_44k = if bypass {
            chunk
        } else {
            match pipeline.lock() {
                Ok(mut p) => p.process_chunk(&chunk).unwrap_or_else(|e| {
                    eprintln!("VC: {e}");
                    chunk
                }),
                Err(_) => chunk,
            }
        };

        let in_rms = widgets::rms(&cap);
        let out_rms = widgets::rms(&out_44k);
        let elapsed = t0.elapsed();

        let out_dev = if device_sr != 44_100 {
            resampler_down
                .as_mut()
                .and_then(|r| r.process_down(&out_44k).ok())
                .unwrap_or(out_44k)
        } else {
            out_44k
        };

        for s in &out_dev {
            let _ = pb_tx.push(*s);
        }

        let dur_ms = (out_dev.len() as f32 / device_sr as f32) * 1000.0;
        let rtf = if dur_ms > 0.0 {
            elapsed.as_secs_f32() / (dur_ms / 1000.0)
        } else {
            0.0
        };
        let _ = metrics_tx.send(RtMetrics {
            input_rms: in_rms,
            output_rms: out_rms,
            latency_ms: dur_ms,
            rtf,
        });
    }
}

fn start_audio(
    cap: &mut Option<rtrb::Consumer<f32>>,
    pb: &mut Option<rtrb::Producer<f32>>,
    sr: &mut u32,
) -> anyhow::Result<lightvc_audio::DuplexStream> {
    let input = lightvc_audio::DuplexStream::default_input()?;
    let output = lightvc_audio::DuplexStream::default_output()?;
    *sr = input
        .default_input_config()
        .map(|c| c.sample_rate())
        .unwrap_or(44_100);

    let (tx, rx1) = rtrb::RingBuffer::new(1 << 16);
    let (tx2, rx2) = rtrb::RingBuffer::new(1 << 16);
    let d = lightvc_audio::DuplexStream::start(&input, &output, tx, rx2)?;
    *cap = Some(rx1);
    *pb = Some(tx2);
    Ok(d)
}

//! Tab 2: Real-time voice conversion.
//! Mic input → DAC encode → converter → DAC decode → speaker output.
//!
//! Contains both UI rendering and the inference thread loop.

use std::sync::{Arc, Mutex};
use std::time::Instant;

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
    icon_stop_tex: Option<&egui::TextureHandle>,
    mut on_load: impl FnMut(&str, &str),
    mut on_ensure_thread: impl FnMut(),
    on_control: impl Fn(RtControl),
) {
    crate::theme::heading(ui, "Real-time Voice Conversion");
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
        if metrics.overrun > 0 || metrics.underrun > 0 {
            ui.label(
                egui::RichText::new(format!(
                    "xruns: {} over / {} under",
                    metrics.overrun, metrics.underrun
                ))
                .size(11.0)
                .color(crate::theme::colors::YELLOW),
            );
        }
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

    // Bypass — styled toggle button
    let old_bp = *bypass;
    if crate::theme::pill_button(ui, if *bypass { "BYPASS ON" } else { "Bypass" }, *bypass) {
        *bypass = !*bypass;
    }
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
            let clicked = if let Some(tex) = icon_stop_tex {
                crate::theme::icon_button(ui, tex, " Stop", true)
            } else {
                crate::theme::pill_button(ui, "■ Stop", true)
            };
            if clicked {
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
    let mut engine: Option<lightvc_audio::AudioEngine> = None;
    let mut capture_consumer: Option<rtrb::Consumer<f32>> = None;
    let mut playback_producer: Option<rtrb::Producer<f32>> = None;
    let mut resampler_up: Option<lightvc_audio::Resampler> = None;
    let mut resampler_down: Option<lightvc_audio::Resampler> = None;
    let mut device_sr: u32 = 44_100;
    let mut disconnected_reported = false;

    // Three-stage buffering decouples the four sample-frame domains:
    //   capture (device_sr) → [in_accum] → process_up →
    //   [pcm_44k_accum]     → process_chunk →
    //   [out_44k_accum]     → process_down → playback (device_sr)
    // Each stage drains in exact multiples of its native frame unit, so
    // the resamplers never see truncated input and partial chunks carry
    // over to the next iteration instead of being zero-padded.
    let mut in_accum: Vec<f32> = Vec::new();
    let mut pcm_44k_accum: Vec<f32> = Vec::new();
    let mut out_44k_accum: Vec<f32> = Vec::new();

    // Metrics from the most recently processed chunk.
    let mut in_rms_last: f32 = 0.0;
    let mut out_rms_last: f32 = 0.0;
    let mut rtf_last: f32 = 0.0;
    let mut latency_ms_last: f32 = 0.0;

    loop {
        while let Ok(msg) = control_rx.try_recv() {
            match msg {
                RtControl::Start => {
                    if running {
                        continue;
                    }
                    disconnected_reported = false;
                    match lightvc_audio::AudioEngine::start_default() {
                        Ok((eng, bufs)) => {
                            device_sr = eng.capture_sample_rate;
                            if let Ok(r1) = lightvc_audio::Resampler::new(device_sr as usize, 4096)
                            {
                                resampler_up = Some(r1);
                            }
                            if let Ok(r2) = lightvc_audio::Resampler::new(device_sr as usize, 4096)
                            {
                                resampler_down = Some(r2);
                            }
                            in_accum.clear();
                            pcm_44k_accum.clear();
                            out_44k_accum.clear();
                            capture_consumer = Some(bufs.capture);
                            playback_producer = Some(bufs.playback);
                            engine = Some(eng);
                            running = true;
                        }
                        Err(e) => eprintln!("Audio: {e}"),
                    }
                }
                RtControl::Stop => {
                    running = false;
                    engine = None;
                    capture_consumer = None;
                    playback_producer = None;
                    resampler_up = None;
                    resampler_down = None;
                    in_accum.clear();
                    pcm_44k_accum.clear();
                    out_44k_accum.clear();
                    disconnected_reported = false;
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

        // [07-4] device disconnection: cpal error callback sets the flag.
        // Tear down and notify the UI so the user can re-select a device.
        if engine
            .as_ref()
            .map(|e| e.is_disconnected())
            .unwrap_or(false)
        {
            if !disconnected_reported {
                eprintln!("Audio device disconnected — stopping pipeline");
                disconnected_reported = true;
            }
            running = false;
            engine = None;
            capture_consumer = None;
            playback_producer = None;
            resampler_up = None;
            resampler_down = None;
            let _ = metrics_tx.send(RtMetrics {
                disconnected: true,
                ..Default::default()
            });
            continue;
        }

        let Some(cap_rx) = capture_consumer.as_mut() else {
            continue;
        };
        let Some(pb_tx) = playback_producer.as_mut() else {
            continue;
        };

        let chunk_sz = pipeline.lock().map(|p| p.chunk_samples()).unwrap_or(2048);
        let is_passthrough_sr = device_sr == 44_100;

        let mut did_work = false;

        // ---- Stage 1: drain capture ring buffer into in_accum (device_sr).
        // Never zero-pad here; if the device underruns we simply process less.
        let mut popped = 0usize;
        while let Ok(s) = cap_rx.pop() {
            in_accum.push(s);
            popped += 1;
            if popped >= 8192 {
                break;
            }
        }
        if popped > 0 {
            did_work = true;
        }

        // ---- Stage 2: resample device_sr → 44.1k in exact input chunks.
        if !is_passthrough_sr {
            if let Some(r_up) = resampler_up.as_mut() {
                let needed = r_up.input_frames_needed_up();
                while in_accum.len() >= needed {
                    let input_chunk: Vec<f32> = in_accum.drain(..needed).collect();
                    match r_up.process_up(&input_chunk) {
                        Ok(out) => {
                            pcm_44k_accum.extend_from_slice(out);
                            did_work = true;
                        }
                        Err(e) => eprintln!("resample up: {e}"),
                    }
                }
            }
        } else if !in_accum.is_empty() {
            // device_sr == 44.1k: no resampling, pass samples straight through.
            pcm_44k_accum.append(&mut in_accum);
            did_work = true;
        }

        // ---- Stage 3: run the converter on whole chunks of chunk_sz frames.
        while pcm_44k_accum.len() >= chunk_sz {
            let chunk: Vec<f32> = pcm_44k_accum.drain(..chunk_sz).collect();
            let t0 = Instant::now();

            let out = if bypass {
                chunk.clone()
            } else {
                match pipeline.lock() {
                    Ok(mut p) => p.process_chunk(&chunk).unwrap_or_else(|e| {
                        eprintln!("VC: {e}");
                        chunk.clone()
                    }),
                    Err(_) => chunk.clone(),
                }
            };

            let elapsed = t0.elapsed();
            in_rms_last = widgets::rms(&chunk);
            out_rms_last = widgets::rms(&out);
            let dur_s = out.len() as f32 / 44_100.0;
            rtf_last = if dur_s > 0.0 {
                elapsed.as_secs_f32() / dur_s
            } else {
                0.0
            };
            // End-to-end latency estimate: capture/playback cpal buffers
            // (~10 ms each per ARCHITECTURE §1.3) + resample (~3 ms each
            // side) + algorithmic (chunk + FRC lookahead). Bypass skips the
            // algorithmic term since encode/convert/decode are not run.
            let algo_ms = if bypass {
                0.0
            } else {
                pipeline
                    .lock()
                    .map(|p| p.algorithmic_latency_ms())
                    .unwrap_or(0.0)
            };
            latency_ms_last = 10.0 + 3.0 + algo_ms + 3.0 + 10.0;

            out_44k_accum.extend_from_slice(&out);
            did_work = true;
        }

        // ---- Stage 4: resample 44.1k → device_sr and push to playback.
        if !is_passthrough_sr {
            if let Some(r_down) = resampler_down.as_mut() {
                // SincFixedOut has a variable input length; read it fresh each
                // iteration. process_down consumes exactly that many frames.
                let mut needed = r_down.input_frames_needed_down();
                while out_44k_accum.len() >= needed && needed > 0 {
                    let input_chunk: Vec<f32> = out_44k_accum.drain(..needed).collect();
                    match r_down.process_down(&input_chunk) {
                        Ok(out) => {
                            for s in out {
                                let _ = pb_tx.push(*s);
                            }
                        }
                        Err(e) => eprintln!("resample down: {e}"),
                    }
                    needed = r_down.input_frames_needed_down();
                }
            }
        } else {
            // device_sr == 44.1k: push decoded samples directly.
            for s in out_44k_accum.drain(..) {
                let _ = pb_tx.push(s);
            }
        }

        let (overrun, underrun) = engine
            .as_ref()
            .map(|e| (e.overrun_count(), e.underrun_count()))
            .unwrap_or((0, 0));
        let _ = metrics_tx.send(RtMetrics {
            input_rms: in_rms_last,
            output_rms: out_rms_last,
            latency_ms: latency_ms_last,
            rtf: rtf_last,
            disconnected: false,
            overrun,
            underrun,
        });

        if !did_work {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}

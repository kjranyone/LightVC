//! Tab 2: Real-time voice conversion.
//! Mic input → DAC encode → converter → DAC decode → speaker output.
//!
//! Contains both UI rendering and the inference thread loop.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use cpal::traits::HostTrait;
use crossbeam_channel::{Receiver, Sender};
use eframe::egui;

use crate::app::AppState;
use crate::app::{RtControl, RtMetrics};
use crate::file_pick::FilePick;
use crate::widgets;

/// Render the realtime tab.
#[allow(clippy::too_many_arguments)]
pub fn render(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    converter_pick: &FilePick,
    state: &Arc<Mutex<AppState>>,
    conv_path: &mut String,
    conv_config: &mut lightvc_core::converter::ConverterConfig,
    running: &mut bool,
    bypass: &mut bool,
    mode: &mut lightvc_core::converter::LatencyMode,
    prosody_mode: &mut lightvc_core::converter::ProsodyMode,
    prosody_blend: &mut f32,
    velocity_scale: &mut f32,
    metrics: &RtMetrics,
    mut on_load: impl FnMut(&str, &lightvc_core::converter::ConverterConfig),
    mut on_ensure_thread: impl FnMut(),
    on_control: impl Fn(RtControl),
) {
    let has_pipeline = state.lock().unwrap().pipeline.is_some();
    let force_bypass = !has_pipeline;

    on_ensure_thread();

    // --- Status determination ---
    let (status_color, status_label) = if *running {
        if *bypass {
            (crate::theme::colors::STATUS_BYPASS, "BYPASS")
        } else {
            (crate::theme::colors::STATUS_CONVERTING, "CONVERTING")
        }
    } else {
        (crate::theme::colors::STATUS_STOPPED, "STOPPED")
    };
    let ref_name: Option<String> = state.lock().ok().and_then(|s| {
        s.selected_voice
            .and_then(|i| s.voices.get(i).map(|v| v.name.clone()))
    });

    // --- Row 1: 2-column grid, height-aligned (Model | Status) ---
    // Use a fixed row height so both cards align bottom edges.
    let col_gap = crate::theme::space::MEDIUM;
    let col_w = (ui.available_width() - col_gap) / 2.0;
    let card_min_h = 150.0;

    ui.horizontal_top(|ui| {
        // LEFT: Model card
        ui.allocate_ui_with_layout(
            egui::vec2(col_w, card_min_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                let frame = if has_pipeline {
                    crate::theme::info_card_frame()
                } else {
                    crate::theme::glow_card_frame()
                };
                frame.show(ui, |ui| {
                    ui.set_min_height(card_min_h - 40.0);
                    crate::theme::subheading(ui, "Model");

                    if !has_pipeline {
                        let out = crate::theme::drop_zone(
                            ui,
                            "Drop converter.safetensors here",
                            if conv_path.is_empty() {
                                None
                            } else {
                                Some(conv_path.as_str())
                            },
                            "Browse",
                        );
                        if out.browse_clicked {
                            converter_pick.open(ctx);
                        }

                        egui::CollapsingHeader::new(
                            egui::RichText::new("[*] Config")
                                .size(11.0)
                                .color(crate::theme::colors::TEXT_DIM),
                        )
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.set_min_width(col_w - 40.0);
                            ui.columns(2, |cols| {
                                cols[0].horizontal(|ui| {
                                    ui.label("latent");
                                    ui.add(
                                        egui::DragValue::new(&mut conv_config.latent_dim)
                                            .range(1..=2048),
                                    );
                                });
                                cols[1].horizontal(|ui| {
                                    ui.label("hidden");
                                    ui.add(
                                        egui::DragValue::new(&mut conv_config.hidden_dim)
                                            .range(1..=4096),
                                    );
                                });
                                cols[0].horizontal(|ui| {
                                    ui.label("blocks");
                                    ui.add(
                                        egui::DragValue::new(&mut conv_config.n_conv_blocks)
                                            .range(1..=32),
                                    );
                                });
                                cols[1].horizontal(|ui| {
                                    ui.label("spk");
                                    ui.add(
                                        egui::DragValue::new(&mut conv_config.speaker_embed_dim)
                                            .range(1..=1024),
                                    );
                                });
                                cols[0].horizontal(|ui| {
                                    ui.label("timbre");
                                    ui.add(
                                        egui::DragValue::new(&mut conv_config.n_timbre_tokens)
                                            .range(1..=256),
                                    );
                                });
                                cols[1].horizontal(|ui| {
                                    ui.label("heads");
                                    ui.add(
                                        egui::DragValue::new(&mut conv_config.n_attn_heads)
                                            .range(1..=64),
                                    );
                                });
                                cols[0].horizontal(|ui| {
                                    ui.label("bneck");
                                    ui.add(
                                        egui::DragValue::new(&mut conv_config.bottleneck_dim)
                                            .range(1..=1024),
                                    );
                                });
                                cols[1].horizontal(|ui| {
                                    ui.label("time");
                                    ui.add(
                                        egui::DragValue::new(&mut conv_config.time_embed_dim)
                                            .range(1..=512),
                                    );
                                });
                                cols[0].horizontal(|ui| {
                                    ui.label("depth");
                                    ui.add(
                                        egui::DragValue::new(&mut conv_config.n_depth_groups)
                                            .range(0..=8),
                                    );
                                });
                                cols[1].checkbox(&mut conv_config.enable_timbre, "timbre");
                            });
                        });

                        ui.add_space(crate::theme::space::TIGHT);
                        ui.horizontal(|ui| {
                            if crate::theme::primary_button(ui, "Load", !conv_path.is_empty())
                                && !conv_path.is_empty()
                            {
                                on_load(conv_path, conv_config);
                            }
                            ui.label(
                                egui::RichText::new("no model -> BYPASS")
                                    .size(10.0)
                                    .color(crate::theme::colors::ERROR),
                            );
                        });
                    } else {
                        let model_name = state
                            .lock()
                            .ok()
                            .and_then(|s| {
                                s.converter_weights.as_ref().and_then(|p| {
                                    p.file_name().map(|f| f.to_string_lossy().into_owned())
                                })
                            })
                            .unwrap_or_else(|| "converter.safetensors".to_string());
                        ui.label(
                            egui::RichText::new(format!("> {model_name}"))
                                .size(13.0)
                                .color(crate::theme::colors::TEXT),
                        );
                        ui.label(
                            egui::RichText::new("ready")
                                .size(11.0)
                                .color(crate::theme::colors::TEXT_DIM),
                        );
                    }
                });
            },
        );

        ui.add_space(col_gap);

        // RIGHT: Status card
        ui.allocate_ui_with_layout(
            egui::vec2(col_w, card_min_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                let frame = if *running && !*bypass {
                    crate::theme::glow_card_frame()
                } else {
                    crate::theme::info_card_frame()
                };
                frame.show(ui, |ui| {
                    ui.set_min_height(card_min_h - 40.0);
                    crate::theme::subheading(ui, "Status");
                    ui.horizontal(|ui| {
                        crate::theme::status_badge(ui, status_label, status_color);
                        ui.add_space(crate::theme::space::SMALL);
                        // Voice inline
                        if let Some(name) = &ref_name {
                            ui.label(
                                egui::RichText::new(format!("> {name}"))
                                    .size(12.0)
                                    .color(crate::theme::colors::TEXT),
                            );
                        } else {
                            ui.label(
                                egui::RichText::new("no voice")
                                    .size(11.0)
                                    .color(crate::theme::colors::TEXT_MUTED),
                            );
                        }
                    });
                    ui.add_space(crate::theme::space::TIGHT);
                    // Stats inline
                    ui.horizontal(|ui| {
                        crate::theme::stat_card(
                            ui,
                            &format!("{:.0}", metrics.latency_ms),
                            "ms",
                            crate::theme::colors::TEXT,
                        );
                        ui.add_space(crate::theme::space::TIGHT);
                        crate::theme::stat_card(
                            ui,
                            &format!("{:.2}", metrics.rtf),
                            "RTF",
                            crate::theme::colors::TEXT,
                        );
                        if metrics.overrun > 0 || metrics.underrun > 0 {
                            ui.add_space(crate::theme::space::TIGHT);
                            ui.label(
                                egui::RichText::new(format!(
                                    "in:{}  out:{}",
                                    metrics.overrun, metrics.underrun
                                ))
                                .size(10.0)
                                .color(crate::theme::colors::ERROR),
                            );
                        }
                    });
                });
            },
        );
    });

    // Handle file picks.
    if let Some(path) = converter_pick.take() {
        *conv_path = path.to_string_lossy().into_owned();
    }

    ui.add_space(crate::theme::space::MEDIUM);

    // --- Row 2: Input signal meter (compact) ---
    crate::theme::info_card_frame().show(ui, |ui| {
        ui.horizontal(|ui| {
            crate::theme::subheading(ui, "Input");
            ui.add_space(crate::theme::space::SMALL);
            if metrics.auto_degraded {
                ui.label(
                    egui::RichText::new(format!("! auto {:?}", metrics.current_mode))
                        .size(10.0)
                        .color(crate::theme::colors::LEMON_DEEP),
                );
            }
            crate::theme::level_meter_kind_compact(
                ui,
                metrics.input_rms,
                crate::theme::MeterKind::Input,
            );
        });
    });

    ui.add_space(crate::theme::space::MEDIUM);

    // --- Row 3: Operation bar (Bypass | Mode pills | Start/Stop) ---
    {
        let frame = crate::theme::info_card_frame();
        frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                // Bypass
                if !force_bypass {
                    let old_bp = *bypass;
                    if crate::theme::operation_button(
                        ui,
                        if *bypass { "BYPASS" } else { "Bypass" },
                        crate::theme::OpKind::Bypass,
                        *bypass,
                    ) {
                        *bypass = !*bypass;
                    }
                    if *bypass != old_bp {
                        on_control(RtControl::Bypass(*bypass));
                    }
                }

                // Mode pills (center) — only when converter loaded
                if !force_bypass {
                    ui.add_space(crate::theme::space::SMALL);
                    let old_mode = *mode;
                    for (m, name) in [
                        (lightvc_core::converter::LatencyMode::Strict, "Strict"),
                        (lightvc_core::converter::LatencyMode::Balanced, "Balanced"),
                        (lightvc_core::converter::LatencyMode::Quality, "Quality"),
                    ] {
                        if crate::theme::pill_button(ui, name, *mode == m) {
                            *mode = m;
                        }
                    }
                    if *mode != old_mode {
                        on_control(RtControl::SetMode(*mode));
                    }
                }

                // Start/Stop (right)
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    if !*running {
                        let label = if force_bypass { "Start" } else { "Start" };
                        if crate::theme::operation_button(
                            ui,
                            label,
                            crate::theme::OpKind::Start,
                            true,
                        ) {
                            if force_bypass {
                                on_control(RtControl::Bypass(true));
                            }
                            on_control(RtControl::StartWithDevices {
                                input_idx: state.lock().unwrap().selected_input,
                                output_idx: state.lock().unwrap().selected_output,
                            });
                            *running = true;
                        }
                    } else {
                        if crate::theme::operation_button(
                            ui,
                            "Stop",
                            crate::theme::OpKind::Stop,
                            true,
                        ) {
                            on_control(RtControl::Stop);
                            *running = false;
                        }
                    }
                });
            });
        });
    }

    // --- Row 4: Prosody + Velocity (compact, converter-only) ---
    if !force_bypass {
        ui.add_space(crate::theme::space::SMALL);
        crate::theme::info_card_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                // Prosody mode
                ui.label(
                    egui::RichText::new("Prosody")
                        .size(11.0)
                        .color(crate::theme::colors::TEXT_DIM),
                );
                let old_pm = *prosody_mode;
                egui::ComboBox::from_id_salt("rt_prosody_mode")
                    .selected_text(format!("{:?}", prosody_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            prosody_mode,
                            lightvc_core::converter::ProsodyMode::ImitateTarget,
                            "Imitate",
                        );
                        ui.selectable_value(
                            prosody_mode,
                            lightvc_core::converter::ProsodyMode::PreserveSource,
                            "Preserve",
                        );
                        ui.selectable_value(
                            prosody_mode,
                            lightvc_core::converter::ProsodyMode::Blend,
                            "Blend",
                        );
                        ui.selectable_value(
                            prosody_mode,
                            lightvc_core::converter::ProsodyMode::FlattenPrivacy,
                            "Flatten",
                        );
                    });
                if *prosody_mode != old_pm {
                    on_control(RtControl::SetProsody {
                        mode: *prosody_mode,
                        blend: *prosody_blend as f64,
                    });
                }

                ui.add_space(crate::theme::space::SMALL);

                // Blend slider
                let old_b = *prosody_blend;
                ui.add_enabled_ui(
                    *prosody_mode == lightvc_core::converter::ProsodyMode::Blend,
                    |ui| {
                        ui.add(
                            egui::Slider::new(prosody_blend, 0.0..=1.0)
                                .text("blend")
                                .fixed_decimals(2),
                        );
                    },
                );
                if *prosody_blend != old_b {
                    on_control(RtControl::SetProsody {
                        mode: *prosody_mode,
                        blend: *prosody_blend as f64,
                    });
                }

                ui.add_space(crate::theme::space::SMALL);

                // Velocity scale
                let old_v = *velocity_scale;
                ui.add(
                    egui::Slider::new(velocity_scale, 0.0..=2.0)
                        .text("velocity")
                        .fixed_decimals(2),
                );
                if *velocity_scale != old_v {
                    on_control(RtControl::SetVelocityScale(*velocity_scale as f64));
                }
            });
        });
    }

    ui.add_space(crate::theme::space::SMALL);
}

// =========================================================================
// Inference thread
// =========================================================================

pub fn inference_loop(
    pipeline_slot: crate::app::PipelineSlot,
    control_rx: Receiver<RtControl>,
    metrics_tx: Sender<RtMetrics>,
) {
    let mut running = false;
    let mut bypass = pipeline_slot.lock().unwrap().is_none(); // force bypass when no converter
    let mut engine: Option<lightvc_audio::AudioEngine> = None;
    let mut capture_consumer: Option<rtrb::Consumer<f32>> = None;
    let mut playback_producer: Option<rtrb::Producer<f32>> = None;
    let mut resampler_up: Option<lightvc_audio::Resampler> = None;
    let mut resampler_down: Option<lightvc_audio::Resampler> = None;
    let mut device_sr: u32 = 44_100;
    let mut playback_sr: u32 = 44_100;
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

    // Pre-allocated work buffers for resampler input ([05-5]): reused across
    // iterations to avoid per-chunk Vec allocation on the realtime path.
    let mut resample_up_buf: Vec<f32> = Vec::with_capacity(8192);
    let mut resample_down_buf: Vec<f32> = Vec::with_capacity(8192);

    // Metrics from the most recently processed chunk.
    let mut in_rms_last: f32 = 0.0;
    let mut out_rms_last: f32 = 0.0;
    let mut rtf_last: f32 = 0.0;
    let mut latency_ms_last: f32 = 0.0;

    // [07-5] underrun auto-degradation: if the playback ring repeatedly
    // underruns (inference too slow), automatically downgrade the latency
    // mode to reduce CPU load.
    let mut last_underrun: u64 = 0;
    let mut underrun_streak: u32 = 0;
    let mut auto_degraded: bool = false;

    loop {
        while let Ok(msg) = control_rx.try_recv() {
            match msg {
                RtControl::StartWithDevices { .. } => {
                    if running {
                        continue;
                    }
                    disconnected_reported = false;
                    // [05-6]: use explicitly selected devices when provided.
                    // [05-7]: device enumeration happens here (inference thread)
                    // because cpal Device is not Send on macOS. The UI thread
                    // passes indices; the inference thread resolves them.
                    let eng_result = match msg {
                        RtControl::StartWithDevices {
                            input_idx: Some(ii),
                            output_idx: Some(oi),
                        } => {
                            let _inputs = lightvc_audio::DuplexStream::list_input_devices()
                                .unwrap_or_default();
                            let _outputs = lightvc_audio::DuplexStream::list_output_devices()
                                .unwrap_or_default();
                            let host = cpal::default_host();
                            let in_dev = host.input_devices().ok().and_then(|mut d| d.nth(ii));
                            let out_dev = host.output_devices().ok().and_then(|mut d| d.nth(oi));
                            match (in_dev, out_dev) {
                                (Some(id), Some(od)) => lightvc_audio::AudioEngine::start(&id, &od),
                                _ => {
                                    eprintln!("Device selection failed, falling back to default");
                                    lightvc_audio::AudioEngine::start_default()
                                }
                            }
                        }
                        RtControl::StartWithDevices {
                            input_idx: Some(ii),
                            output_idx: None,
                        } => {
                            let host = cpal::default_host();
                            let in_dev = host.input_devices().ok().and_then(|mut d| d.nth(ii));
                            let out_dev = lightvc_audio::DuplexStream::default_output().ok();
                            match (in_dev, out_dev) {
                                (Some(id), Some(od)) => lightvc_audio::AudioEngine::start(&id, &od),
                                _ => lightvc_audio::AudioEngine::start_default(),
                            }
                        }
                        RtControl::StartWithDevices {
                            input_idx: None,
                            output_idx: Some(oi),
                        } => {
                            let host = cpal::default_host();
                            let in_dev = lightvc_audio::DuplexStream::default_input().ok();
                            let out_dev = host.output_devices().ok().and_then(|mut d| d.nth(oi));
                            match (in_dev, out_dev) {
                                (Some(id), Some(od)) => lightvc_audio::AudioEngine::start(&id, &od),
                                _ => lightvc_audio::AudioEngine::start_default(),
                            }
                        }
                        _ => lightvc_audio::AudioEngine::start_default(),
                    };
                    match eng_result {
                        Ok((eng, bufs)) => {
                            device_sr = eng.capture_sample_rate;
                            playback_sr = eng.playback_sample_rate;
                            // Capture resampler: device_sr → 44.1k ([05-8]).
                            if let Ok(r1) = lightvc_audio::Resampler::new(device_sr as usize, 4096)
                            {
                                resampler_up = Some(r1);
                            }
                            // Playback resampler: 44.1k → playback_sr ([05-8]).
                            // Uses playback_sr, which may differ from capture_sr.
                            if let Ok(r2) =
                                lightvc_audio::Resampler::new(playback_sr as usize, 4096)
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
                    if let Some(mut p) = pipeline_slot
                        .lock()
                        .unwrap()
                        .as_ref()
                        .and_then(|p| p.lock().ok())
                    {
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
                RtControl::SetProsody { mode, blend } => {
                    if let Some(mut p) = pipeline_slot
                        .lock()
                        .unwrap()
                        .as_ref()
                        .and_then(|p| p.lock().ok())
                    {
                        p.set_prosody(mode, blend);
                    }
                }
                RtControl::SetVelocityScale(scale) => {
                    if let Some(mut p) = pipeline_slot
                        .lock()
                        .unwrap()
                        .as_ref()
                        .and_then(|p| p.lock().ok())
                    {
                        p.set_velocity_scale(scale);
                    }
                }
                RtControl::Bypass(b) => bypass = b,
                RtControl::LoadReference(pcm) => {
                    if let Some(mut p) = pipeline_slot
                        .lock()
                        .unwrap()
                        .as_ref()
                        .and_then(|p| p.lock().ok())
                    {
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

        let chunk_sz = pipeline_slot
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|p| p.lock().ok())
            .map(|p| p.chunk_samples())
            .unwrap_or(2048);
        // Capture and playback SR bypass are independent ([05-8]):
        // e.g. 44.1k mic + 48k HDMI needs capture bypass but playback resample.
        let capture_passthrough = device_sr == 44_100;
        let playback_passthrough = playback_sr == 44_100;

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
        // [07-5] overrun fix: if the consumer-side buffer has grown beyond
        // 3× the chunk size, the producer is overflowing. Drop the OLDEST
        // samples (front of in_accum) to stay near real-time. This implements
        // the "drop oldest" policy from ARCHITECTURE §8 that the rtrb SPSC
        // producer couldn't do directly.
        let max_pending = chunk_sz * 3;
        if in_accum.len() > max_pending {
            let excess = in_accum.len() - chunk_sz;
            in_accum.drain(..excess);
        }
        if popped > 0 {
            did_work = true;
        }

        // ---- Stage 2: resample device_sr → 44.1k in exact input chunks.
        if !capture_passthrough {
            if let Some(r_up) = resampler_up.as_mut() {
                let needed = r_up.input_frames_needed_up();
                while in_accum.len() >= needed {
                    resample_up_buf.clear();
                    resample_up_buf.extend(in_accum.drain(..needed));
                    match r_up.process_up(&resample_up_buf) {
                        Ok(out) => {
                            pcm_44k_accum.extend_from_slice(out);
                            did_work = true;
                        }
                        Err(e) => eprintln!("resample up: {e}"),
                    }
                }
            }
        } else if !in_accum.is_empty() {
            // capture_sr == 44.1k: no resampling needed.
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
                match pipeline_slot
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|p| p.lock().ok())
                {
                    Some(mut p) => p.process_chunk(&chunk).unwrap_or_else(|e| {
                        eprintln!("VC: {e}");
                        chunk.clone()
                    }),
                    None => chunk.clone(),
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
                pipeline_slot
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|p| p.lock().ok())
                    .map(|p| p.algorithmic_latency_ms())
                    .unwrap_or(0.0)
            };
            latency_ms_last = 10.0 + 3.0 + algo_ms + 3.0 + 10.0;

            out_44k_accum.extend_from_slice(&out);
            did_work = true;
        }

        // ---- Stage 4: resample 44.1k → playback_sr and push to playback.
        if !playback_passthrough {
            if let Some(r_down) = resampler_down.as_mut() {
                // SincFixedOut has a variable input length; read it fresh each
                // iteration. process_down consumes exactly that many frames.
                let mut needed = r_down.input_frames_needed_down();
                while out_44k_accum.len() >= needed && needed > 0 {
                    resample_down_buf.clear();
                    resample_down_buf.extend(out_44k_accum.drain(..needed));
                    match r_down.process_down(&resample_down_buf) {
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
            // playback_sr == 44.1k: push decoded samples directly.
            for s in out_44k_accum.drain(..) {
                let _ = pb_tx.push(s);
            }
        }

        let (overrun, underrun) = engine
            .as_ref()
            .map(|e| (e.overrun_count(), e.underrun_count()))
            .unwrap_or((0, 0));

        // [07-5] underrun auto-degradation.
        let new_underruns = underrun.saturating_sub(last_underrun);
        last_underrun = underrun;
        if new_underruns > 0 {
            underrun_streak += 1;
        } else {
            underrun_streak = 0;
        }
        // After 10 consecutive iterations with underruns, downgrade one level.
        if underrun_streak >= 10 && !auto_degraded {
            if let Some(mut p) = pipeline_slot
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|p| p.lock().ok())
            {
                let current = p.algorithmic_latency_ms();
                if current > 50.0 {
                    // Quality → Balanced, or Balanced → Strict.
                    let new_mode = if current > 150.0 {
                        lightvc_core::converter::LatencyMode::Balanced
                    } else {
                        lightvc_core::converter::LatencyMode::Strict
                    };
                    p.set_mode(new_mode);
                    auto_degraded = true;
                    eprintln!("Auto-degraded to {new_mode:?} due to underruns");
                }
            }
        }

        let current_mode = pipeline_slot
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|p| p.lock().ok())
            .map(|p| p.mode())
            .unwrap_or(lightvc_core::converter::LatencyMode::Strict);
        let _ = metrics_tx.send(RtMetrics {
            input_rms: in_rms_last,
            output_rms: out_rms_last,
            latency_ms: latency_ms_last,
            rtf: rtf_last,
            disconnected: false,
            overrun,
            underrun,
            current_mode,
            auto_degraded,
        });

        if !did_work {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}

//! egui desktop application for real-time VC.
//!
//! Inference thread runs the full pipeline:
//!   cpal capture → ringbuf → resample → DAC encode → converter → DAC decode → resample → ringbuf → cpal playback

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;

#[derive(Clone, Debug, Default)]
struct Metrics {
    input_rms: f32,
    output_rms: f32,
    latency_ms: f32,
    rtf: f32,
    processing: bool,
}

enum ControlMsg {
    Start,
    Stop,
    SetMode(lightvc_core::converter::LatencyMode),
    Bypass(bool),
}

pub struct LightVcApp {
    control_tx: Sender<ControlMsg>,
    metrics_rx: Receiver<Metrics>,
    metrics: Metrics,
    selected_mode: lightvc_core::converter::LatencyMode,
    bypass: bool,
    has_target: bool,
    running: bool,
    error: Option<String>,
    device_names: (Vec<String>, Vec<String>),
}

impl LightVcApp {
    pub fn new(pipeline: lightvc_core::pipeline::VcPipeline) -> Result<Self> {
        let (control_tx, control_rx) = unbounded();
        let (metrics_tx, metrics_rx) = unbounded();

        let has_target = pipeline.has_target();
        let pipeline = Arc::new(Mutex::new(pipeline));
        let pipeline_clone = pipeline.clone();
        std::thread::spawn(move || {
            inference_loop(pipeline_clone, control_rx, metrics_tx);
        });

        let input_devs = lightvc_audio::DuplexStream::list_input_devices()
            .unwrap_or_default()
            .iter()
            .map(|d| format!("{} ({}Hz)", d.name, d.sample_rate))
            .collect();
        let output_devs = lightvc_audio::DuplexStream::list_output_devices()
            .unwrap_or_default()
            .iter()
            .map(|d| format!("{} ({}Hz)", d.name, d.sample_rate))
            .collect();

        Ok(Self {
            control_tx,
            metrics_rx,
            metrics: Metrics::default(),
            selected_mode: lightvc_core::converter::LatencyMode::Balanced,
            bypass: false,
            has_target,
            running: false,
            error: None,
            device_names: (input_devs, output_devs),
        })
    }

    pub fn render(&mut self, ctx: &egui::Context) {
        while let Ok(m) = self.metrics_rx.try_recv() {
            self.metrics = m;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("LightVC-X");
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                let status = if self.running {
                    if self.bypass {
                        "BYPASS"
                    } else if self.has_target {
                        "CONVERTING"
                    } else {
                        "RUNNING (no target)"
                    }
                } else {
                    "STOPPED"
                };
                let color = if self.running {
                    if self.bypass {
                        egui::Color32::from_rgb(200, 200, 80)
                    } else {
                        egui::Color32::from_rgb(80, 200, 80)
                    }
                } else {
                    egui::Color32::from_rgb(160, 160, 160)
                };
                ui.colored_label(color, format!("* {status}"));
            });

            ui.add_space(8.0);
            ui.label("Input Level:");
            level_meter(ui, self.metrics.input_rms);
            ui.label("Output Level:");
            level_meter(ui, self.metrics.output_rms);

            ui.add_space(8.0);
            ui.label(format!(
                "Latency: {:.1} ms | RTF: {:.2}",
                self.metrics.latency_ms, self.metrics.rtf
            ));

            ui.add_space(12.0);
            ui.horizontal(|ui| {
                ui.label("Quality Mode:");
                ui.radio_value(
                    &mut self.selected_mode,
                    lightvc_core::converter::LatencyMode::Strict,
                    "Strict",
                );
                ui.radio_value(
                    &mut self.selected_mode,
                    lightvc_core::converter::LatencyMode::Balanced,
                    "Balanced",
                );
                ui.radio_value(
                    &mut self.selected_mode,
                    lightvc_core::converter::LatencyMode::Quality,
                    "Quality",
                );
            });

            ui.add_space(8.0);
            ui.checkbox(&mut self.bypass, "Bypass");

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if !self.running {
                    if ui.button("Start").clicked() {
                        let _ = self.control_tx.send(ControlMsg::Start);
                        self.running = true;
                    }
                } else {
                    if ui.button("Stop").clicked() {
                        let _ = self.control_tx.send(ControlMsg::Stop);
                        self.running = false;
                    }
                }
            });

            ui.add_space(12.0);
            ui.collapsing("Audio Devices", |ui| {
                ui.label("Inputs:");
                for name in &self.device_names.0 {
                    ui.label(format!("  {name}"));
                }
                ui.label("Outputs:");
                for name in &self.device_names.1 {
                    ui.label(format!("  {name}"));
                }
            });

            if let Some(ref err) = self.error {
                ui.add_space(8.0);
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    format!("Error: {err}"),
                );
            }
        });

        ctx.request_repaint();
    }
}

fn level_meter(ui: &mut egui::Ui, rms: f32) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 12.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(40, 40, 40));

    let level = (rms * 10.0).min(1.0).max(0.0);
    let bar_width = rect.width() * level;
    let color = if level > 0.85 {
        egui::Color32::from_rgb(220, 80, 80)
    } else if level > 0.6 {
        egui::Color32::from_rgb(220, 200, 80)
    } else {
        egui::Color32::from_rgb(80, 200, 80)
    };
    let bar_rect = egui::Rect::from_min_size(rect.min, egui::vec2(bar_width, rect.height()));
    painter.rect_filled(bar_rect, 2.0, color);
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Inference thread: manages cpal duplex stream + VC pipeline.
fn inference_loop(
    pipeline: Arc<Mutex<lightvc_core::pipeline::VcPipeline>>,
    control_rx: Receiver<ControlMsg>,
    metrics_tx: Sender<Metrics>,
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
        // Handle control messages
        while let Ok(msg) = control_rx.try_recv() {
            match msg {
                ControlMsg::Start => {
                    if running {
                        continue;
                    }
                    match start_audio_stream(
                        &mut capture_consumer,
                        &mut playback_producer,
                        &mut device_sr,
                    ) {
                        Ok(d) => {
                            // Create resamplers for device_sr ↔ 44100
                            match lightvc_audio::Resampler::new(device_sr as usize, 4096) {
                                Ok(rs) => {
                                    resampler_up = Some(rs);
                                    // Separate instance for down direction
                                    match lightvc_audio::Resampler::new(device_sr as usize, 4096) {
                                        Ok(rs2) => resampler_down = Some(rs2),
                                        Err(e) => eprintln!("Down resampler init error: {e}"),
                                    }
                                }
                                Err(e) => eprintln!("Up resampler init error: {e}"),
                            }
                            duplex = Some(d);
                            running = true;
                        }
                        Err(e) => {
                            eprintln!("Audio start error: {e}");
                            let _ = metrics_tx.send(Metrics {
                                processing: false,
                                ..Default::default()
                            });
                        }
                    }
                }
                ControlMsg::Stop => {
                    running = false;
                    duplex = None;
                    capture_consumer = None;
                    playback_producer = None;
                    resampler_up = None;
                    resampler_down = None;
                }
                ControlMsg::SetMode(mode) => {
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
                ControlMsg::Bypass(b) => {
                    bypass = b;
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

        // Determine how many device-rate samples we need for one pipeline chunk
        let pipeline_chunk_44k = pipeline.lock().map(|p| p.chunk_samples()).unwrap_or(2048);

        // Resampler input size needed
        let needed_up = resampler_up
            .as_ref()
            .map(|r| r.input_frames_needed_up())
            .unwrap_or(pipeline_chunk_44k);

        // Collect samples from capture ring buffer
        let mut cap_samples = Vec::with_capacity(needed_up);
        while cap_samples.len() < needed_up {
            match cap_rx.pop() {
                Ok(s) => cap_samples.push(s),
                Err(_) => break,
            }
        }

        if cap_samples.len() < needed_up.min(512) {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }

        // Pad if short
        if cap_samples.len() < needed_up {
            cap_samples.resize(needed_up, 0.0);
        }

        let t_start = Instant::now();

        // Resample device_sr → 44100
        let pcm_44k = if device_sr != 44_100 {
            resampler_up
                .as_mut()
                .and_then(|r| r.process_up(&cap_samples).ok())
                .unwrap_or_else(|| cap_samples.clone())
        } else {
            cap_samples.clone()
        };

        // Pad/truncate to pipeline chunk size
        let mut chunk = pcm_44k;
        if chunk.len() < pipeline_chunk_44k {
            chunk.resize(pipeline_chunk_44k, 0.0);
        } else if chunk.len() > pipeline_chunk_44k {
            chunk.truncate(pipeline_chunk_44k);
        }

        // Process or bypass
        let output_44k = if bypass {
            chunk
        } else {
            match pipeline.lock() {
                Ok(mut p) => p.process_chunk(&chunk).unwrap_or_else(|e| {
                    eprintln!("VC error: {e}");
                    chunk
                }),
                Err(e) => {
                    eprintln!("Pipeline lock error: {e}");
                    chunk
                }
            }
        };

        let in_rms = rms(&cap_samples);
        let out_rms = rms(&output_44k);
        let elapsed = t_start.elapsed();

        // Resample 44100 → device_sr
        let output_dev = if device_sr != 44_100 {
            resampler_down
                .as_mut()
                .and_then(|r| r.process_down(&output_44k).ok())
                .unwrap_or(output_44k)
        } else {
            output_44k
        };

        // Write to playback ring buffer
        for s in &output_dev {
            let _ = pb_tx.push(*s);
        }

        // Send metrics (use output_dev length which equals the actual playback samples)
        let chunk_dur_ms = (output_dev.len() as f32 / device_sr as f32) * 1000.0;
        let rtf = if chunk_dur_ms > 0.0 {
            elapsed.as_secs_f32() / (chunk_dur_ms / 1000.0)
        } else {
            0.0
        };
        let _ = metrics_tx.send(Metrics {
            input_rms: in_rms,
            output_rms: out_rms,
            latency_ms: chunk_dur_ms,
            rtf,
            processing: !bypass,
        });
    }
}

/// Start cpal duplex stream, returning the consumer/producer ends.
fn start_audio_stream(
    capture_consumer: &mut Option<rtrb::Consumer<f32>>,
    playback_producer: &mut Option<rtrb::Producer<f32>>,
    device_sr: &mut u32,
) -> Result<lightvc_audio::DuplexStream> {
    let input = lightvc_audio::DuplexStream::default_input()?;
    let output = lightvc_audio::DuplexStream::default_output()?;

    *device_sr = cpal::traits::DeviceTrait::default_input_config(&input)
        .map(|c| c.sample_rate())
        .unwrap_or(44_100);

    let buf_size = 1 << 16; // 65536 samples
    let (capture_tx, cap_rx) = rtrb::RingBuffer::new(buf_size);
    let (pb_tx, playback_rx) = rtrb::RingBuffer::new(buf_size);

    let duplex = lightvc_audio::DuplexStream::start(&input, &output, capture_tx, playback_rx)?;

    *capture_consumer = Some(cap_rx);
    *playback_producer = Some(pb_tx);

    Ok(duplex)
}

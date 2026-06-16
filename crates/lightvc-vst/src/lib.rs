//! LightVC-X VST3 Plugin
//!
//! Real-time voice conversion as a VST3 audio effect.
//! Uses nice-plug + nice-plug-egui.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{unbounded, Receiver, Sender};
use nice_plug::prelude::*;
use nice_plug_egui::{create_egui_editor, EguiState};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

#[derive(Params)]
struct LightVcParams {
    #[id = "bypass"]
    pub bypass: BoolParam,

    #[id = "mode"]
    pub mode: IntParam,

    #[id = "mix"]
    pub mix: FloatParam,

    #[id = "gain"]
    pub output_gain: FloatParam,

    #[persist = "model-path"]
    pub model_path: Arc<Mutex<String>>,

    #[persist = "dac-path"]
    pub dac_path: Arc<Mutex<String>>,

    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
}

impl Default for LightVcParams {
    fn default() -> Self {
        Self {
            bypass: BoolParam::new("Bypass", false),
            mode: IntParam::new("Mode", 1, IntRange::Linear { min: 0, max: 2 }),
            mix: FloatParam::new(
                "Mix",
                100.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(50.0))
            .with_unit("%"),
            output_gain: FloatParam::new(
                "Output",
                0.0,
                FloatRange::Skewed {
                    min: -24.0,
                    max: 24.0,
                    factor: FloatRange::gain_skew_factor(-24.0, 24.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(20.0))
            .with_unit(" dB"),
            model_path: Arc::new(Mutex::new(String::new())),
            dac_path: Arc::new(Mutex::new(String::new())),
            editor_state: EguiState::from_size(400, 300),
        }
    }
}

// ---------------------------------------------------------------------------
// Communication
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct Metrics {
    input_rms: f32,
    output_rms: f32,
    rtf: f32,
    pipeline_ready: bool,
}

enum Task {
    LoadModels {
        dac_path: String,
        converter_path: String,
    },
    SetRingBuffers {
        capture_rx: rtrb::Consumer<f32>,
        playback_tx: rtrb::Producer<f32>,
    },
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

struct LightVcPlugin {
    params: Arc<LightVcParams>,
    task_tx: Sender<Task>,
    metrics_rx: Receiver<Metrics>,
    pipeline_ready: Arc<AtomicBool>,
    metrics: Arc<Mutex<Metrics>>,
    // Ring buffer handles for process() (host audio thread)
    capture_tx: Option<rtrb::Producer<f32>>,
    playback_rx: Option<rtrb::Consumer<f32>>,
    // Latency in samples (reported to host)
    latency_samples: u32,
}

impl Default for LightVcPlugin {
    fn default() -> Self {
        let (task_tx, task_rx) = unbounded();
        let (metrics_tx, metrics_rx) = unbounded();
        let pipeline_ready = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(Mutex::new(Metrics::default()));

        let pr = pipeline_ready.clone();
        let mt = metrics.clone();

        std::thread::spawn(move || {
            inference_thread(task_rx, metrics_tx, pr, mt);
        });

        Self {
            params: Arc::new(LightVcParams::default()),
            task_tx,
            metrics_rx,
            pipeline_ready,
            metrics,
            capture_tx: None,
            playback_rx: None,
            latency_samples: 0,
        }
    }
}

impl Plugin for LightVcPlugin {
    const NAME: &'static str = "LightVC-X";
    const VENDOR: &'static str = "LightVC";
    const URL: &'static str = "https://github.com/kjranyone/LightVC";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = "0.1.0";

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(1),
        main_output_channels: NonZeroU32::new(1),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames {
            layout: Some("LightVC Mono"),
            main_input: Some("Input"),
            main_output: Some("Output"),
            aux_inputs: &[],
            aux_outputs: &[],
        },
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = Task;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let metrics = self.metrics.clone();
        let ready = self.pipeline_ready.load(Ordering::Relaxed);

        struct EditorUserState {
            metrics: Arc<Mutex<Metrics>>,
            params: Arc<LightVcParams>,
            ready: bool,
        }

        create_egui_editor(
            self.params.editor_state.clone(),
            EditorUserState {
                metrics,
                params,
                ready,
            },
            nice_plug_egui::EguiSettings::default(),
            |_ctx, _queue, _state| {},
            move |ui, setter, _queue, state| {
                let m = state.metrics.lock().unwrap().clone();
                let params = state.params.clone();

                ui.heading("LightVC-X");
                ui.add_space(8.0);

                let color = if state.ready {
                    egui::Color32::from_rgb(80, 200, 80)
                } else {
                    egui::Color32::from_rgb(160, 160, 160)
                };
                ui.label(
                    egui::RichText::new(if state.ready {
                        "● READY"
                    } else {
                        "● NO MODEL"
                    })
                    .color(color)
                    .strong(),
                );

                ui.add_space(8.0);

                if state.ready {
                    let in_db = if m.input_rms > 0.0 {
                        20.0 * m.input_rms.log10()
                    } else {
                        -99.0
                    };
                    let out_db = if m.output_rms > 0.0 {
                        20.0 * m.output_rms.log10()
                    } else {
                        -99.0
                    };
                    ui.label(format!(
                        "In: {in_db:+.0}dB | Out: {out_db:+.0}dB | RTF: {:.2}",
                        m.rtf
                    ));
                }

                ui.add_space(12.0);

                ui.collapsing("Parameters", |ui| {
                    if ui
                        .button(if params.bypass.value() {
                            "Unbypass"
                        } else {
                            "Bypass"
                        })
                        .clicked()
                    {
                        setter.set_parameter(&params.bypass, !params.bypass.value());
                    }
                    ui.add_space(4.0);
                    ui.label(format!("Mix: {:.0}%", params.mix.value()));
                    ui.label(format!("Gain: {:+.1}dB", params.output_gain.value()));
                    let mode = params.mode.value();
                    ui.label(format!(
                        "Mode: {}",
                        match mode {
                            0 => "Strict",
                            1 => "Balanced",
                            2 => "Quality",
                            _ => "?",
                        }
                    ));
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = buffer_config.sample_rate;
        let cap = (sr as usize / 5).max(16384);
        let (capture_tx, capture_rx) = rtrb::RingBuffer::new(cap);
        let (playback_tx, playback_rx) = rtrb::RingBuffer::new(cap);

        // Keep write/playback ends in plugin for process()
        self.capture_tx = Some(capture_tx);
        self.playback_rx = Some(playback_rx);

        // Send read/playback-write ends to inference thread
        let _ = self.task_tx.send(Task::SetRingBuffers {
            capture_rx,
            playback_tx,
        });

        // Report initial latency (balanced mode: ~4 chunks at 44100Hz)
        // chunk = 4 * 512 = 2048 samples at 44100Hz
        // Rescaled to host sample rate
        let chunk_44k = 2048.0_f32;
        let latency_44k = chunk_44k * 3.0; // 3 chunks of buffer + processing
        self.latency_samples = (latency_44k * sr / 44100.0) as u32;
        context.set_latency_samples(self.latency_samples);

        let model = self.params.model_path.lock().unwrap().clone();
        let dac = self.params.dac_path.lock().unwrap().clone();
        if !model.is_empty() && !dac.is_empty() {
            let _ = self.task_tx.send(Task::LoadModels {
                dac_path: dac,
                converter_path: model,
            });
        }

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Update metrics
        {
            let mut m = self.metrics.lock().unwrap();
            while let Ok(r) = self.metrics_rx.try_recv() {
                *m = r;
            }
        }

        let bypass = self.params.bypass.value();
        let mix = self.params.mix.smoothed.next() / 100.0;
        let gain_db = self.params.output_gain.smoothed.next();
        let gain_linear = 10.0f32.powf(gain_db / 20.0);

        if bypass || !self.pipeline_ready.load(Ordering::Relaxed) {
            // Bypass: dry pass-through with gain
            for channel_samples in buffer.iter_samples() {
                for sample in channel_samples {
                    *sample *= gain_linear;
                }
            }
            return ProcessStatus::Normal;
        }

        let (Some(tx), Some(rx)) = (self.capture_tx.as_mut(), self.playback_rx.as_mut()) else {
            return ProcessStatus::Normal;
        };

        // Push input samples to capture ring buffer, pop from playback ring buffer
        for channel_samples in buffer.iter_samples() {
            for sample in channel_samples {
                let input = *sample;
                let _ = tx.push(input);
                let processed = rx.pop().unwrap_or(0.0);
                // Dry/wet mix + output gain
                *sample = (input * (1.0 - mix) + processed * mix) * gain_linear;
            }
        }

        ProcessStatus::Normal
    }
}

// ---------------------------------------------------------------------------
// Inference thread
// ---------------------------------------------------------------------------

fn inference_thread(
    task_rx: Receiver<Task>,
    metrics_tx: Sender<Metrics>,
    pipeline_ready: Arc<AtomicBool>,
    metrics: Arc<Mutex<Metrics>>,
) {
    let mut pipeline: Option<Arc<Mutex<lightvc_core::pipeline::VcPipeline>>> = None;
    let mut capture_rx: Option<rtrb::Consumer<f32>> = None;
    let mut playback_tx: Option<rtrb::Producer<f32>> = None;

    loop {
        while let Ok(task) = task_rx.try_recv() {
            match task {
                Task::SetRingBuffers {
                    capture_rx: crx,
                    playback_tx: ptx,
                } => {
                    capture_rx = Some(crx);
                    playback_tx = Some(ptx);
                }
                Task::LoadModels {
                    dac_path,
                    converter_path,
                } => match load_pipeline(&dac_path, &converter_path) {
                    Ok(p) => {
                        pipeline = Some(Arc::new(Mutex::new(p)));
                        pipeline_ready.store(true, Ordering::Relaxed);
                        let mut m = metrics.lock().unwrap();
                        m.pipeline_ready = true;
                        drop(m);
                    }
                    Err(e) => {
                        nice_log!("Model load failed: {e}");
                        pipeline_ready.store(false, Ordering::Relaxed);
                    }
                },
            }
        }

        let (Some(p), Some(crx), Some(ptx)) = (&pipeline, &mut capture_rx, &mut playback_tx) else {
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        };

        // Run inference loop
        let chunk_sz = p.lock().map(|pl| pl.chunk_samples()).unwrap_or(2048);
        let needed = chunk_sz;

        let mut cap = Vec::with_capacity(needed);
        while cap.len() < needed {
            match crx.pop() {
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

        let in_rms = rms(&cap);
        let out = match p.lock() {
            Ok(mut pl) => pl.process_chunk(&cap).unwrap_or_else(|e| {
                nice_log!("VC: {e}");
                cap.clone()
            }),
            Err(_) => continue,
        };

        let out_rms = rms(&out);
        for s in &out {
            let _ = ptx.push(*s);
        }
        let _ = metrics_tx.send(Metrics {
            input_rms: in_rms,
            output_rms: out_rms,
            rtf: 0.0, // TODO: measure
            pipeline_ready: true,
        });
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn load_pipeline(
    dac_path: &str,
    converter_path: &str,
) -> anyhow::Result<lightvc_core::pipeline::VcPipeline> {
    let device = candle_core::Device::Cpu;
    let dac_config = lightvc_core::DacConfig::default();

    let vb = lightvc_core::weights::load_varbuilder(
        std::path::Path::new(converter_path),
        candle_core::DType::F32,
        &device,
    )?;
    let config = lightvc_core::converter::ConverterConfig::default();
    let converter = lightvc_core::converter::AnyConverter::new(config, vb)?;

    lightvc_core::pipeline::VcPipeline::new(
        std::path::Path::new(dac_path),
        &dac_config,
        converter,
        lightvc_core::converter::LatencyMode::Balanced,
        device,
    )
}

// ---------------------------------------------------------------------------
// VST3 export
// ---------------------------------------------------------------------------

impl Vst3Plugin for LightVcPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"LightVCXPluginID";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Tools];
}

nice_export_vst3!(LightVcPlugin);

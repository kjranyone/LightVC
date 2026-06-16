//! LightVC-X VST3 Plugin
//!
//! Real-time voice conversion as a VST3 audio effect.
//! Uses nice-plug (community fork of nih_plug).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{unbounded, Receiver, Sender};
use nice_plug::prelude::*;

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
}

enum Task {
    LoadModels {
        dac_path: String,
        converter_path: String,
    },
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

struct LightVcPlugin {
    params: Arc<LightVcParams>,
    capture_tx: Mutex<Option<rtrb::Producer<f32>>>,
    playback_rx: Mutex<Option<rtrb::Consumer<f32>>>,
    task_tx: Sender<Task>,
    metrics_rx: Receiver<Metrics>,
    pipeline_ready: AtomicBool,
    metrics: Mutex<Metrics>,
}

impl Default for LightVcPlugin {
    fn default() -> Self {
        let (task_tx, task_rx) = unbounded();
        let (metrics_tx, metrics_rx) = unbounded();

        std::thread::spawn(move || {
            inference_thread(task_rx, metrics_tx);
        });

        Self {
            params: Arc::new(LightVcParams::default()),
            capture_tx: Mutex::new(None),
            playback_rx: Mutex::new(None),
            task_tx,
            metrics_rx,
            pipeline_ready: AtomicBool::new(false),
            metrics: Mutex::new(Metrics::default()),
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

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let cap = (buffer_config.sample_rate as usize / 5).max(16384);
        let (capture_tx, _capture_rx) = rtrb::RingBuffer::new(cap);
        let (_playback_tx, playback_rx) = rtrb::RingBuffer::new(cap);

        *self.capture_tx.lock().unwrap() = Some(capture_tx);
        *self.playback_rx.lock().unwrap() = Some(playback_rx);

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
        let bypass = self.params.bypass.value();
        let mix = self.params.mix.smoothed.next() / 100.0;
        let gain_db = self.params.output_gain.smoothed.next();
        let gain_linear = 10.0f32.powf(gain_db / 20.0);

        {
            let mut metrics = self.metrics.lock().unwrap();
            while let Ok(m) = self.metrics_rx.try_recv() {
                *metrics = m;
            }
        }

        if bypass || !self.pipeline_ready.load(Ordering::Relaxed) {
            for channel_samples in buffer.iter_samples() {
                for sample in channel_samples {
                    *sample *= gain_linear;
                }
            }
            return ProcessStatus::Normal;
        }

        let mut cap_tx = self.capture_tx.lock().unwrap();
        let mut pb_rx = self.playback_rx.lock().unwrap();

        let (Some(tx), Some(rx)) = (cap_tx.as_mut(), pb_rx.as_mut()) else {
            return ProcessStatus::Normal;
        };

        for channel_samples in buffer.iter_samples() {
            for sample in channel_samples {
                let _ = tx.push(*sample);
                let processed = rx.pop().unwrap_or(0.0);
                *sample = (*sample * (1.0 - mix) + processed * mix) * gain_linear;
            }
        }

        ProcessStatus::Normal
    }
}

// ---------------------------------------------------------------------------
// Inference thread
// ---------------------------------------------------------------------------

fn inference_thread(task_rx: Receiver<Task>, metrics_tx: Sender<Metrics>) {
    let mut _pipeline: Option<Arc<Mutex<lightvc_core::pipeline::VcPipeline>>> = None;

    loop {
        while let Ok(task) = task_rx.try_recv() {
            match task {
                Task::LoadModels {
                    dac_path,
                    converter_path,
                } => match load_pipeline(&dac_path, &converter_path) {
                    Ok(p) => {
                        _pipeline = Some(Arc::new(Mutex::new(p)));
                        let _ = metrics_tx.send(Metrics::default());
                    }
                    Err(e) => {
                        nice_log!("Model load failed: {e}");
                    }
                },
            }
        }

        // TODO: wire ring buffers and run VC inference loop
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
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

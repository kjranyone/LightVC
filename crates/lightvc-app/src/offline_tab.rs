//! Tab 1: Offline voice conversion.
//! Source WAV + Reference WAV → Convert → Preview playback.

use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_file_dialog::FileDialog;

use crate::app::AppState;
use crate::audio_playback::{self, AudioPlayer};

#[derive(Default)]
pub struct OfflineState {
    pub source_path: String,
    pub reference_path: String,
    pub output_path: String,
    pub converting: bool,
    pub converted_samples: Option<Vec<f32>>,
    pub player: Option<AudioPlayer>,
    pub source_preview: Option<AudioPlayer>,
    pub reference_preview: Option<AudioPlayer>,
    pub pick_target: Option<String>,
}

pub fn render(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    file_dialog: &mut FileDialog,
    state: &Arc<Mutex<AppState>>,
    offline: &mut OfflineState,
) {
    let has_converter = state.lock().unwrap().converter_weights.is_some();

    ui.heading("Offline Voice Conversion");
    ui.add_space(8.0);

    // Source file
    ui.horizontal(|ui| {
        ui.label("Source:");
        ui.text_edit_singleline(&mut offline.source_path);
        if ui.button("Browse").clicked() {
            offline.pick_target = Some("source".to_string());
            file_dialog.pick_file();
        }
        if ui.button("▶").clicked() && !offline.source_path.is_empty() {
            let path = std::path::PathBuf::from(&offline.source_path);
            if let Ok((wav, sr)) = audio_playback::load_wav_mono(&path) {
                let wav44 = audio_playback::resample_linear(&wav, sr);
                offline.source_preview = AudioPlayer::play(wav44).ok();
            }
        }
    });

    // Reference file
    ui.horizontal(|ui| {
        ui.label("Reference:");
        ui.text_edit_singleline(&mut offline.reference_path);
        if ui.button("Browse").clicked() {
            offline.pick_target = Some("reference".to_string());
            file_dialog.pick_file();
        }
        if ui.button("▶").clicked() && !offline.reference_path.is_empty() {
            let path = std::path::PathBuf::from(&offline.reference_path);
            if let Ok((wav, sr)) = audio_playback::load_wav_mono(&path) {
                let wav44 = audio_playback::resample_linear(&wav, sr);
                offline.reference_preview = AudioPlayer::play(wav44).ok();
            }
        }
    });

    ui.add_space(8.0);

    // Quick-pick from voice catalog
    {
        let s = state.lock().unwrap();
        if !s.voices.is_empty() {
            ui.label("Or pick from Voice Catalog:");
            ui.horizontal_wrapped(|ui| {
                for voice in &s.voices {
                    if ui.small_button(&voice.name).clicked() {
                        offline.reference_path = voice.path.to_string_lossy().into_owned();
                    }
                }
            });
        }
    }

    ui.add_space(8.0);

    // Convert button
    ui.horizontal(|ui| {
        let can_convert = !offline.source_path.is_empty()
            && !offline.reference_path.is_empty()
            && has_converter
            && !offline.converting;

        ui.add_enabled_ui(can_convert, |ui| {
            if ui.button("Convert").clicked() {
                offline.converting = true;
                let state_clone = state.clone();
                let src = offline.source_path.clone();
                let refp = offline.reference_path.clone();
                std::thread::spawn(move || {
                    run_offline_conversion(state_clone, &src, &refp);
                });
            }
        });

        if offline.converting {
            ui.spinner();
            ui.label("Converting...");
        }
    });

    ui.add_space(8.0);

    // Output
    if let Some(ref samples) = offline.converted_samples {
        ui.label(format!(
            "Output: {} samples ({:.1}s)",
            samples.len(),
            samples.len() as f32 / 44100.0
        ));
        ui.horizontal(|ui| {
            if ui.button("▶ Play Output").clicked() {
                offline.player = AudioPlayer::play(samples.clone()).ok();
            }
            if ui.button("Save As...").clicked() {
                if let Some(path) = rfd::FileDialog::new().save_file() {
                    let _ = audio_playback::save_wav_mono(&path, samples, 44100);
                }
            }
        });
    }

    if !has_converter {
        ui.add_space(8.0);
        ui.colored_label(
            egui::Color32::from_rgb(200, 180, 80),
            "No converter loaded. Set model weights in Realtime tab.",
        );
    }

    // Check conversion status
    {
        let s = state.lock().unwrap();
        if offline.converting && !s.status.starts_with("Convert") {
            offline.converting = false;
        }
    }

    // Handle file dialog result
    if let Some(path) = file_dialog.take_picked() {
        if let Some(ref target) = offline.pick_target {
            match target.as_str() {
                "source" => offline.source_path = path.to_string_lossy().into_owned(),
                "reference" => offline.reference_path = path.to_string_lossy().into_owned(),
                _ => {}
            }
        }
        offline.pick_target = None;
    }
}

fn run_offline_conversion(state: Arc<Mutex<AppState>>, source_path: &str, reference_path: &str) {
    {
        let mut s = state.lock().unwrap();
        s.status = "Converting...".to_string();
        s.error = None;
    }

    let result = (|| -> anyhow::Result<Vec<f32>> {
        let pipeline = {
            let s = state.lock().unwrap();
            s.pipeline
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Pipeline not initialized"))?
        };

        let (src_wav, src_sr) = audio_playback::load_wav_mono(std::path::Path::new(source_path))?;
        let (ref_wav, ref_sr) =
            audio_playback::load_wav_mono(std::path::Path::new(reference_path))?;

        let src_44k = audio_playback::resample_linear(&src_wav, src_sr);
        let ref_44k = audio_playback::resample_linear(&ref_wav, ref_sr);

        let src_padded = lightvc_core::codec::pad_to_hop(src_44k);
        let ref_padded = lightvc_core::codec::pad_to_hop(ref_44k);

        let mut p = pipeline.lock().unwrap();
        p.reset();
        p.set_target(&ref_padded)?;

        let chunk_size = p.chunk_samples();
        let mut output = Vec::with_capacity(src_padded.len());
        let mut i = 0;
        while i < src_padded.len() {
            let end = (i + chunk_size).min(src_padded.len());
            let mut chunk = src_padded[i..end].to_vec();
            if chunk.len() < chunk_size {
                chunk.resize(chunk_size, 0.0);
            }
            let out = p.process_chunk(&chunk)?;
            output.extend_from_slice(&out[..end - i]);
            i = end;
        }
        Ok(output)
    })();

    match result {
        Ok(samples) => {
            let mut s = state.lock().unwrap();
            s.status = format!("Converted: {} samples", samples.len());
            s.error = None;
        }
        Err(e) => {
            let mut s = state.lock().unwrap();
            s.error = Some(format!("Conversion failed: {e}"));
            s.status = "Error".to_string();
        }
    }
}

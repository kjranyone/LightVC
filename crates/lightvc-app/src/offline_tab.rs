//! Tab 1: Offline voice conversion.
//! Source WAV + Reference WAV → Convert → Preview playback.

use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_file_dialog::FileDialog;

use crate::app::AppState;
use crate::audio_playback::{self, AudioPlayer};

const LABEL_WIDTH: f32 = 80.0;

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
    let panel_width = ui.available_width();

    crate::theme::heading(ui, "Offline Voice Conversion");
    ui.add_space(8.0);

    // Use centered column with max width for readability
    egui::Frame::NONE.show(ui, |ui| {
        egui::Frame::NONE.show(ui, |ui| {
            ui.set_max_width(520.0);

            // --- Source ---
            crate::theme::info_card(ui, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.add_sized(
                            [LABEL_WIDTH, 20.0],
                            egui::Label::new(
                                egui::RichText::new("Source")
                                    .size(13.0)
                                    .color(crate::theme::colors::TEXT_DIM),
                            ),
                        );
                        ui.add_sized(
                            [ui.available_width(), 20.0],
                            egui::TextEdit::singleline(&mut offline.source_path)
                                .hint_text("source audio file"),
                        );
                    });
                    ui.add_space(6.0);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if crate::theme::pill_button(ui, "▶ Play", !offline.source_path.is_empty())
                            && !offline.source_path.is_empty()
                        {
                            play_audio(&offline.source_path);
                        }
                        ui.add_space(4.0);
                        if crate::theme::pill_button(ui, "Browse", true) {
                            offline.pick_target = Some("source".to_string());
                            file_dialog.pick_file();
                        }
                    });
                });
            });

            ui.add_space(8.0);

            // --- Reference ---
            crate::theme::info_card(ui, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.add_sized(
                            [LABEL_WIDTH, 20.0],
                            egui::Label::new(
                                egui::RichText::new("Reference")
                                    .size(13.0)
                                    .color(crate::theme::colors::TEXT_DIM),
                            ),
                        );
                        ui.add_sized(
                            [ui.available_width(), 20.0],
                            egui::TextEdit::singleline(&mut offline.reference_path)
                                .hint_text("target voice reference"),
                        );
                    });
                    ui.add_space(6.0);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if crate::theme::pill_button(
                            ui,
                            "▶ Play",
                            !offline.reference_path.is_empty(),
                        ) && !offline.reference_path.is_empty()
                        {
                            play_audio(&offline.reference_path);
                        }
                        ui.add_space(4.0);
                        if crate::theme::pill_button(ui, "Browse", true) {
                            offline.pick_target = Some("reference".to_string());
                            file_dialog.pick_file();
                        }
                    });
                });
            });

            ui.add_space(8.0);

            // --- Voice catalog quick-pick ---
            {
                let s = state.lock().unwrap();
                if !s.voices.is_empty() {
                    ui.label(
                        egui::RichText::new("Or pick from Voice Catalog")
                            .size(12.0)
                            .color(crate::theme::colors::CYAN),
                    );
                    ui.add_space(2.0);
                    ui.horizontal_wrapped(|ui| {
                        for voice in &s.voices {
                            if crate::theme::pill_button(ui, &voice.name, false) {
                                offline.reference_path = voice.path.to_string_lossy().into_owned();
                            }
                        }
                    });
                }
            }

            ui.add_space(12.0);

            // --- Convert CTA ---
            ui.horizontal(|ui| {
                let can_convert = !offline.source_path.is_empty()
                    && !offline.reference_path.is_empty()
                    && has_converter
                    && !offline.converting;

                let btn =
                    egui::Button::new(egui::RichText::new("✦ Convert").size(18.0).strong().color(
                        if can_convert {
                            crate::theme::colors::TEXT
                        } else {
                            crate::theme::colors::TEXT_MUTED
                        },
                    ))
                    .fill(if can_convert {
                        crate::theme::colors::PINK
                    } else {
                        crate::theme::colors::BG_PANEL
                    })
                    .stroke(egui::Stroke::new(
                        2.0,
                        if can_convert {
                            crate::theme::colors::PINK_BRIGHT
                        } else {
                            crate::theme::colors::TEXT_MUTED
                        },
                    ))
                    .min_size(egui::vec2(160.0, 42.0));

                ui.add_enabled_ui(can_convert, |ui| {
                    if ui.add(btn).clicked() {
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
                    ui.label(
                        egui::RichText::new("Converting...")
                            .size(13.0)
                            .color(crate::theme::colors::LAVENDER),
                    );
                }
            });

            ui.add_space(8.0);

            // --- Output ---
            if let Some(ref samples) = offline.converted_samples {
                crate::theme::info_card(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new("✦ Output")
                                .size(14.0)
                                .strong()
                                .color(crate::theme::colors::MINT),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "{} samples ({:.1}s)",
                                samples.len(),
                                samples.len() as f32 / 44100.0
                            ))
                            .size(12.0)
                            .color(crate::theme::colors::TEXT_DIM),
                        );
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if crate::theme::pill_button(ui, "▶ Play Output", true) {
                                offline.player = AudioPlayer::play(samples.clone()).ok();
                            }
                            ui.add_space(4.0);
                            if crate::theme::pill_button(ui, "Save As...", true) {
                                if let Some(path) = rfd::FileDialog::new().save_file() {
                                    let _ = audio_playback::save_wav_mono(&path, samples, 44100);
                                }
                            }
                        });
                    });
                });
            }

            if !has_converter {
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(
                        "⚠ No converter loaded. Set model weights in Realtime tab.",
                    )
                    .size(12.0)
                    .color(crate::theme::colors::YELLOW),
                );
            }
        });
    });

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

fn play_audio(path_str: &str) {
    let path = std::path::PathBuf::from(path_str);
    if let Ok((wav, sr)) = audio_playback::load_wav_mono(&path) {
        let wav44 = audio_playback::resample_linear(&wav, sr);
        let _ = AudioPlayer::play(wav44);
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

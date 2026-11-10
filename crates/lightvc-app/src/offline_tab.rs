//! Tab 1: Offline voice conversion.
//! Source WAV + 目標の声の参照 → SnvStream(48 kHz・240 サンプル)→ Preview playback / Save.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use eframe::egui;
use lightvc_core::snv_stream::SnvStream;

use crate::app::{AppState, RtControl};
use crate::audio_playback::{self, AudioPlayer};
use crate::file_pick::FilePick;

const SR: u32 = 48_000;
const SEMITONES: [(f32, &str); 4] = [(0.0, "+0"), (5.0, "+5"), (7.0, "+7"), (12.0, "+12")];

#[derive(Default)]
pub struct OfflineState {
    pub source_path: String,
    pub reference_path: String,
    pub converting: bool,
    /// 変換の出力(48 kHz・入力と時刻を揃えた)
    pub converted_samples: Option<Vec<f32>>,
    pub player: Option<AudioPlayer>,
    pub source_preview: Option<AudioPlayer>,
    pub reference_preview: Option<AudioPlayer>,
    /// One picker per potential target — avoids a shared mutable flag.
    pub source_pick: FilePick,
    pub reference_pick: FilePick,
}

#[allow(clippy::too_many_arguments)]
pub fn render(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    state: &Arc<Mutex<AppState>>,
    offline: &mut OfflineState,
    icon_folder: &egui::TextureHandle,
    icon_play: &egui::TextureHandle,
    icon_convert: &egui::TextureHandle,
    icon_speaker: &egui::TextureHandle,
    icon_mic: &egui::TextureHandle,
) {
    let (has_operator, semitones_before) = {
        let s = state.lock().unwrap();
        (!s.settings.operator.is_empty(), s.settings.semitones)
    };
    let mut semitones = semitones_before;

    ui.vertical_centered(|ui| {
        egui::Frame::NONE.show(ui, |ui| {
            ui.set_max_width(520.0);
            ui.set_min_width(300.0);

            // --- Source ---
            crate::theme::info_card(ui, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Image::from_texture(icon_mic)
                                .fit_to_exact_size(egui::vec2(16.0, 16.0)),
                        );
                        crate::theme::form_label(ui, "Source");
                        crate::theme::path_text_edit(
                            ui,
                            &mut offline.source_path,
                            "source audio file",
                        );
                    });
                    ui.add_space(crate::theme::space::SMALL + 2.0);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                        let src_playing = offline
                            .source_preview
                            .as_ref()
                            .map(|p| p.is_playing())
                            .unwrap_or(false);
                        let src_label = if src_playing { "Stop" } else { "Play" };
                        let src_active = !offline.source_path.is_empty() || src_playing;
                        if crate::theme::icon_button(ui, icon_play, src_label, src_active) {
                            if src_playing {
                                if let Some(p) = offline.source_preview.take() {
                                    p.stop();
                                }
                            } else if !offline.source_path.is_empty() {
                                offline.source_preview = play_audio(&offline.source_path);
                            }
                        }
                        ui.add_space(crate::theme::space::SMALL);
                        if crate::theme::icon_button(ui, icon_folder, "Browse", true) {
                            offline.source_pick.open(ctx);
                        }
                    });
                });
            });

            ui.add_space(crate::theme::space::MEDIUM);

            // --- Reference ---
            crate::theme::info_card(ui, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Image::from_texture(icon_speaker)
                                .fit_to_exact_size(egui::vec2(16.0, 16.0)),
                        );
                        crate::theme::form_label(ui, "Reference");
                        crate::theme::path_text_edit(
                            ui,
                            &mut offline.reference_path,
                            "target voice reference (20 s+)",
                        );
                    });
                    ui.add_space(crate::theme::space::SMALL + 2.0);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                        let ref_playing = offline
                            .reference_preview
                            .as_ref()
                            .map(|p| p.is_playing())
                            .unwrap_or(false);
                        let ref_label = if ref_playing { "Stop" } else { "Play" };
                        let ref_active = !offline.reference_path.is_empty() || ref_playing;
                        if crate::theme::icon_button(ui, icon_play, ref_label, ref_active) {
                            if ref_playing {
                                if let Some(p) = offline.reference_preview.take() {
                                    p.stop();
                                }
                            } else if !offline.reference_path.is_empty() {
                                offline.reference_preview = play_audio(&offline.reference_path);
                            }
                        }
                        ui.add_space(crate::theme::space::SMALL);
                        if crate::theme::icon_button(ui, icon_folder, "Browse", true) {
                            offline.reference_pick.open(ctx);
                        }
                    });
                });
            });

            ui.add_space(crate::theme::space::MEDIUM);

            // --- Voice catalog quick-pick ---
            {
                let s = state.lock().unwrap();
                if !s.voices.is_empty() {
                    ui.label(
                        egui::RichText::new("Or pick from Voice Catalog")
                            .size(12.0)
                            .color(crate::theme::colors::TEXT_DIM),
                    );
                    ui.add_space(crate::theme::space::TIGHT);
                    ui.horizontal_wrapped(|ui| {
                        for voice in &s.voices {
                            if crate::theme::pill_button(ui, &voice.name, false) {
                                offline.reference_path = voice.path.to_string_lossy().into_owned();
                            }
                        }
                    });
                }
            }

            ui.add_space(crate::theme::space::LARGE);

            // --- Pitch (固定の半音・Realtime と共通) ---
            crate::theme::info_card(ui, |ui| {
                ui.horizontal(|ui| {
                    crate::theme::subheading(ui, "Pitch");
                    ui.add_space(crate::theme::space::SMALL);
                    for (k, name) in SEMITONES {
                        if crate::theme::pill_button(ui, name, semitones == k) {
                            semitones = k;
                        }
                    }
                });
            });

            ui.add_space(crate::theme::space::LARGE);

            // --- Convert CTA ---
            let can_convert = !offline.source_path.is_empty()
                && !offline.reference_path.is_empty()
                && has_operator
                && !offline.converting;

            ui.horizontal_centered(|ui| {
                let btn = egui::Button::image_and_text(
                    egui::Image::from_texture(icon_convert)
                        .fit_to_exact_size(egui::vec2(16.0, 16.0)),
                    egui::RichText::new("Convert")
                        .size(18.0)
                        .strong()
                        .color(if can_convert {
                            egui::Color32::from_rgb(0xFF, 0xFF, 0xFF)
                        } else {
                            crate::theme::colors::TEXT_MUTED
                        }),
                )
                .fill(if can_convert {
                    egui::Color32::from_rgb(0x1E, 0x1E, 0x2E)
                } else {
                    crate::theme::colors::BG_PANEL
                })
                .min_size(egui::vec2(160.0, crate::theme::CTA_HEIGHT));

                ui.add_enabled_ui(can_convert, |ui| {
                    if ui.add(btn).clicked() {
                        offline.converted_samples = None;
                        offline.player = None;
                        {
                            let mut s = state.lock().unwrap();
                            s.offline_result = None;
                            s.error = None;
                            s.status = "Converting...".to_string();
                        }
                        offline.converting = true;
                        let st = state.clone();
                        let src = PathBuf::from(&offline.source_path);
                        let refp = PathBuf::from(&offline.reference_path);
                        std::thread::spawn(move || {
                            run_offline_conversion(st, src, refp);
                        });
                    }
                });

                if offline.converting {
                    ui.spinner();
                    ui.label(
                        egui::RichText::new("Converting...")
                            .size(13.0)
                            .color(crate::theme::colors::TEXT_DIM),
                    );
                }
            });

            ui.add_space(crate::theme::space::MEDIUM);

            // --- Output ---
            if let Some(ref samples) = offline.converted_samples {
                crate::theme::info_card(ui, |ui| {
                    ui.vertical(|ui| {
                        crate::theme::subheading(ui, "Output");
                        ui.label(
                            egui::RichText::new(format!(
                                "{} samples ({:.1}s, 48 kHz)",
                                samples.len(),
                                samples.len() as f32 / SR as f32
                            ))
                            .size(12.0)
                            .color(crate::theme::colors::TEXT_DIM),
                        );
                        ui.add_space(crate::theme::space::SMALL + 2.0);
                        // right_to_left so Play stays on the right,
                        // matching Source/Reference cards.
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                            let out_playing = offline
                                .player
                                .as_ref()
                                .map(|p| p.is_playing())
                                .unwrap_or(false);
                            let out_label = if out_playing { "Stop" } else { "Play Output" };
                            if crate::theme::icon_button(ui, icon_play, out_label, true) {
                                if out_playing {
                                    if let Some(p) = offline.player.take() {
                                        p.stop();
                                    }
                                } else {
                                    let wav44 = audio_playback::resample_linear(samples, SR);
                                    offline.player = AudioPlayer::play(wav44).ok();
                                }
                            }
                            ui.add_space(crate::theme::space::SMALL);
                            if crate::theme::icon_button(ui, icon_speaker, "Save As...", true)
                                && let Some(path) = rfd::FileDialog::new().save_file()
                            {
                                let _ = audio_playback::save_wav_mono(&path, samples, SR);
                            }
                        });
                    });
                });
            }

            if !has_operator {
                ui.add_space(crate::theme::space::MEDIUM);
                ui.label(
                    egui::RichText::new("No operator enrollment. Add operator WAV in the Realtime tab.")
                        .size(12.0)
                        .color(crate::theme::colors::ERROR),
                );
            }
        });
    }); // vertical_centered

    if semitones != semitones_before {
        let mut s = state.lock().unwrap();
        s.settings.semitones = semitones;
        if let Some(ref tx) = s.rt_control_tx {
            let _ = tx.send(RtControl::SetSemitones(semitones));
        }
    }

    // Check conversion status and pick up result
    {
        let mut s = state.lock().unwrap();
        if offline.converting && (s.status.starts_with("Converted:") || s.error.is_some()) {
            offline.converting = false;
        }
        if let Some(r) = s.offline_result.take() {
            offline.converted_samples = Some(r);
        }
    }

    // Handle file picks (rfd runs off-thread; results land here next frame).
    if let Some(path) = offline.source_pick.take() {
        offline.source_path = path.to_string_lossy().into_owned();
    }
    if let Some(path) = offline.reference_pick.take() {
        offline.reference_path = path.to_string_lossy().into_owned();
    }
}

fn play_audio(path_str: &str) -> Option<AudioPlayer> {
    let path = std::path::PathBuf::from(path_str);
    if let Ok((wav, sr)) = audio_playback::load_wav_mono(&path) {
        let wav44 = audio_playback::resample_linear(&wav, sr);
        return AudioPlayer::play(wav44).ok();
    }
    None
}

/// 240 サンプルずつ流し、経路の遅れ(LATENCY_SAMPLES)を落として入力と時刻を揃える。
fn convert(stream: &mut SnvStream, x: &[f32]) -> Vec<f32> {
    const HOP: usize = SnvStream::BLOCK;
    let lat = SnvStream::LATENCY_SAMPLES;
    let n = (x.len() + lat).div_ceil(HOP);
    let mut xin = x.to_vec();
    xin.resize(n * HOP, 0.0);
    let mut y = vec![0f32; n * HOP];
    for (xi, yi) in xin.chunks_exact(HOP).zip(y.chunks_exact_mut(HOP)) {
        stream.process_block(xi, yi);
    }
    y.drain(..lat);
    y.truncate(x.len());
    let pk = y.iter().fold(0f32, |a, v| a.max(v.abs()));
    if pk > 0.99 {
        y.iter_mut().for_each(|v| *v *= 0.99 / pk);
    }
    y
}

fn run_offline_conversion(state: Arc<Mutex<AppState>>, source: PathBuf, reference: PathBuf) {
    let cfg = state.lock().unwrap().settings.clone();
    let result = (|| -> anyhow::Result<Vec<f32>> {
        let mut stream = crate::app::build_stream(&cfg, &[reference])?;
        let x = crate::app::load_48k(&source)?;
        Ok(convert(&mut stream, &x))
    })();

    let mut s = state.lock().unwrap();
    match result {
        Ok(samples) => {
            s.status = format!("Converted: {} samples", samples.len());
            s.error = None;
            s.offline_result = Some(samples);
        }
        Err(e) => {
            s.error = Some(format!("Conversion failed: {e}"));
            s.status = "Error".to_string();
        }
    }
}

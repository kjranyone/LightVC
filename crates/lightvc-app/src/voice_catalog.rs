//! Tab 3: Voice catalog — manage zero-shot reference voices.
//! Register WAV files as named voice profiles for quick reuse.

use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_file_dialog::FileDialog;

use crate::app::AppState;
use crate::app::VoiceEntry;
use crate::audio_playback;

pub fn render(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    file_dialog: &mut FileDialog,
    state: &Arc<Mutex<AppState>>,
) {
    ui.heading("Voice Catalog");
    ui.add_space(8.0);

    ui.label("Register reference audio files as named voice profiles for zero-shot VC.");
    ui.add_space(8.0);

    // Add new voice
    ui.horizontal(|ui| {
        ui.label("Add voice:");
        let new_name = ctx.data_mut(|d| {
            d.get_temp_mut_or_insert_with::<String>("catalog_new_name".into(), || String::new())
                .clone()
        });
        let mut name_buf = new_name;
        ui.text_edit_singleline(&mut name_buf);
        if ui.button("Browse WAV...").clicked() {
            file_dialog.pick_file();
            ctx.data_mut(|d| d.insert_temp("catalog_pick".into(), true));
        }
        if ui.button("Add").clicked() && !name_buf.is_empty() {
            let picked =
                ctx.data_mut(|d| d.get_temp::<std::path::PathBuf>("catalog_picked".into()));
            if let Some(path) = picked {
                let mut s = state.lock().unwrap();
                s.voices.push(VoiceEntry {
                    name: name_buf.clone(),
                    path,
                });
                s.status = format!("Added voice: {}", name_buf);
                ctx.data_mut(|d| {
                    d.remove_temp::<std::path::PathBuf>("catalog_picked".into());
                });
                name_buf.clear();
            }
        }
        ctx.data_mut(|d| d.insert_temp("catalog_new_name".into(), name_buf));
    });

    ui.add_space(8.0);

    // Voice list
    {
        let s = state.lock().unwrap();
        if s.voices.is_empty() {
            ui.colored_label(
                egui::Color32::from_rgb(140, 140, 140),
                "No voices registered yet.",
            );
        } else {
            ui.label(format!("{} voices:", s.voices.len()));
            ui.add_space(4.0);

            let mut to_delete = None;
            for (i, voice) in s.voices.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("{}. {}", i + 1, voice.name));
                    ui.label(
                        egui::RichText::new(voice.path.to_string_lossy().as_ref())
                            .small()
                            .color(egui::Color32::from_rgb(140, 140, 140)),
                    );
                    if ui.small_button("▶").clicked() {
                        if let Ok((wav, sr)) = audio_playback::load_wav_mono(&voice.path) {
                            let wav44 = audio_playback::resample_linear(&wav, sr);
                            let _ = audio_playback::AudioPlayer::play(wav44);
                        }
                    }
                    if ui.small_button("✕").clicked() {
                        to_delete = Some(i);
                    }
                });
            }

            // Release lock before mutation
            drop(s);
            if let Some(idx) = to_delete {
                let mut s = state.lock().unwrap();
                let name = s.voices[idx].name.clone();
                s.voices.remove(idx);
                s.status = format!("Removed voice: {name}");
            }
        }
    }

    ui.add_space(12.0);

    // Export / Import catalog
    ui.collapsing("Catalog Import/Export", |ui| {
        if ui.button("Export to JSON").clicked() {
            let s = state.lock().unwrap();
            let catalog: Vec<_> = s
                .voices
                .iter()
                .map(|v| serde_json::json!({"name": v.name, "path": v.path.to_string_lossy()}))
                .collect();
            let json = serde_json::to_string_pretty(&catalog).unwrap_or_default();
            if let Some(path) = rfd::FileDialog::new().save_file() {
                let _ = std::fs::write(path, json);
            }
        }

        if ui.button("Import from JSON").clicked() {
            file_dialog.pick_file();
            ctx.data_mut(|d| d.insert_temp("catalog_import".into(), true));
        }
    });

    // Handle file dialog
    if let Some(path) = file_dialog.take_picked() {
        let is_import =
            ctx.data_mut(|d| d.get_temp::<bool>("catalog_import".into()).unwrap_or(false));
        if is_import {
            // Import JSON catalog
            if let Ok(json) = std::fs::read_to_string(&path) {
                if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(&json) {
                    let count = arr.len();
                    let mut s = state.lock().unwrap();
                    for entry in arr {
                        if let (Some(name), Some(path_str)) =
                            (entry["name"].as_str(), entry["path"].as_str())
                        {
                            s.voices.push(VoiceEntry {
                                name: name.to_string(),
                                path: std::path::PathBuf::from(path_str),
                            });
                        }
                    }
                    s.status = format!("Imported {} voices", count);
                }
            }
            ctx.data_mut(|d| d.remove_temp::<bool>("catalog_import".into()));
        } else {
            ctx.data_mut(|d| d.insert_temp("catalog_picked".into(), path));
        }
    }
}

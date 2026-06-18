//! Tab 3: Voice catalog — manage zero-shot reference voices.
//! Register WAV files as named voice profiles for quick reuse.

use std::sync::{Arc, Mutex};

use eframe::egui;

use crate::app::AppState;
use crate::app::VoiceEntry;
use crate::audio_playback::{self, AudioPlayer};
use crate::file_pick::FilePick;

/// Catalog tab state. Holds the currently-playing preview so it can be stopped.
#[derive(Default)]
pub struct CatalogState {
    pub player: Option<AudioPlayer>,
}

pub fn render(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    add_pick: &FilePick,
    import_pick: &FilePick,
    state: &Arc<Mutex<AppState>>,
    catalog: &mut CatalogState,
    icon_folder: &egui::TextureHandle,
    icon_play: &egui::TextureHandle,
    icon_trash: &egui::TextureHandle,
    empty_stars: &egui::TextureHandle,
    on_select: impl FnMut(usize),
) {
    let mut on_select = on_select;
    crate::theme::heading(ui, "Voice Catalog");
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new("Register reference audio for zero-shot voice conversion")
            .size(12.0)
            .color(crate::theme::colors::TEXT_DIM),
    );
    ui.add_space(10.0);

    // Add new voice — kawaii card
    crate::theme::info_card(ui, |ui| {
        ui.label(
            egui::RichText::new("Add Voice")
                .size(14.0)
                .strong()
                .color(crate::theme::colors::PINK_BRIGHT),
        );
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            let new_name = ctx.data_mut(|d| {
                d.get_temp_mut_or_insert_with::<String>("catalog_new_name".into(), || String::new())
                    .clone()
            });
            let mut name_buf = new_name;
            ui.label(
                egui::RichText::new("Name")
                    .size(12.0)
                    .color(crate::theme::colors::TEXT_DIM),
            );
            ui.add_sized([120.0, 20.0], egui::TextEdit::singleline(&mut name_buf));
            if crate::theme::icon_button(ui, icon_folder, "Browse", false) {
                add_pick.open();
            }
            // Receive the picked path from the background rfd thread.
            if let Some(path) = add_pick.take() {
                ctx.data_mut(|d| d.insert_temp("catalog_picked".into(), path));
            }
            if crate::theme::pill_button(ui, "Add", !name_buf.is_empty()) && !name_buf.is_empty() {
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
    });

    ui.add_space(10.0);

    // Voice list (scrollable)
    egui::ScrollArea::vertical()
        .max_height(400.0)
        .show(ui, |ui| {
            let s = state.lock().unwrap();
            if s.voices.is_empty() {
                // Empty state with illustration
                ui.add_space(20.0);
                ui.vertical_centered(|ui| {
                    ui.add(
                        egui::Image::from_texture(empty_stars)
                            .fit_to_exact_size(egui::vec2(120.0, 120.0))
                            .tint(egui::Color32::from_rgba_premultiplied(170, 140, 255, 180)),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("No voices registered yet.")
                            .size(13.0)
                            .color(crate::theme::colors::TEXT_MUTED),
                    );
                });
            } else {
                ui.label(
                    egui::RichText::new(format!("{} voices", s.voices.len()))
                        .size(14.0)
                        .strong()
                        .color(crate::theme::colors::CYAN),
                );
                ui.add_space(4.0);

                let mut to_delete = None;
                let selected = s.selected_voice;
                for (i, voice) in s.voices.iter().enumerate() {
                    let is_selected = selected == Some(i);
                    let card_bg = if is_selected {
                        egui::Color32::from_rgba_premultiplied(90, 60, 120, 60)
                    } else {
                        egui::Color32::from_rgba_premultiplied(40, 30, 60, 30)
                    };
                    egui::Frame::group(ui.style())
                        .fill(card_bg)
                        .stroke(egui::Stroke::new(
                            if is_selected { 2.0 } else { 1.0 },
                            if is_selected {
                                crate::theme::colors::PINK
                            } else {
                                crate::theme::colors::BG_PANEL_LIGHT
                            },
                        ))
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                // Index + name
                                let idx_str = format!("{}", i + 1);
                                ui.label(
                                    egui::RichText::new(if is_selected {
                                        "★"
                                    } else {
                                        idx_str.as_str()
                                    })
                                    .size(12.0)
                                    .color(if is_selected {
                                        crate::theme::colors::PINK_BRIGHT
                                    } else {
                                        crate::theme::colors::LAVENDER
                                    })
                                    .monospace(),
                                );
                                ui.label(
                                    egui::RichText::new(&voice.name)
                                        .size(14.0)
                                        .strong()
                                        .color(crate::theme::colors::TEXT),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if crate::theme::icon_button(
                                            ui, icon_trash, "Remove", false,
                                        ) {
                                            to_delete = Some(i);
                                        }
                                        // Play / Stop toggle ([F4]).
                                        let playing_idx = catalog
                                            .player
                                            .as_ref()
                                            .map(|p| p.is_playing())
                                            .unwrap_or(false);
                                        let play_label =
                                            if playing_idx { "■ Stop" } else { "▶ Play" };
                                        if crate::theme::icon_button(
                                            ui, icon_play, play_label, false,
                                        ) {
                                            if playing_idx {
                                                if let Some(p) = catalog.player.take() {
                                                    p.stop();
                                                }
                                            } else if let Ok((wav, sr)) =
                                                audio_playback::load_wav_mono(&voice.path)
                                            {
                                                let wav44 =
                                                    audio_playback::resample_linear(&wav, sr);
                                                catalog.player =
                                                    audio_playback::AudioPlayer::play(wav44).ok();
                                            }
                                        }
                                        // Select this voice as the Realtime reference.
                                        let select_label =
                                            if is_selected { "✓ Selected" } else { "Use" };
                                        if crate::theme::pill_button(ui, select_label, is_selected)
                                        {
                                            on_select(i);
                                        }
                                    },
                                );
                            });
                            ui.label(
                                egui::RichText::new(voice.path.to_string_lossy().as_ref())
                                    .size(10.0)
                                    .color(crate::theme::colors::TEXT_MUTED),
                            );
                        });
                    ui.add_space(4.0);
                }

                drop(s);
                if let Some(idx) = to_delete {
                    let mut s = state.lock().unwrap();
                    if s.selected_voice == Some(idx) {
                        s.selected_voice = None;
                    }
                    let name = s.voices[idx].name.clone();
                    s.voices.remove(idx);
                    s.status = format!("Removed voice: {name}");
                }
            }
        });

    ui.add_space(8.0);

    // Import / Export
    ui.collapsing(
        egui::RichText::new("Import / Export")
            .size(13.0)
            .color(crate::theme::colors::CYAN),
        |ui| {
            ui.horizontal(|ui| {
                if crate::theme::icon_button(ui, icon_folder, "Export", true) {
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
                if crate::theme::icon_button(ui, icon_folder, "Import", true) {
                    import_pick.open();
                }
            });
        },
    );

    // Handle import pick.
    if let Some(path) = import_pick.take() {
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
    }
}

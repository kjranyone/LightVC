//! Custom egui widgets — thin wrappers that delegate to the theme module.
//!
//! For the full kawaii-styled widgets, use `crate::theme::*` directly.

use eframe::egui;

/// Draw a kawaii level meter.
pub fn level_meter(ui: &mut egui::Ui, rms: f32, label: &str) {
    crate::theme::level_meter(ui, rms, label);
}

/// Compute RMS of a sample buffer.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

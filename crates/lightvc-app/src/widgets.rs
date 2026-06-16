//! Custom egui widgets: level meters, sliders.

use eframe::egui;

/// Draw a horizontal level meter.
pub fn level_meter(ui: &mut egui::Ui, rms: f32, label: &str) {
    ui.horizontal(|ui| {
        ui.label(format!("{label}:"));
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width() - 80.0, 14.0),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);

        // Background
        painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(30, 30, 30));

        // Level (RMS → dB-ish scale)
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
        painter.rect_filled(bar_rect, 1.0, color);

        // dB text
        let db = if rms > 0.0 { 20.0 * rms.log10() } else { -99.0 };
        ui.label(format!("{db:+.0}dB"));
    });
}

/// Status indicator dot.
pub fn status_dot(ui: &mut egui::Ui, active: bool) {
    let color = if active {
        egui::Color32::from_rgb(80, 200, 80)
    } else {
        egui::Color32::from_rgb(100, 100, 100)
    };
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, color);
}

/// Compute RMS of a sample buffer.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

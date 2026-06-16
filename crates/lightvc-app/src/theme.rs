//! Kawaii Future Bass theme for LightVC.
//!
//! Visual language:
//! - Pastel colors (pink, lavender, cyan, mint)
//! - Rounded corners everywhere
//! - Soft glow on active elements
//! - Pop-cute typography

use eframe::egui;
use egui::Color32;

// ---------------------------------------------------------------------------
// Color palette
// ---------------------------------------------------------------------------

pub mod colors {
    use eframe::egui::Color32;

    pub const BG_DARK: Color32 = Color32::from_rgb(28, 22, 38);
    pub const BG_PANEL: Color32 = Color32::from_rgb(42, 32, 56);
    pub const BG_PANEL_LIGHT: Color32 = Color32::from_rgb(52, 40, 68);

    pub const PINK: Color32 = Color32::from_rgb(255, 130, 190);
    pub const PINK_BRIGHT: Color32 = Color32::from_rgb(255, 160, 210);
    pub const LAVENDER: Color32 = Color32::from_rgb(170, 140, 255);
    pub const CYAN: Color32 = Color32::from_rgb(120, 230, 255);
    pub const MINT: Color32 = Color32::from_rgb(130, 255, 200);
    pub const YELLOW: Color32 = Color32::from_rgb(255, 220, 130);

    pub const TEXT: Color32 = Color32::from_rgb(240, 235, 250);
    pub const TEXT_DIM: Color32 = Color32::from_rgb(160, 150, 180);
    pub const TEXT_MUTED: Color32 = Color32::from_rgb(110, 100, 130);

    pub const ACCENT: Color32 = PINK;
    pub const ACCENT2: Color32 = CYAN;
}

// ---------------------------------------------------------------------------
// Style setup
// ---------------------------------------------------------------------------

/// Apply the Kawaii Future Bass theme to an egui context.
pub fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    // Spacing
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(16.0, 8.0);

    // Dark base
    style.visuals.dark_mode = true;
    style.visuals.panel_fill = colors::BG_DARK;
    style.visuals.extreme_bg_color = colors::BG_DARK;
    style.visuals.faint_bg_color = colors::BG_PANEL;

    // Widget colors — pastel
    use colors::*;

    style.visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_DIM);
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(0.5, BG_PANEL_LIGHT);

    style.visuals.widgets.inactive.bg_fill = BG_PANEL_LIGHT;
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT);
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, LAVENDER);

    style.visuals.widgets.hovered.bg_fill = PINK;
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, TEXT);
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(2.0, PINK_BRIGHT);

    style.visuals.widgets.active.bg_fill = LAVENDER;
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(1.5, TEXT);
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(2.0, CYAN);

    style.visuals.widgets.open.bg_fill = LAVENDER;
    style.visuals.widgets.open.fg_stroke = egui::Stroke::new(1.5, TEXT);

    // Selection
    style.visuals.selection.bg_fill = PINK;
    style.visuals.selection.stroke = egui::Stroke::new(1.0, PINK_BRIGHT);

    // Hyperlinks
    style.visuals.hyperlink_color = CYAN;

    ctx.set_style(style);
}

// ---------------------------------------------------------------------------
// Background
// ---------------------------------------------------------------------------

/// Draw a gradient-ish background (3 bands).
pub fn gradient_background(ctx: &egui::Context) {
    let rect = ctx.screen_rect();
    let painter = ctx.layer_painter(egui::LayerId::background());

    let h = rect.height();
    let band = h / 3.0;

    painter.rect_filled(
        egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), band)),
        0.0,
        Color32::from_rgb(30, 24, 42),
    );
    painter.rect_filled(
        egui::Rect::from_min_size(
            egui::pos2(rect.min.x, rect.min.y + band),
            egui::vec2(rect.width(), band),
        ),
        0.0,
        Color32::from_rgb(36, 26, 50),
    );
    painter.rect_filled(
        egui::Rect::from_min_size(
            egui::pos2(rect.min.x, rect.min.y + band * 2.0),
            egui::vec2(rect.width(), band),
        ),
        0.0,
        Color32::from_rgb(26, 20, 40),
    );
}

/// A pill-shaped button with icon image + text.
pub fn icon_button(
    ui: &mut egui::Ui,
    icon: &egui::TextureHandle,
    text: &str,
    active: bool,
) -> bool {
    let (fill, stroke_color) = if active {
        (colors::LAVENDER, colors::CYAN)
    } else {
        (colors::BG_PANEL_LIGHT, colors::PINK)
    };

    let btn = egui::Button::image_and_text(
        egui::Image::from_texture(icon).fit_to_exact_size(egui::vec2(16.0, 16.0)),
        egui::RichText::new(text)
            .size(13.0)
            .strong()
            .color(colors::TEXT),
    )
    .fill(fill)
    .stroke(egui::Stroke::new(2.0, stroke_color))
    .min_size(egui::vec2(80.0, 30.0));

    ui.add(btn).clicked()
}

// ---------------------------------------------------------------------------
// Custom widgets
// ---------------------------------------------------------------------------

/// A kawaii-styled section heading.
pub fn heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(text)
            .size(20.0)
            .strong()
            .color(colors::PINK_BRIGHT),
    );
    ui.add_space(2.0);
}

/// A pill-shaped button with glow.
pub fn pill_button(ui: &mut egui::Ui, text: &str, active: bool) -> bool {
    let (fill, stroke_color) = if active {
        (colors::LAVENDER, colors::CYAN)
    } else {
        (colors::BG_PANEL_LIGHT, colors::PINK)
    };

    let btn = egui::Button::new(
        egui::RichText::new(text)
            .size(14.0)
            .strong()
            .color(colors::TEXT),
    )
    .fill(fill)
    .stroke(egui::Stroke::new(2.0, stroke_color))
    .min_size(egui::vec2(80.0, 32.0));

    ui.add(btn).clicked()
}

/// A tab button with kawaii styling (responsive width).
pub fn tab_button(ui: &mut egui::Ui, text: &str, selected: bool) -> bool {
    let (bg, fg, stroke) = if selected {
        (colors::PINK, colors::TEXT, colors::PINK_BRIGHT)
    } else {
        (colors::BG_PANEL, colors::TEXT_DIM, colors::BG_PANEL_LIGHT)
    };

    let btn = egui::Button::new(egui::RichText::new(text).size(14.0).strong().color(fg))
        .fill(bg)
        .stroke(egui::Stroke::new(if selected { 2.0 } else { 1.0 }, stroke))
        .min_size(egui::vec2(70.0, 30.0));

    ui.add(btn).clicked()
}

/// A status indicator dot with glow.
pub fn status_dot(ui: &mut egui::Ui, active: bool, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
    let painter = ui.painter();

    if active {
        painter.circle_filled(
            rect.center(),
            9.0,
            Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), 50),
        );
    }
    painter.circle_filled(
        rect.center(),
        5.0,
        if active { color } else { colors::TEXT_MUTED },
    );
}

/// A kawaii level meter with gradient fill.
pub fn level_meter(ui: &mut egui::Ui, level: f32, label: &str) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .size(12.0)
                .color(colors::TEXT_DIM),
        );

        let meter_width = (ui.available_width() - 70.0).max(40.0);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(meter_width, 16.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 8.0, colors::BG_DARK);

        let bar_level = (level * 10.0).min(1.0).max(0.0);
        let bar_width = rect.width() * bar_level;
        if bar_width > 1.0 {
            let color = if bar_level > 0.85 {
                colors::PINK
            } else if bar_level > 0.65 {
                colors::YELLOW
            } else if bar_level > 0.4 {
                colors::CYAN
            } else {
                colors::MINT
            };

            painter.rect_filled(
                egui::Rect::from_min_size(rect.min, egui::vec2(bar_width, rect.height())),
                8.0,
                color,
            );

            // Glow on right edge
            if bar_level > 0.05 {
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(rect.min.x + bar_width - 6.0, rect.min.y),
                        egui::vec2(12.0, rect.height()),
                    ),
                    8.0,
                    Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), 60),
                );
            }
        }

        let db = if level > 0.0 {
            20.0 * level.log10()
        } else {
            -99.0
        };
        ui.label(
            egui::RichText::new(format!("{db:+.0}"))
                .size(11.0)
                .color(colors::TEXT_MUTED)
                .monospace(),
        );
    });
}

/// A kawaii info card (rounded panel with border, generous padding).
pub fn info_card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::NONE
        .fill(colors::BG_PANEL)
        .stroke(egui::Stroke::new(1.5, colors::LAVENDER))
        .inner_margin(16)
        .show(ui, add_contents);
}

// ---------------------------------------------------------------------------
// Knob widget (sprite-sheet based)
// ---------------------------------------------------------------------------

/// Knob sprite sheet: 12 frames, each 64×64px, stacked vertically (768px total).
/// Frame 0 = minimum (7 o'clock), Frame 11 = maximum (5 o'clock).
const KNOB_FRAMES: usize = 12;
const KNOB_SIZE: f32 = 64.0;

/// Draw an interactive knob from the sprite sheet.
///
/// - `value`: normalized 0.0..=1.0
/// - `label`: text below the knob
/// - Returns: new value if the knob was dragged, or None
pub fn knob(
    ui: &mut egui::Ui,
    knob_tex: &egui::TextureHandle,
    id: egui::Id,
    value: f32,
    label: &str,
) -> Option<f32> {
    let size = egui::vec2(KNOB_SIZE, KNOB_SIZE);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(KNOB_SIZE, KNOB_SIZE + 20.0), egui::Sense::drag());

    let knob_rect = egui::Rect::from_min_size(rect.min, size);

    // Handle drag interaction
    let mut new_value = None;
    if response.dragged() {
        let drag_delta = response.drag_delta().y;
        let sensitivity = 0.005; // 200px for full range
        let delta = -drag_delta * sensitivity;
        new_value = Some((value + delta).clamp(0.0, 1.0));
    }

    // Double-click resets to center
    if response.double_clicked() {
        new_value = Some(0.5);
    }

    // Select frame from value
    let display_value = new_value.unwrap_or(value);
    let frame_idx =
        ((display_value * (KNOB_FRAMES - 1) as f32).round() as usize).min(KNOB_FRAMES - 1);

    // Draw the knob frame
    let frame_h = 1.0 / KNOB_FRAMES as f32;
    let frame_top = frame_idx as f32 / KNOB_FRAMES as f32;
    let uv = egui::Rect::from_min_max(
        egui::pos2(0.0, frame_top),
        egui::pos2(1.0, frame_top + frame_h),
    );

    let painter = ui.painter_at(knob_rect);
    painter.image(
        knob_tex.id(),
        knob_rect,
        uv,
        if response.dragged() {
            egui::Color32::from_rgba_premultiplied(255, 200, 240, 255)
        } else if response.hovered() {
            egui::Color32::from_rgba_premultiplied(255, 180, 220, 255)
        } else {
            egui::Color32::WHITE
        },
    );

    // Glow ring when active
    if response.dragged() {
        painter.circle_stroke(
            knob_rect.center(),
            KNOB_SIZE * 0.48,
            egui::Stroke::new(2.0, colors::PINK),
        );
    }

    // Label below
    let label_pos = egui::pos2(knob_rect.center().x, knob_rect.max.y + 6.0);
    painter.text(
        label_pos,
        egui::Align2::CENTER_TOP,
        label,
        egui::FontId::proportional(11.0),
        colors::TEXT_DIM,
    );

    new_value
}

/// A labeled knob row: [Knob] [Label + value text]
pub fn knob_labeled(
    ui: &mut egui::Ui,
    knob_tex: &egui::TextureHandle,
    id_str: &str,
    value: f32,
    label: &str,
    value_text: &str,
) -> Option<f32> {
    ui.horizontal(|ui| {
        let id = ui.make_persistent_id(id_str);
        let result = knob(ui, knob_tex, id, value, label);
        ui.vertical(|ui| {
            ui.label(
                egui::RichText::new(label)
                    .size(12.0)
                    .color(colors::TEXT_DIM),
            );
            ui.label(
                egui::RichText::new(value_text)
                    .size(14.0)
                    .strong()
                    .color(colors::PINK_BRIGHT),
            );
        });
        result
    })
    .inner
}

//! Kawaii Future Bass theme for LightVC.
//!
//! Visual language:
//! - Neon glassmorphism (translucent cards with soft glow)
//! - Pastel gradients (pink→cyan, lavender→mint)
//! - Candy-like rounded buttons with subtle gradients
//! - Music-plugin-style meters with tick marks and peak markers
//! - Y2K pastel-cyber accents (stars, dots, soft scanlines)
//!
//! Some helpers are reserved for the next iteration of the UI refresh and
//! are kept here so call sites can adopt them incrementally.

#![allow(dead_code)]

use eframe::egui;
use egui::Color32;

// ===========================================================================
// Design tokens
// ===========================================================================

pub mod space {
    pub const TIGHT: f32 = 2.0;
    pub const SMALL: f32 = 4.0;
    pub const MEDIUM: f32 = 8.0;
    pub const LARGE: f32 = 12.0;
    pub const XLARGE: f32 = 20.0;
}

pub const LABEL_WIDTH: f32 = 80.0;
pub const FIELD_HEIGHT: f32 = 22.0;
pub const BUTTON_HEIGHT: f32 = 32.0;
pub const CTA_HEIGHT: f32 = 44.0;
/// Card corner rounding — generous, candy-like.
pub const CARD_ROUNDING: f32 = 14.0;
/// Button corner rounding — pill-ish but not fully round.
pub const BUTTON_ROUNDING: f32 = 10.0;
pub const METER_HEIGHT: f32 = 18.0;

// ===========================================================================
// Color palette — Kawaii Future Bass / Neon Glassmorphism
// ===========================================================================

pub mod colors {
    use eframe::egui::Color32;

    // Backgrounds — deep club-night purples
    pub const BG_DEEP: Color32 = Color32::from_rgb(0x15, 0x11, 0x21);
    pub const BG_DARK: Color32 = Color32::from_rgb(0x1D, 0x17, 0x30);
    pub const BG_PANEL: Color32 = Color32::from_rgb(0x24, 0x1A, 0x3A);

    // Semi-transparent card tints (used with alpha in glassmorphism)
    pub const CARD_GLASS: Color32 = Color32::from_rgb(0x2A, 0x20, 0x44);
    pub const CARD_GLASS_LIGHT: Color32 = Color32::from_rgb(0x34, 0x28, 0x52);

    // Pastel neons
    pub const PINK: Color32 = Color32::from_rgb(0xFF, 0x8A, 0xC8);
    pub const PINK_BRIGHT: Color32 = Color32::from_rgb(0xFF, 0xB8, 0xDC);
    pub const PINK_DEEP: Color32 = Color32::from_rgb(0xE0, 0x5C, 0xA8);
    pub const LAVENDER: Color32 = Color32::from_rgb(0xB7, 0x9C, 0xFF);
    pub const CYAN: Color32 = Color32::from_rgb(0x6F, 0xEA, 0xFF);
    pub const MINT: Color32 = Color32::from_rgb(0x8F, 0xFF, 0xE0);
    pub const LEMON: Color32 = Color32::from_rgb(0xFF, 0xE3, 0x7A);
    pub const CORAL: Color32 = Color32::from_rgb(0xFF, 0x9E, 0x7A);

    // Text
    pub const TEXT: Color32 = Color32::from_rgb(0xF7, 0xEF, 0xFF);
    pub const TEXT_DIM: Color32 = Color32::from_rgb(0xB8, 0xAF, 0xCB);
    pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x7A, 0x6F, 0x92);

    // Borders — translucent lavender
    pub const BORDER: Color32 = Color32::from_rgb(0x6A, 0x52, 0xA8);

    // Functional
    pub const ERROR: Color32 = Color32::from_rgb(0xFF, 0x6A, 0x7A);
    pub const WARN: Color32 = LEMON;

    // Compatibility aliases (legacy names used in some call sites)
    pub const BG_PANEL_LIGHT: Color32 = CARD_GLASS_LIGHT;
    pub const YELLOW: Color32 = LEMON;
}

// ===========================================================================
// Gradient helpers
// ===========================================================================

/// A two-stop linear gradient as (top_color, bottom_color).
pub struct Gradient(pub Color32, pub Color32);

impl Gradient {
    #[allow(dead_code)]
    pub const PINK_CYAN: Gradient = Gradient(colors::PINK, colors::CYAN);
    #[allow(dead_code)]
    pub const LAVENDER_MINT: Gradient = Gradient(colors::LAVENDER, colors::MINT);
    #[allow(dead_code)]
    pub const CYAN_PINK: Gradient = Gradient(colors::CYAN, colors::PINK);
    #[allow(dead_code)]
    pub const LEMON_CORAL: Gradient = Gradient(colors::LEMON, colors::CORAL);

    /// Vertical gradient fill on a rect with given rounding.
    pub fn fill_vertical(&self, painter: &egui::Painter, rect: egui::Rect, rounding: f32) {
        let steps = 24;
        let step_h = rect.height() / steps as f32;
        for i in 0..steps {
            let t = i as f32 / (steps - 1) as f32;
            let c = lerp_color(self.0, self.1, t);
            let y = rect.min.y + step_h * i as f32;
            let r = egui::Rect::from_min_size(
                egui::pos2(rect.min.x, y),
                egui::vec2(rect.width(), step_h + 1.0),
            );
            painter.rect_filled(r, rounding, c);
        }
    }
}

pub fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgb(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
    )
}

pub fn with_alpha(c: Color32, a: u8) -> Color32 {
    Color32::from_rgba_premultiplied(
        (c.r() as u16 * a as u16 / 255) as u8,
        (c.g() as u16 * a as u16 / 255) as u8,
        (c.b() as u16 * a as u16 / 255) as u8,
        a,
    )
}

// ===========================================================================
// Global theme setup
// ===========================================================================

pub fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.global_style()).clone();

    // Tighter rhythm — music-plugin density, not web-form looseness.
    style.spacing.item_spacing = egui::vec2(6.0, 6.0);
    style.spacing.button_padding = egui::vec2(14.0, 6.0);
    style.spacing.interact_size = egui::vec2(0.0, BUTTON_HEIGHT);

    use colors::*;

    style.visuals.dark_mode = true;
    style.visuals.panel_fill = BG_DEEP;
    style.visuals.extreme_bg_color = BG_DARK;
    style.visuals.faint_bg_color = CARD_GLASS;
    style.visuals.window_fill = CARD_GLASS;

    // Noninteractive: subtle glass panels
    style.visuals.widgets.noninteractive.bg_fill = with_alpha(CARD_GLASS, 80);
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_DIM);
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, with_alpha(BORDER, 90));
    style.visuals.widgets.noninteractive.corner_radius = CARD_ROUNDING.into();
    style.visuals.widgets.noninteractive.expansion = 0.0;

    // Inactive: translucent lavender tint
    style.visuals.widgets.inactive.bg_fill = with_alpha(CARD_GLASS_LIGHT, 120);
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT);
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, with_alpha(LAVENDER, 120));
    style.visuals.widgets.inactive.corner_radius = BUTTON_ROUNDING.into();
    style.visuals.widgets.inactive.expansion = 0.0;

    // Hovered: warm pink glow
    style.visuals.widgets.hovered.bg_fill = with_alpha(PINK, 140);
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT);
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.5, PINK_BRIGHT);
    style.visuals.widgets.hovered.corner_radius = BUTTON_ROUNDING.into();
    style.visuals.widgets.hovered.expansion = 0.0;

    // Active: cool lavender-cyan
    style.visuals.widgets.active.bg_fill = with_alpha(LAVENDER, 160);
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, TEXT);
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.5, CYAN);
    style.visuals.widgets.active.corner_radius = BUTTON_ROUNDING.into();
    style.visuals.widgets.active.expansion = 0.0;

    // Open (expanded combos etc.)
    style.visuals.widgets.open.bg_fill = with_alpha(LAVENDER, 140);
    style.visuals.widgets.open.fg_stroke = egui::Stroke::new(1.0, TEXT);
    style.visuals.widgets.open.corner_radius = BUTTON_ROUNDING.into();
    style.visuals.widgets.open.expansion = 0.0;

    // Selection — pink candy
    style.visuals.selection.bg_fill = with_alpha(PINK, 180);
    style.visuals.selection.stroke = egui::Stroke::new(1.0, PINK_BRIGHT);

    // Hyperlinks
    style.visuals.hyperlink_color = CYAN;

    ctx.set_global_style(style);
}

// ===========================================================================
// Typography helpers
// ===========================================================================

/// Screen / page title — large kawaii display.
pub fn page_title(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(26.0)
            .strong()
            .color(colors::PINK_BRIGHT),
    );
}

/// Section heading inside a card.
pub fn heading(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(18.0)
            .strong()
            .color(colors::PINK_BRIGHT),
    );
    ui.add_space(space::TIGHT);
}

/// Subheading — small caps style, lavender.
pub fn subheading(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(13.0)
            .strong()
            .color(colors::LAVENDER),
    );
    ui.add_space(space::SMALL);
}

/// Monospace numeric label (latency, RTF, dB).
pub fn numeric(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(13.0)
            .color(colors::CYAN)
            .monospace(),
    );
}

// ===========================================================================
// Form helpers
// ===========================================================================

pub fn form_label(ui: &mut egui::Ui, text: &str) {
    ui.add_sized(
        [LABEL_WIDTH, FIELD_HEIGHT],
        egui::Label::new(egui::RichText::new(text).size(12.0).color(colors::TEXT_DIM)),
    );
}

pub fn path_text_edit(ui: &mut egui::Ui, buf: &mut String, hint: &str) {
    let w = (ui.available_width() * 0.7).max(120.0);
    ui.add_sized(
        [w, FIELD_HEIGHT],
        egui::TextEdit::singleline(buf).hint_text(hint),
    );
}

// ===========================================================================
// Buttons — candy style with subtle gradient + glow
// ===========================================================================

fn candy_button(
    text: &str,
    fill: Color32,
    stroke: Color32,
    min_size: egui::Vec2,
) -> egui::Button<'static> {
    egui::Button::new(
        egui::RichText::new(text)
            .size(13.0)
            .strong()
            .color(colors::TEXT),
    )
    .fill(fill)
    .stroke(egui::Stroke::new(1.0, stroke))
    .corner_radius(BUTTON_ROUNDING)
    .min_size(min_size)
}

/// Standard pill button — pink candy when active, glass when idle.
pub fn pill_button(ui: &mut egui::Ui, text: &str, active: bool) -> bool {
    let (fill, stroke) = if active {
        (with_alpha(colors::LAVENDER, 160), colors::CYAN)
    } else {
        (
            with_alpha(colors::CARD_GLASS_LIGHT, 140),
            with_alpha(colors::PINK, 180),
        )
    };
    ui.add(candy_button(
        text,
        fill,
        stroke,
        egui::vec2(80.0, BUTTON_HEIGHT),
    ))
    .clicked()
}

/// Icon + text button — same candy sizing as pill_button.
pub fn icon_button(
    ui: &mut egui::Ui,
    icon: &egui::TextureHandle,
    text: &str,
    active: bool,
) -> bool {
    let (fill, stroke) = if active {
        (with_alpha(colors::LAVENDER, 160), colors::CYAN)
    } else {
        (
            with_alpha(colors::CARD_GLASS_LIGHT, 140),
            with_alpha(colors::PINK, 180),
        )
    };
    let btn = egui::Button::image_and_text(
        egui::Image::from_texture(icon).fit_to_exact_size(egui::vec2(14.0, 14.0)),
        egui::RichText::new(text)
            .size(12.0)
            .strong()
            .color(colors::TEXT),
    )
    .fill(fill)
    .stroke(egui::Stroke::new(1.0, stroke))
    .corner_radius(BUTTON_ROUNDING)
    .min_size(egui::vec2(70.0, BUTTON_HEIGHT));
    ui.add(btn).clicked()
}

/// Primary CTA — bigger, pink gradient vibe.
pub fn primary_button(ui: &mut egui::Ui, text: &str, enabled: bool) -> bool {
    let btn = candy_button(
        text,
        if enabled {
            with_alpha(colors::PINK, 180)
        } else {
            with_alpha(colors::CARD_GLASS, 100)
        },
        if enabled {
            colors::PINK_BRIGHT
        } else {
            colors::TEXT_MUTED
        },
        egui::vec2(160.0, CTA_HEIGHT),
    )
    .corner_radius(CTA_HEIGHT * 0.5);
    ui.add_enabled(enabled, btn).clicked()
}

/// Tab button — pill, selected has gradient background hint.
pub fn tab_button(ui: &mut egui::Ui, text: &str, selected: bool) -> bool {
    let (fill, fg, stroke) = if selected {
        (
            with_alpha(colors::PINK, 150),
            colors::TEXT,
            colors::PINK_BRIGHT,
        )
    } else {
        (
            with_alpha(colors::CARD_GLASS, 100),
            colors::TEXT_DIM,
            with_alpha(colors::BORDER, 80),
        )
    };
    let btn = egui::Button::new(egui::RichText::new(text).size(13.0).strong().color(fg))
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, stroke))
        .corner_radius(BUTTON_HEIGHT * 0.5)
        .min_size(egui::vec2(72.0, BUTTON_HEIGHT));
    ui.add(btn).clicked()
}

// ===========================================================================
// Status — glowing badge instead of bare dot
// ===========================================================================

/// Status badge with glow. Use for LIVE / BYPASS / STOPPED indicators.
pub fn status_dot(ui: &mut egui::Ui, active: bool, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
    let painter = ui.painter();
    let center = rect.center();

    if active {
        // Outer halo
        painter.circle_filled(center, 10.0, with_alpha(color, 40));
        painter.circle_filled(center, 7.0, with_alpha(color, 90));
    }
    painter.circle_filled(center, 4.5, if active { color } else { colors::TEXT_MUTED });
    if active {
        // Inner highlight — glossy candy
        painter.circle_filled(
            egui::pos2(center.x - 1.2, center.y - 1.2),
            1.5,
            Color32::from_rgba_premultiplied(255, 255, 255, 160),
        );
    }
}

/// A pill-shaped status badge with label and glow.
pub fn status_badge(ui: &mut egui::Ui, text: &str, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(96.0, 22.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 11.0, with_alpha(color, 50));
    painter.rect_stroke(
        rect,
        11.0,
        egui::Stroke::new(1.0, color),
        egui::StrokeKind::Outside,
    );
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(11.0),
        color,
    );
}

// ===========================================================================
// Level meter — music-plugin style with gradient + ticks + peak
// ===========================================================================

/// A neon level meter with gradient fill, tick marks, and peak marker.
pub fn level_meter(ui: &mut egui::Ui, level: f32, label: &str) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .size(11.0)
                .color(colors::TEXT_DIM),
        );

        let meter_width = (ui.available_width() - 56.0).max(40.0);
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(meter_width, METER_HEIGHT), egui::Sense::hover());
        let painter = ui.painter_at(rect);

        // Track background — translucent dark with subtle inner shadow
        painter.rect_filled(rect, 9.0, with_alpha(colors::BG_DEEP, 200));
        painter.rect_stroke(
            rect,
            9.0,
            egui::Stroke::new(1.0, with_alpha(colors::BORDER, 90)),
            egui::StrokeKind::Outside,
        );

        // Tick marks — subtle vertical grid every 10%
        for i in 1..10 {
            let x = rect.min.x + rect.width() * (i as f32 / 10.0);
            painter.line_segment(
                [
                    egui::pos2(x, rect.min.y + 2.0),
                    egui::pos2(x, rect.max.y - 2.0),
                ],
                egui::Stroke::new(1.0, with_alpha(colors::TEXT_MUTED, 40)),
            );
        }

        // Bar — gradient based on level. Use ×5 so RMS 0.05-0.3 maps to 0.25-1.5.
        let bar_level = (level * 5.0).clamp(0.0, 1.0);
        let bar_width = rect.width() * bar_level;
        if bar_width > 1.0 {
            // Choose gradient: input→pink/coral, output→cyan/mint, clip→pink/lemon
            let grad = if bar_level > 0.85 {
                Gradient(colors::PINK, colors::LEMON)
            } else if bar_level > 0.65 {
                Gradient(colors::LEMON, colors::CORAL)
            } else if bar_level > 0.4 {
                Gradient(colors::CYAN, colors::MINT)
            } else {
                Gradient(colors::MINT, colors::CYAN)
            };
            let bar_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(bar_width, rect.height()));
            // Clip the gradient to the bar with rounded left edge
            painter.rect_filled(bar_rect, 9.0, grad.0);
            grad.fill_vertical(&painter, bar_rect, 9.0);

            // Peak marker — small glowing dot at the bar tip
            let peak_x = rect.min.x + bar_width - 3.0;
            let peak_pos = egui::pos2(peak_x, rect.center().y);
            painter.circle_filled(peak_pos, 3.0, colors::TEXT);
            painter.circle_filled(peak_pos, 5.0, with_alpha(grad.1, 100));
        }

        let db = if level > 0.0 {
            20.0 * level.log10()
        } else {
            -99.0
        };
        ui.label(
            egui::RichText::new(format!("{db:+.0}"))
                .size(11.0)
                .color(colors::TEXT_DIM)
                .monospace(),
        );
    });
}

// ===========================================================================
// Info card — glassmorphism with translucent fill + soft border + glow
// ===========================================================================

/// A glass card: translucent lavender-tinted panel with soft glow border.
pub fn info_card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::NONE
        .fill(with_alpha(colors::CARD_GLASS, 130))
        .stroke(egui::Stroke::new(1.0, with_alpha(colors::LAVENDER, 120)))
        .corner_radius(CARD_ROUNDING)
        .inner_margin(14)
        .show(ui, add_contents);
}

/// A highlighted card variant — used for the active/primary card.
pub fn glow_card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::NONE
        .fill(with_alpha(colors::PINK, 40))
        .stroke(egui::Stroke::new(1.5, colors::PINK_BRIGHT))
        .corner_radius(CARD_ROUNDING)
        .inner_margin(14)
        .show(ui, add_contents);
}

// ===========================================================================
// Knob widget (unchanged — sprite-sheet based)
// ===========================================================================

const KNOB_FRAMES: usize = 12;
const KNOB_SIZE: f32 = 64.0;

pub fn knob(
    ui: &mut egui::Ui,
    knob_tex: &egui::TextureHandle,
    _id: egui::Id,
    value: f32,
    label: &str,
) -> Option<f32> {
    let size = egui::vec2(KNOB_SIZE, KNOB_SIZE);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(KNOB_SIZE, KNOB_SIZE + 20.0), egui::Sense::drag());
    let knob_rect = egui::Rect::from_min_size(rect.min, size);

    let mut new_value = None;
    if response.dragged() {
        let drag_delta = response.drag_delta().y;
        let delta = -drag_delta * 0.005;
        new_value = Some((value + delta).clamp(0.0, 1.0));
    }
    if response.double_clicked() {
        new_value = Some(0.5);
    }

    let display_value = new_value.unwrap_or(value);
    let frame_idx =
        ((display_value * (KNOB_FRAMES - 1) as f32).round() as usize).min(KNOB_FRAMES - 1);

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

    if response.dragged() {
        painter.circle_stroke(
            knob_rect.center(),
            KNOB_SIZE * 0.48,
            egui::Stroke::new(2.0, colors::PINK),
        );
    }

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

#[allow(dead_code)]
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

//! Asset loading — embeds all images at compile time via include_bytes!.

use eframe::egui;

pub const ICON_256_PNG: &[u8] = include_bytes!("../assets/icons/icon_256.png");
pub const LOGO_PNG: &[u8] = include_bytes!("../assets/logo/logo_header.png");
pub const LOGO_2X_PNG: &[u8] = include_bytes!("../assets/logo/logo_header@2x.png");
pub const BG_TEXTURE_PNG: &[u8] = include_bytes!("../assets/textures/bg_texture.png");
pub const KNOB_FRAMES_PNG: &[u8] = include_bytes!("../assets/knobs/knob_64_frames.png");
pub const SPLASH_PNG: &[u8] = include_bytes!("../assets/splash/splash.png");

/// Load RGBA image data from PNG bytes, returning (width, height, rgba_pixels).
pub fn load_image_from_png(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    use std::io::Cursor;
    let img = image::load_from_memory(data).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Some((w as usize, h as usize, rgba.into_raw()))
}

/// Load the app icon as an egui::IconData.
pub fn load_icon() -> Option<egui::IconData> {
    let (w, h, rgba) = load_image_from_png(ICON_256_PNG)?;
    Some(egui::IconData {
        rgba,
        width: w as u32,
        height: h as u32,
    })
}

/// Cache for texture handles (created once per egui context).
pub struct AssetCache {
    pub bg_texture: Option<egui::TextureHandle>,
    pub logo_texture: Option<egui::TextureHandle>,
    pub knob_texture: Option<egui::TextureHandle>,
    pub splash_texture: Option<egui::TextureHandle>,
}

impl Default for AssetCache {
    fn default() -> Self {
        Self {
            bg_texture: None,
            logo_texture: None,
            knob_texture: None,
            splash_texture: None,
        }
    }
}

impl AssetCache {
    fn make_texture(ctx: &egui::Context, id: &str, data: &[u8]) -> egui::TextureHandle {
        let (w, h, rgba) = load_image_from_png(data).unwrap_or_else(|| panic!("{id} is valid"));
        let pixels: Vec<egui::Color32> = rgba
            .chunks_exact(4)
            .map(|c| egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
            .collect();
        ctx.load_texture(
            id,
            egui::ColorImage {
                size: [w, h],
                pixels,
                source_size: egui::Vec2::new(w as f32, h as f32),
            },
            egui::TextureOptions::LINEAR,
        )
    }

    pub fn bg(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.bg_texture
            .get_or_insert_with(|| Self::make_texture(ctx, "bg_texture", BG_TEXTURE_PNG))
    }

    pub fn logo(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.logo_texture
            .get_or_insert_with(|| Self::make_texture(ctx, "logo", LOGO_PNG))
    }

    pub fn knob(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.knob_texture
            .get_or_insert_with(|| Self::make_texture(ctx, "knob_frames", KNOB_FRAMES_PNG))
    }

    pub fn splash(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        self.splash_texture
            .get_or_insert_with(|| Self::make_texture(ctx, "splash", SPLASH_PNG))
    }
}

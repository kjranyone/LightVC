//! Cross-platform file picker built on rfd (native dialogs).
//!
//! Replaces egui-file-dialog, which pulled in egui 0.31 alongside the app's
//! egui 0.34, causing a version split and broken dialogs (update(ctx) was
//! never called). rfd uses the OS native picker and runs off the UI thread,
//! so the egui loop is never blocked.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// A handle to an in-flight or completed file selection.
///
/// Clone-cheap (Arc inside). Poll [`take`](Self::take) each frame; it returns
/// `Some(path)` once the user has chosen and the background thread finishes.
#[derive(Clone, Default)]
pub struct FilePick {
    result: Arc<Mutex<Option<PathBuf>>>,
}

impl FilePick {
    /// Start a file-open dialog in a background thread.
    pub fn open(&self) {
        let r = self.result.clone();
        std::thread::spawn(move || {
            let p = rfd::FileDialog::new().pick_file();
            *r.lock().unwrap() = p;
        });
    }

    /// Start a file-save dialog in a background thread.
    #[allow(dead_code)]
    pub fn save(&self) {
        let r = self.result.clone();
        std::thread::spawn(move || {
            let p = rfd::FileDialog::new().save_file();
            *r.lock().unwrap() = p;
        });
    }

    /// Returns the chosen path if ready, clearing the slot.
    pub fn take(&self) -> Option<PathBuf> {
        self.result.lock().unwrap().take()
    }
}

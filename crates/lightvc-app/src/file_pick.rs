//! Cross-platform file picker built on rfd (native dialogs, off the UI thread).
//! UI は每フレーム(またはタイマー刻みで)[`take`](Self::take) を呼んで結果を受け取る。

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A handle to an in-flight or completed file selection.
/// Clone-cheap (Arc inside). `pending()` tells the UI tick to keep repainting.
#[derive(Clone, Default)]
pub struct FilePick {
    result: Arc<Mutex<Option<PathBuf>>>,
    pending: Arc<AtomicBool>,
}

impl FilePick {
    /// Start a file-open dialog in a background thread.
    pub fn open(&self) {
        if self.pending.load(Ordering::Relaxed) {
            return;
        }
        self.pending.store(true, Ordering::Relaxed);
        let r = self.result.clone();
        let pending = self.pending.clone();
        std::thread::spawn(move || {
            let p = rfd::FileDialog::new().pick_file();
            *r.lock().unwrap() = p;
            pending.store(false, Ordering::Relaxed);
        });
    }

    /// True while the dialog is open (UI keeps ticking so the result lands promptly).
    pub fn pending(&self) -> bool {
        self.pending.load(Ordering::Relaxed)
    }

    /// Returns the chosen path if ready, clearing the slot.
    pub fn take(&self) -> Option<PathBuf> {
        self.result.lock().unwrap().take()
    }
}

//! GUI の多重起動の禁止: 利用者ごとのロックファイル(%APPDATA%\LightVC\gui.lock)を排他で握る。
//! プロセスが終われば(落ちても)OS が外す。2 つ目の起動は、動いている窓を前に出して終わる。

use std::fs::{File, OpenOptions, TryLockError};
use std::path::PathBuf;

fn dir() -> PathBuf {
    std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir).join("LightVC")
}

/// 先に動いている GUI が無ければロックを握って true(プロセスが終わるまで握り続ける)。
/// ロックファイルを作れないときは、起動を止めずに true を返す。
pub fn acquire() -> bool {
    let d = dir();
    let lock = std::fs::create_dir_all(&d).and_then(|_| OpenOptions::new().create(true).truncate(false).write(true).open(d.join("gui.lock")));
    let f: File = match lock {
        Ok(f) => f,
        Err(e) => {
            eprintln!("多重起動の確認ができません({e})。そのまま起動します");
            return true;
        }
    };
    match f.try_lock() {
        Ok(()) => {
            let _ = std::fs::write(d.join("gui.pid"), std::process::id().to_string());
            Box::leak(Box::new(f));
            true
        }
        Err(TryLockError::WouldBlock) => false,
        Err(TryLockError::Error(e)) => {
            eprintln!("多重起動の確認ができません({e})。そのまま起動します");
            true
        }
    }
}

/// 動いている GUI の窓を前に出す。出せなければ、起動済みであることを知らせる。
pub fn notify_running() {
    eprintln!("LightVC はすでに起動しています");
    let pid = std::fs::read_to_string(dir().join("gui.pid")).ok().and_then(|s| s.trim().parse::<u32>().ok());
    #[cfg(windows)]
    if !pid.is_some_and(win::activate) {
        win::message("LightVC はすでに起動しています。");
    }
    #[cfg(not(windows))]
    let _ = pid;
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;

    type Hwnd = *mut c_void;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn EnumWindows(cb: unsafe extern "system" fn(Hwnd, isize) -> i32, lparam: isize) -> i32;
        fn GetWindowThreadProcessId(h: Hwnd, pid: *mut u32) -> u32;
        fn GetWindow(h: Hwnd, cmd: u32) -> Hwnd;
        fn IsWindowVisible(h: Hwnd) -> i32;
        fn IsIconic(h: Hwnd) -> i32;
        fn ShowWindow(h: Hwnd, cmd: i32) -> i32;
        fn SetForegroundWindow(h: Hwnd) -> i32;
        fn MessageBoxW(h: Hwnd, text: *const u16, caption: *const u16, ty: u32) -> i32;
    }

    const GW_OWNER: u32 = 4;
    const SW_RESTORE: i32 = 9;
    const MB_ICONINFORMATION: u32 = 0x40;

    struct Find {
        pid: u32,
        hwnd: Hwnd,
    }

    unsafe extern "system" fn each(h: Hwnd, l: isize) -> i32 {
        let f = unsafe { &mut *(l as *mut Find) };
        let mut pid = 0u32;
        unsafe { GetWindowThreadProcessId(h, &mut pid) };
        if pid == f.pid && unsafe { IsWindowVisible(h) } != 0 && unsafe { GetWindow(h, GW_OWNER) }.is_null() {
            f.hwnd = h;
            return 0;
        }
        1
    }

    /// pid の見えている最上位の窓を元の大きさに戻して前に出す。
    pub fn activate(pid: u32) -> bool {
        let mut f = Find { pid, hwnd: std::ptr::null_mut() };
        unsafe { EnumWindows(each, &mut f as *mut Find as isize) };
        if f.hwnd.is_null() {
            return false;
        }
        unsafe {
            if IsIconic(f.hwnd) != 0 {
                ShowWindow(f.hwnd, SW_RESTORE);
            }
            SetForegroundWindow(f.hwnd) != 0
        }
    }

    pub fn message(text: &str) {
        let w = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
        let (t, c) = (w(text), w("LightVC"));
        unsafe { MessageBoxW(std::ptr::null_mut(), t.as_ptr(), c.as_ptr(), MB_ICONINFORMATION) };
    }
}

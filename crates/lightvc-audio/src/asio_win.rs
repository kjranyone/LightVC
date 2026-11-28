//! Native Windows COM (`IASIO`) backend without requiring the proprietary Steinberg ASIO SDK.
//!
//! Discovers 64-bit ASIO drivers from `HKLM\SOFTWARE\ASIO`, instantiates them via `CoCreateInstance`,
//! and runs input, output, or synchronized full-duplex callbacks into lock-free `rtrb` ring buffers.

#![cfg(all(windows, target_arch = "x86_64"))]

use std::ffi::{c_char, c_double, c_long, c_void, CStr};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};
use rtrb::{Consumer, Producer};

#[derive(Clone, Debug)]
pub struct AsioDeviceInfo {
    pub name: String,
    pub clsid: String,
    pub channels_in: u16,
    pub channels_out: u16,
    /// 入力チャンネルの名前(ドライバが返すもの・例 "IN 1")。
    pub in_channel_names: Vec<String>,
    /// 出力チャンネルの名前(例 "OUT 1")。
    pub out_channel_names: Vec<String>,
    pub sample_rate: u32,
    pub buffer_min: i32,
    pub buffer_max: i32,
    pub buffer_preferred: i32,
    pub buffer_granularity: i32,
}

#[derive(Default)]
pub struct AsioFaultFlags {
    pub disconnected: AtomicBool,
    pub overrun: AtomicU64,
    pub underrun: AtomicU64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

#[repr(C)]
struct AsioChannelInfo {
    channel: c_long,
    is_input: c_long,
    is_active: c_long,
    channel_group: c_long,
    sample_type: c_long,
    name: [c_char; 32],
}

#[repr(C)]
struct AsioBufferInfo {
    is_input: c_long,
    channel_num: c_long,
    buffers: [*mut c_void; 2],
}

#[repr(C)]
struct AsioCallbacks {
    buffer_switch: Option<unsafe extern "C" fn(c_long, c_long)>,
    sample_rate_did_change: Option<unsafe extern "C" fn(c_double)>,
    asio_message: Option<unsafe extern "C" fn(c_long, c_long, *mut c_void, *mut c_double) -> c_long>,
    buffer_switch_time_info: Option<unsafe extern "C" fn(*mut c_void, c_long, c_long) -> *mut c_void>,
}

#[repr(C)]
struct IAsioVtbl {
    query_interface: unsafe extern "system" fn(*mut IAsio, *const Guid, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(*mut IAsio) -> u32,
    release: unsafe extern "system" fn(*mut IAsio) -> u32,
    init: unsafe extern "system" fn(*mut IAsio, *mut c_void) -> c_long,
    get_driver_name: unsafe extern "system" fn(*mut IAsio, *mut c_char),
    get_driver_version: unsafe extern "system" fn(*mut IAsio) -> c_long,
    get_error_message: unsafe extern "system" fn(*mut IAsio, *mut c_char),
    start: unsafe extern "system" fn(*mut IAsio) -> c_long,
    stop: unsafe extern "system" fn(*mut IAsio) -> c_long,
    get_channels: unsafe extern "system" fn(*mut IAsio, *mut c_long, *mut c_long) -> c_long,
    get_latencies: unsafe extern "system" fn(*mut IAsio, *mut c_long, *mut c_long) -> c_long,
    get_buffer_size: unsafe extern "system" fn(*mut IAsio, *mut c_long, *mut c_long, *mut c_long, *mut c_long) -> c_long,
    can_sample_rate: unsafe extern "system" fn(*mut IAsio, c_double) -> c_long,
    get_sample_rate: unsafe extern "system" fn(*mut IAsio, *mut c_double) -> c_long,
    set_sample_rate: unsafe extern "system" fn(*mut IAsio, c_double) -> c_long,
    get_clock_sources: unsafe extern "system" fn(*mut IAsio, *mut c_void, *mut c_long) -> c_long,
    set_clock_source: unsafe extern "system" fn(*mut IAsio, c_long) -> c_long,
    get_sample_position: unsafe extern "system" fn(*mut IAsio, *mut c_void, *mut c_void) -> c_long,
    get_channel_info: unsafe extern "system" fn(*mut IAsio, *mut AsioChannelInfo) -> c_long,
    create_buffers: unsafe extern "system" fn(*mut IAsio, *mut AsioBufferInfo, c_long, c_long, *mut AsioCallbacks) -> c_long,
    dispose_buffers: unsafe extern "system" fn(*mut IAsio) -> c_long,
    control_panel: unsafe extern "system" fn(*mut IAsio) -> c_long,
    future: unsafe extern "system" fn(*mut IAsio, c_long, *mut c_void) -> c_long,
    output_ready: unsafe extern "system" fn(*mut IAsio) -> c_long,
}

#[repr(C)]
struct IAsio {
    vtbl: *const IAsioVtbl,
}

const HKEY_LOCAL_MACHINE: isize = -2147483646i32 as isize;
const KEY_READ: u32 = 0x20019;
const REG_SZ: u32 = 1;
const CLSCTX_INPROC_SERVER: u32 = 1;
const COINIT_APARTMENTTHREADED: u32 = 2;
const ASE_OK: c_long = 0;
const ASE_SUCCESS: c_long = 0x3f4847a0;

const K_ASIO_SELECTOR_SUPPORTED: c_long = 1;
const K_ASIO_ENGINE_VERSION: c_long = 2;
const K_ASIO_RESET_REQUEST: c_long = 3;
const K_ASIO_BUFFER_SIZE_CHANGE: c_long = 4;
const K_ASIO_RESYNC_REQUEST: c_long = 5;
const K_ASIO_LATENCIES_CHANGED: c_long = 6;
const K_ASIO_SUPPORTS_TIME_INFO: c_long = 7;
const K_ASIO_OVERLOAD: c_long = 15;

#[link(name = "ole32")]
unsafe extern "system" {
    fn CoInitializeEx(pv_reserved: *mut c_void, dw_co_init: u32) -> i32;
    fn CoCreateInstance(
        rclsid: *const Guid,
        p_unk_outer: *mut c_void,
        dw_cls_context: u32,
        riid: *const Guid,
        ppv: *mut *mut c_void,
    ) -> i32;
    fn CLSIDFromString(lpsz: *const u16, pclsid: *mut Guid) -> i32;
}

#[link(name = "advapi32")]
unsafe extern "system" {
    fn RegOpenKeyExW(
        hkey: isize,
        lp_sub_key: *const u16,
        ul_options: u32,
        sam_desired: u32,
        phk_result: *mut isize,
    ) -> i32;
    fn RegEnumKeyExW(
        hkey: isize,
        dw_index: u32,
        lp_name: *mut u16,
        lpcch_name: *mut u32,
        lp_reserved: *mut u32,
        lp_class: *mut u16,
        lpcch_class: *mut u32,
        lpft_last_write_time: *mut c_void,
    ) -> i32;
    fn RegQueryValueExW(
        hkey: isize,
        lp_value_name: *const u16,
        lp_reserved: *mut u32,
        lp_type: *mut u32,
        lp_data: *mut u8,
        lpcb_data: *mut u32,
    ) -> i32;
    fn RegCloseKey(hkey: isize) -> i32;
}

#[link(name = "user32")]
unsafe extern "system" {
    fn GetDesktopWindow() -> *mut c_void;
}

fn to_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn from_wide_nul(buf: &[u16]) -> String {
    let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    String::from_utf16_lossy(&buf[..end])
}

fn ensure_com_sta() {
    thread_local! {
        static INIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    INIT.with(|done| {
        if !done.get() {
            let _ = unsafe { CoInitializeEx(null_mut(), COINIT_APARTMENTTHREADED) };
            done.set(true);
        }
    });
}

struct RegKey(isize);

impl Drop for RegKey {
    fn drop(&mut self) {
        if self.0 != 0 {
            unsafe { RegCloseKey(self.0) };
        }
    }
}

fn reg_read_sz(hkey: isize, value_name: &str) -> Option<String> {
    let wname = to_wide(value_name);
    let mut ty: u32 = 0;
    let mut bytes: u32 = 0;
    let rc = unsafe { RegQueryValueExW(hkey, wname.as_ptr(), null_mut(), &mut ty, null_mut(), &mut bytes) };
    if rc != 0 || ty != REG_SZ || bytes < 2 {
        return None;
    }
    let mut buf = vec![0u16; (bytes as usize).div_ceil(2)];
    let mut actual_bytes = (buf.len() * 2) as u32;
    let rc = unsafe {
        RegQueryValueExW(
            hkey,
            wname.as_ptr(),
            null_mut(),
            &mut ty,
            buf.as_mut_ptr() as *mut u8,
            &mut actual_bytes,
        )
    };
    if rc != 0 {
        return None;
    }
    let s = from_wide_nul(&buf).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

struct RegistryDriver {
    name: String,
    clsid: String,
}

fn enumerate_registry_drivers() -> Vec<RegistryDriver> {
    let mut out = Vec::new();
    let root_path = to_wide("SOFTWARE\\ASIO");
    let mut h_asio: isize = 0;
    if unsafe { RegOpenKeyExW(HKEY_LOCAL_MACHINE, root_path.as_ptr(), 0, KEY_READ, &mut h_asio) } != 0 {
        return out;
    }
    let _asio_guard = RegKey(h_asio);

    let mut idx = 0u32;
    loop {
        let mut name_buf = [0u16; 256];
        let mut name_len = name_buf.len() as u32;
        let rc = unsafe {
            RegEnumKeyExW(
                h_asio,
                idx,
                name_buf.as_mut_ptr(),
                &mut name_len,
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut(),
            )
        };
        if rc != 0 {
            break;
        }
        idx += 1;
        let subkey = from_wide_nul(&name_buf[..name_len as usize]);
        let wsub = to_wide(&subkey);
        let mut h_sub: isize = 0;
        if unsafe { RegOpenKeyExW(h_asio, wsub.as_ptr(), 0, KEY_READ, &mut h_sub) } != 0 {
            continue;
        }
        let _sub_guard = RegKey(h_sub);
        if let Some(clsid) = reg_read_sz(h_sub, "CLSID") {
            let desc = reg_read_sz(h_sub, "Description").unwrap_or(subkey);
            out.push(RegistryDriver { name: desc, clsid });
        }
    }
    out
}

pub fn has_drivers() -> bool {
    !enumerate_registry_drivers().is_empty()
}

struct RawDriver {
    ptr: *mut IAsio,
}

unsafe impl Send for RawDriver {}

impl RawDriver {
    fn open(clsid_str: &str) -> Result<Self> {
        ensure_com_sta();
        let wclsid = to_wide(clsid_str);
        let mut guid = Guid { data1: 0, data2: 0, data3: 0, data4: [0; 8] };
        let hr = unsafe { CLSIDFromString(wclsid.as_ptr(), &mut guid) };
        if hr < 0 {
            bail!("Invalid ASIO CLSID {clsid_str}: 0x{hr:08x}");
        }
        let mut pv: *mut c_void = null_mut();
        let hr = unsafe { CoCreateInstance(&guid, null_mut(), CLSCTX_INPROC_SERVER, &guid, &mut pv) };
        if hr < 0 || pv.is_null() {
            bail!("CoCreateInstance failed for {clsid_str}: 0x{hr:08x}");
        }
        let drv = Self { ptr: pv as *mut IAsio };
        let hwnd = unsafe { GetDesktopWindow() };
        let ok = unsafe { ((*(*drv.ptr).vtbl).init)(drv.ptr, hwnd) };
        if ok == 0 {
            let msg = drv.error_message();
            bail!("ASIO init failed: {msg}");
        }
        Ok(drv)
    }

    fn vtbl(&self) -> &IAsioVtbl {
        unsafe { &*(*self.ptr).vtbl }
    }

    fn error_message(&self) -> String {
        let mut buf = [0u8; 128];
        unsafe { (self.vtbl().get_error_message)(self.ptr, buf.as_mut_ptr() as *mut c_char) };
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
        let s = cstr.to_string_lossy().trim().to_string();
        if s.is_empty() { "unknown ASIO error".to_string() } else { s }
    }

    fn is_ok(code: c_long) -> bool {
        code == ASE_OK || code == ASE_SUCCESS
    }

    fn channels(&self) -> Result<(c_long, c_long)> {
        let (mut ins, mut outs) = (0, 0);
        let rc = unsafe { (self.vtbl().get_channels)(self.ptr, &mut ins, &mut outs) };
        if !Self::is_ok(rc) {
            bail!("ASIO getChannels failed ({rc}): {}", self.error_message());
        }
        Ok((ins, outs))
    }

    fn buffer_size(&self) -> Result<(c_long, c_long, c_long, c_long)> {
        let (mut min_s, mut max_s, mut pref_s, mut gran) = (0, 0, 0, 0);
        let rc = unsafe { (self.vtbl().get_buffer_size)(self.ptr, &mut min_s, &mut max_s, &mut pref_s, &mut gran) };
        if !Self::is_ok(rc) {
            bail!("ASIO getBufferSize failed ({rc}): {}", self.error_message());
        }
        Ok((min_s, max_s, pref_s, gran))
    }

    fn sample_rate(&self) -> Option<f64> {
        let mut sr = 0.0f64;
        let rc = unsafe { (self.vtbl().get_sample_rate)(self.ptr, &mut sr) };
        if Self::is_ok(rc) && sr > 0.0 { Some(sr) } else { None }
    }

    fn can_sample_rate(&self, sr: f64) -> bool {
        let rc = unsafe { (self.vtbl().can_sample_rate)(self.ptr, sr) };
        Self::is_ok(rc)
    }

    fn set_sample_rate(&self, sr: f64) -> Result<()> {
        let rc = unsafe { (self.vtbl().set_sample_rate)(self.ptr, sr) };
        if Self::is_ok(rc) {
            Ok(())
        } else {
            bail!("ASIO setSampleRate({sr}) failed ({rc}): {}", self.error_message())
        }
    }

    fn channel_info(&self, channel: c_long, is_input: bool) -> Result<AsioChannelInfo> {
        let mut info = AsioChannelInfo {
            channel,
            is_input: i32::from(is_input),
            is_active: 0,
            channel_group: 0,
            sample_type: 0,
            name: [0; 32],
        };
        let rc = unsafe { (self.vtbl().get_channel_info)(self.ptr, &mut info) };
        if !Self::is_ok(rc) {
            bail!("ASIO getChannelInfo failed ({rc}): {}", self.error_message());
        }
        Ok(info)
    }
}

impl Drop for RawDriver {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { (self.vtbl().release)(self.ptr) };
            self.ptr = null_mut();
        }
    }
}

static ACTIVE_DRIVER_NAME: Mutex<Option<String>> = Mutex::new(None);
static CACHED_DEVICES: Mutex<Vec<AsioDeviceInfo>> = Mutex::new(Vec::new());

/// Enumerates ASIO drivers from the Windows registry.
/// Avoids instantiating COM objects while an ASIO stream is active so live audio is undisturbed.
pub fn list_devices() -> Vec<AsioDeviceInfo> {
    if ACTIVE_DRIVER_NAME.lock().unwrap().is_some() {
        let cached = CACHED_DEVICES.lock().unwrap();
        if !cached.is_empty() {
            return cached.clone();
        }
    }

    let entries = enumerate_registry_drivers();
    let mut devices = Vec::new();
    for entry in entries {
        let Ok(drv) = RawDriver::open(&entry.clsid) else { continue };
        let Ok((ins, outs)) = drv.channels() else { continue };
        if ins <= 0 && outs <= 0 {
            continue;
        }
        let Ok((bmin, bmax, bpref, bgran)) = drv.buffer_size() else { continue };
        let sr = drv
            .sample_rate()
            .or_else(|| {
                [48_000.0, 44_100.0, 96_000.0, 88_200.0]
                    .into_iter()
                    .find(|&r| drv.can_sample_rate(r))
            })
            .unwrap_or(48_000.0);
        let names = |n: c_long, input: bool| -> Vec<String> {
            (0..n.max(0))
                .map(|c| {
                    drv.channel_info(c, input)
                        .map(|i| channel_name(&i.name))
                        .ok()
                        .filter(|n| !n.is_empty())
                        .unwrap_or_else(|| format!("{} {}", if input { "IN" } else { "OUT" }, c + 1))
                })
                .collect()
        };
        let (in_channel_names, out_channel_names) = (names(ins, true), names(outs, false));
        devices.push(AsioDeviceInfo {
            name: entry.name,
            clsid: entry.clsid,
            channels_in: ins.max(0) as u16,
            channels_out: outs.max(0) as u16,
            in_channel_names,
            out_channel_names,
            sample_rate: sr.round() as u32,
            buffer_min: bmin,
            buffer_max: bmax,
            buffer_preferred: bpref,
            buffer_granularity: bgran,
        });
    }
    *CACHED_DEVICES.lock().unwrap() = devices.clone();
    devices
}

fn channel_name(raw: &[c_char; 32]) -> String {
    let bytes: Vec<u8> = raw.iter().take_while(|&&b| b != 0).map(|&b| b as u8).collect();
    String::from_utf8_lossy(&bytes).trim().to_string()
}

#[derive(Clone, Copy)]
struct ChannelBuffer {
    bufs: [*mut c_void; 2],
    sample_type: c_long,
}

unsafe impl Send for ChannelBuffer {}

struct CallbackCtx {
    stopping: Arc<AtomicBool>,
    started: Arc<AtomicBool>,
    driver_ptr: usize,
    buffer_frames: usize,
    in_channels: Vec<ChannelBuffer>,
    out_channels: Vec<ChannelBuffer>,
    capture_tx: Option<Producer<f32>>,
    playback_rx: Option<Consumer<f32>>,
    flags: Arc<AsioFaultFlags>,
}

static CB_CTX: Mutex<Option<CallbackCtx>> = Mutex::new(None);

fn snap_buffer_size(requested: Option<u32>, min_s: i32, max_s: i32, pref_s: i32, gran: i32) -> i32 {
    let min_s = min_s.max(16);
    let max_s = max_s.max(min_s);
    let pref_s = if pref_s >= min_s && pref_s <= max_s { pref_s } else { min_s };
    let Some(req) = requested else {
        return pref_s;
    };
    let req = (req as i32).clamp(min_s, max_s);
    if min_s == max_s {
        return min_s;
    }
    if gran == -1 {
        let mut best = min_s;
        let mut best_diff = (req - best).abs();
        let mut cur = min_s;
        while cur <= max_s {
            let d = (req - cur).abs();
            if d < best_diff {
                best = cur;
                best_diff = d;
            }
            match cur.checked_mul(2) {
                Some(next) if next > cur => cur = next,
                _ => break,
            }
        }
        return best;
    }
    if gran > 0 {
        let steps = ((req - min_s) + gran / 2) / gran;
        return (min_s + steps * gran).clamp(min_s, max_s);
    }
    req
}

#[inline]
unsafe fn read_sample(ptr: *const u8, idx: usize, sample_type: c_long) -> f32 {
    unsafe {
        match sample_type {
            16 => {
                let p = (ptr as *const [u8; 2]).add(idx);
                i16::from_le_bytes(*p) as f32 / 32768.0
            }
            0 => {
                let p = (ptr as *const [u8; 2]).add(idx);
                i16::from_be_bytes(*p) as f32 / 32768.0
            }
            17 => {
                let p = ptr.add(idx * 3);
                let v = (*p as i32) | ((*p.add(1) as i32) << 8) | ((*p.add(2) as i8 as i32) << 16);
                v as f32 / 8388608.0
            }
            1 => {
                let p = ptr.add(idx * 3);
                let v = ((*p as i8 as i32) << 16) | ((*p.add(1) as i32) << 8) | (*p.add(2) as i32);
                v as f32 / 8388608.0
            }
            18 => {
                let p = (ptr as *const [u8; 4]).add(idx);
                i32::from_le_bytes(*p) as f32 / 2147483648.0
            }
            2 => {
                let p = (ptr as *const [u8; 4]).add(idx);
                i32::from_be_bytes(*p) as f32 / 2147483648.0
            }
            19 => {
                let p = (ptr as *const [u8; 4]).add(idx);
                f32::from_le_bytes(*p)
            }
            3 => {
                let p = (ptr as *const [u8; 4]).add(idx);
                f32::from_be_bytes(*p)
            }
            20 => {
                let p = (ptr as *const [u8; 8]).add(idx);
                f64::from_le_bytes(*p) as f32
            }
            4 => {
                let p = (ptr as *const [u8; 8]).add(idx);
                f64::from_be_bytes(*p) as f32
            }
            24 => {
                let p = (ptr as *const [u8; 4]).add(idx);
                i32::from_le_bytes(*p) as f32 / 32768.0
            }
            27 => {
                let p = (ptr as *const [u8; 4]).add(idx);
                i32::from_le_bytes(*p) as f32 / 8388608.0
            }
            _ => 0.0,
        }
    }
}

#[inline]
unsafe fn write_sample(ptr: *mut u8, idx: usize, sample_type: c_long, s: f32) {
    let s = s.clamp(-1.0, 1.0);
    unsafe {
        match sample_type {
            16 => {
                let v = (s * 32767.0).round() as i16;
                *(ptr as *mut [u8; 2]).add(idx) = v.to_le_bytes();
            }
            0 => {
                let v = (s * 32767.0).round() as i16;
                *(ptr as *mut [u8; 2]).add(idx) = v.to_be_bytes();
            }
            17 => {
                let v = (s * 8388607.0).round() as i32;
                let b = v.to_le_bytes();
                let p = ptr.add(idx * 3);
                *p = b[0];
                *p.add(1) = b[1];
                *p.add(2) = b[2];
            }
            1 => {
                let v = (s * 8388607.0).round() as i32;
                let b = v.to_be_bytes();
                let p = ptr.add(idx * 3);
                *p = b[1];
                *p.add(1) = b[2];
                *p.add(2) = b[3];
            }
            18 => {
                let v = (s * 2147483647.0).round() as i32;
                *(ptr as *mut [u8; 4]).add(idx) = v.to_le_bytes();
            }
            2 => {
                let v = (s * 2147483647.0).round() as i32;
                *(ptr as *mut [u8; 4]).add(idx) = v.to_be_bytes();
            }
            19 => {
                *(ptr as *mut [u8; 4]).add(idx) = s.to_le_bytes();
            }
            3 => {
                *(ptr as *mut [u8; 4]).add(idx) = s.to_be_bytes();
            }
            20 => {
                *(ptr as *mut [u8; 8]).add(idx) = (s as f64).to_le_bytes();
            }
            4 => {
                *(ptr as *mut [u8; 8]).add(idx) = (s as f64).to_be_bytes();
            }
            24 => {
                let v = (s * 32767.0).round() as i32;
                *(ptr as *mut [u8; 4]).add(idx) = v.to_le_bytes();
            }
            27 => {
                let v = (s * 8388607.0).round() as i32;
                *(ptr as *mut [u8; 4]).add(idx) = v.to_le_bytes();
            }
            _ => {}
        }
    }
}

unsafe extern "C" fn on_buffer_switch(double_buffer_index: c_long, _direct_process: c_long) {
    let idx = double_buffer_index as usize;
    if idx > 1 {
        return;
    }
    let Ok(mut guard) = CB_CTX.try_lock() else { return };
    let Some(ctx) = guard.as_mut() else { return };
    if ctx.stopping.load(Ordering::Acquire) {
        return;
    }
    let n_frames = ctx.buffer_frames;
    let n_in = ctx.in_channels.len();
    if let Some(tx) = ctx.capture_tx.as_mut()
        && n_in > 0
    {
        let inv = 1.0 / n_in as f32;
        for f in 0..n_frames {
            let mut sum = 0.0f32;
            for ch in &ctx.in_channels {
                let ptr = ch.bufs[idx] as *const u8;
                if !ptr.is_null() {
                    sum += unsafe { read_sample(ptr, f, ch.sample_type) };
                }
            }
            if tx.push(sum * inv).is_err() {
                ctx.flags.overrun.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    let n_out = ctx.out_channels.len();
    if let Some(rx) = ctx.playback_rx.as_mut()
        && n_out > 0
    {
        for f in 0..n_frames {
            let s = match rx.pop() {
                Ok(v) => v,
                Err(_) => {
                    ctx.flags.underrun.fetch_add(1, Ordering::Relaxed);
                    0.0
                }
            };
            for ch in &ctx.out_channels {
                let ptr = ch.bufs[idx] as *mut u8;
                if !ptr.is_null() {
                    unsafe { write_sample(ptr, f, ch.sample_type, s) };
                }
            }
        }
        let drv = ctx.driver_ptr as *mut IAsio;
        if !drv.is_null() {
            unsafe { ((*(*drv).vtbl).output_ready)(drv) };
        }
    }
}

unsafe extern "C" fn on_buffer_switch_time_info(
    params: *mut c_void,
    double_buffer_index: c_long,
    direct_process: c_long,
) -> *mut c_void {
    unsafe { on_buffer_switch(double_buffer_index, direct_process) };
    params
}

unsafe extern "C" fn on_sample_rate_did_change(_srate: c_double) {
    if let Ok(guard) = CB_CTX.try_lock()
        && let Some(ctx) = guard.as_ref()
        && ctx.started.load(Ordering::Acquire)
    {
        ctx.flags.disconnected.store(true, Ordering::Release);
    }
}

unsafe extern "C" fn on_asio_message(
    selector: c_long,
    value: c_long,
    _message: *mut c_void,
    _opt: *mut c_double,
) -> c_long {
    match selector {
        K_ASIO_SELECTOR_SUPPORTED => match value {
            K_ASIO_ENGINE_VERSION
            | K_ASIO_RESET_REQUEST
            | K_ASIO_BUFFER_SIZE_CHANGE
            | K_ASIO_RESYNC_REQUEST
            | K_ASIO_LATENCIES_CHANGED
            | K_ASIO_SUPPORTS_TIME_INFO
            | K_ASIO_OVERLOAD => 1,
            _ => 0,
        },
        K_ASIO_ENGINE_VERSION => 2,
        K_ASIO_SUPPORTS_TIME_INFO => 1,
        K_ASIO_RESET_REQUEST | K_ASIO_RESYNC_REQUEST => {
            if let Ok(guard) = CB_CTX.try_lock()
                && let Some(ctx) = guard.as_ref()
                && ctx.started.load(Ordering::Acquire)
            {
                ctx.flags.disconnected.store(true, Ordering::Release);
            }
            1
        }
        K_ASIO_BUFFER_SIZE_CHANGE | K_ASIO_LATENCIES_CHANGED | K_ASIO_OVERLOAD => 1,
        _ => 0,
    }
}

static mut CALLBACKS: AsioCallbacks = AsioCallbacks {
    buffer_switch: Some(on_buffer_switch),
    sample_rate_did_change: Some(on_sample_rate_did_change),
    asio_message: Some(on_asio_message),
    buffer_switch_time_info: Some(on_buffer_switch_time_info),
};

pub struct AsioSession {
    driver: RawDriver,
    stopping: Arc<AtomicBool>,
    pub driver_name: String,
    pub sample_rate: u32,
    pub buffer_frames: u32,
}

impl AsioSession {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        driver_name: &str,
        requested_sr: u32,
        requested_buffer: Option<u32>,
        in_channels: &[u16],
        out_channels: &[u16],
        capture_tx: Option<Producer<f32>>,
        playback_rx: Option<Consumer<f32>>,
        flags: Arc<AsioFaultFlags>,
    ) -> Result<Self> {
        let want_in = capture_tx.is_some();
        let want_out = playback_rx.is_some();
        if !want_in && !want_out {
            bail!("ASIO session requires at least input or output");
        }
        {
            let active = ACTIVE_DRIVER_NAME.lock().unwrap();
            if let Some(ref cur) = *active {
                bail!("ASIO ドライバ '{cur}' が既に使用中です(ASIO は同時に 1 つまで)");
            }
        }

        let entries = enumerate_registry_drivers();
        let entry = entries
            .into_iter()
            .find(|e| e.name.eq_ignore_ascii_case(driver_name))
            .ok_or_else(|| anyhow!("ASIO ドライバ '{driver_name}' が見つかりません"))?;

        let driver = RawDriver::open(&entry.clsid)?;
        let (total_in, total_out) = driver.channels()?;
        let pick = |want: bool, chosen: &[u16], total: c_long, default_n: c_long, what: &str| -> Result<Vec<c_long>> {
            if !want {
                return Ok(Vec::new());
            }
            if total <= 0 {
                bail!("ASIO デバイス '{}' に{what}チャンネルがありません", entry.name);
            }
            if chosen.is_empty() {
                return Ok((0..total.min(default_n)).collect());
            }
            let mut v: Vec<c_long> = Vec::with_capacity(chosen.len());
            for &c in chosen {
                if c as c_long >= total {
                    bail!("ASIO デバイス '{}' に{what}チャンネル {} はありません(1〜{total})", entry.name, c + 1);
                }
                if !v.contains(&(c as c_long)) {
                    v.push(c as c_long);
                }
            }
            Ok(v)
        };
        let in_list = pick(want_in, in_channels, total_in, 1, "入力")?;
        let out_list = pick(want_out, out_channels, total_out, 2, "出力")?;
        let (n_in, n_out) = (in_list.len(), out_list.len());

        let target_sr = requested_sr as f64;
        let actual_sr = if driver.can_sample_rate(target_sr) {
            let _ = driver.set_sample_rate(target_sr);
            driver.sample_rate().unwrap_or(target_sr)
        } else if let Some(cur_sr) = driver.sample_rate() {
            cur_sr
        } else {
            let fallback = [48_000.0, 44_100.0, 96_000.0, 88_200.0]
                .into_iter()
                .find(|&r| driver.can_sample_rate(r))
                .ok_or_else(|| anyhow!("ASIO デバイス '{}' で対応サンプルレートが見つかりません", entry.name))?;
            driver.set_sample_rate(fallback)?;
            driver.sample_rate().unwrap_or(fallback)
        };

        let (min_s, max_s, pref_s, gran) = driver.buffer_size()?;
        let mut chosen_buf = snap_buffer_size(requested_buffer, min_s, max_s, pref_s, gran);

        let mut buf_infos = Vec::with_capacity(n_in + n_out);
        for &ch in &in_list {
            buf_infos.push(AsioBufferInfo {
                is_input: 1,
                channel_num: ch,
                buffers: [null_mut(), null_mut()],
            });
        }
        for &ch in &out_list {
            buf_infos.push(AsioBufferInfo {
                is_input: 0,
                channel_num: ch,
                buffers: [null_mut(), null_mut()],
            });
        }

        let cb_ptr = std::ptr::addr_of_mut!(CALLBACKS);
        let mut rc = unsafe {
            (driver.vtbl().create_buffers)(
                driver.ptr,
                buf_infos.as_mut_ptr(),
                buf_infos.len() as c_long,
                chosen_buf,
                cb_ptr,
            )
        };
        if !RawDriver::is_ok(rc) && pref_s > 0 && chosen_buf != pref_s {
            chosen_buf = pref_s;
            rc = unsafe {
                (driver.vtbl().create_buffers)(
                    driver.ptr,
                    buf_infos.as_mut_ptr(),
                    buf_infos.len() as c_long,
                    chosen_buf,
                    cb_ptr,
                )
            };
        }
        if !RawDriver::is_ok(rc) {
            bail!("ASIO createBuffers({chosen_buf}) failed ({rc}): {}", driver.error_message());
        }

        let mut in_channels = Vec::with_capacity(n_in);
        for info in &buf_infos[..n_in] {
            let cinfo = driver.channel_info(info.channel_num, true)?;
            in_channels.push(ChannelBuffer {
                bufs: info.buffers,
                sample_type: cinfo.sample_type,
            });
        }
        let mut out_channels = Vec::with_capacity(n_out);
        for info in &buf_infos[n_in..] {
            let cinfo = driver.channel_info(info.channel_num, false)?;
            out_channels.push(ChannelBuffer {
                bufs: info.buffers,
                sample_type: cinfo.sample_type,
            });
        }

        let stopping = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicBool::new(false));
        *CB_CTX.lock().unwrap() = Some(CallbackCtx {
            stopping: stopping.clone(),
            started: started.clone(),
            driver_ptr: driver.ptr as usize,
            buffer_frames: chosen_buf as usize,
            in_channels,
            out_channels,
            capture_tx,
            playback_rx,
            flags,
        });
        *ACTIVE_DRIVER_NAME.lock().unwrap() = Some(entry.name.clone());

        let start_rc = unsafe { (driver.vtbl().start)(driver.ptr) };
        if !RawDriver::is_ok(start_rc) {
            *CB_CTX.lock().unwrap() = None;
            *ACTIVE_DRIVER_NAME.lock().unwrap() = None;
            unsafe { (driver.vtbl().dispose_buffers)(driver.ptr) };
            bail!("ASIO start failed ({start_rc}): {}", driver.error_message());
        }
        started.store(true, Ordering::Release);
        let final_sr = driver.sample_rate().unwrap_or(actual_sr);

        Ok(Self {
            driver,
            stopping,
            driver_name: entry.name,
            sample_rate: final_sr.round() as u32,
            buffer_frames: chosen_buf as u32,
        })
    }
}

impl Drop for AsioSession {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        unsafe {
            (self.driver.vtbl().stop)(self.driver.ptr);
        }
        *CB_CTX.lock().unwrap() = None;
        unsafe {
            (self.driver.vtbl().dispose_buffers)(self.driver.ptr);
        }
        *ACTIVE_DRIVER_NAME.lock().unwrap() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_buffer_size_handles_power_of_two_and_granularity() {
        assert_eq!(snap_buffer_size(None, 64, 2048, 256, -1), 256);
        assert_eq!(snap_buffer_size(Some(240), 64, 2048, 256, -1), 256);
        assert_eq!(snap_buffer_size(Some(128), 64, 2048, 256, -1), 128);
        assert_eq!(snap_buffer_size(Some(240), 64, 2048, 256, 32), 256);
        assert_eq!(snap_buffer_size(Some(230), 64, 2048, 256, 32), 224);
        assert_eq!(snap_buffer_size(Some(240), 256, 256, 256, 0), 256);
    }

    #[test]
    fn asio_registry_probe_and_open_when_available() {
        let devs = list_devices();
        if devs.is_empty() {
            return;
        }
        let d = &devs[0];
        let (cap_tx, mut cap_rx) = rtrb::RingBuffer::new(65536);
        let (mut pb_tx, pb_rx) = rtrb::RingBuffer::new(65536);
        for _ in 0..4096 {
            let _ = pb_tx.push(0.0);
        }
        let flags = Arc::new(AsioFaultFlags::default());
        let cap = if d.channels_in > 0 { Some(cap_tx) } else { None };
        let pb = if d.channels_out > 0 { Some(pb_rx) } else { None };
        let session = AsioSession::start(&d.name, 48_000, Some(240), &[], &[], cap, pb, flags.clone()).expect("start ASIO session");
        std::thread::sleep(std::time::Duration::from_millis(200));
        let mut got = 0usize;
        while cap_rx.pop().is_ok() {
            got += 1;
        }
        eprintln!(
            "ASIO live test: driver={} sr={} buf={} cap_samples={} disconnected={}",
            session.driver_name,
            session.sample_rate,
            session.buffer_frames,
            got,
            flags.disconnected.load(Ordering::Relaxed)
        );
        assert!(session.sample_rate > 0);
        assert!(session.buffer_frames > 0);
        if d.channels_in > 0 {
            assert!(got > 0, "expected ASIO capture callbacks to produce samples");
        }
        assert!(!flags.disconnected.load(Ordering::Relaxed));
        drop(session);

        if d.channels_in > 1 {
            let last = d.channels_in - 1;
            let (cap_tx, mut cap_rx) = rtrb::RingBuffer::new(65536);
            let flags = Arc::new(AsioFaultFlags::default());
            let session = AsioSession::start(&d.name, 48_000, Some(240), &[last], &[], Some(cap_tx), None, flags).expect("start ASIO on the last input");
            std::thread::sleep(std::time::Duration::from_millis(200));
            let mut got = 0usize;
            while cap_rx.pop().is_ok() {
                got += 1;
            }
            eprintln!("ASIO input {} ({}): cap_samples={got}", last + 1, d.in_channel_names[last as usize]);
            assert!(got > 0);
            drop(session);
            let (cap_tx, _cap_rx) = rtrb::RingBuffer::new(1024);
            let r = AsioSession::start(&d.name, 48_000, Some(240), &[d.channels_in], &[], Some(cap_tx), None, Arc::new(AsioFaultFlags::default()));
            assert!(r.is_err(), "a channel past the last must be refused");
        }

        if d.channels_out > 3 {
            let (mut pb_tx, pb_rx) = rtrb::RingBuffer::new(65536);
            for _ in 0..4096 {
                let _ = pb_tx.push(0.0);
            }
            let flags = Arc::new(AsioFaultFlags::default());
            let session = AsioSession::start(&d.name, 48_000, Some(240), &[], &[2, 3], None, Some(pb_rx), flags.clone()).expect("start ASIO on outputs 3-4");
            std::thread::sleep(std::time::Duration::from_millis(200));
            let consumed = pb_tx.slots().saturating_sub(65536 - 4096);
            eprintln!("ASIO outputs 3-4 ({} / {}): consumed={consumed}", d.out_channel_names[2], d.out_channel_names[3]);
            assert!(consumed > 0, "expected ASIO playback callbacks to consume samples");
            assert!(!flags.disconnected.load(Ordering::Relaxed));
            drop(session);
        }
    }
}

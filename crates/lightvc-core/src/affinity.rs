/// 混成の CPU(P / E コア)で呼び出しのスレッドを P コアに置き、MMCSS の Pro Audio クラスへ入れる(D626: E コアに載ると 1 ブロックの尾が伸びる・横取りを減らす)。
/// 既定は P コアごとに論理 CPU を 1 つだけ(同じコアの 2 つ目のハードウェアのスレッドに重いスレッドが並ばない・段 P94 で p99 4.82 → 4.45 ms)。LVC_PIN_SMT で P コアの全ての論理 CPU。
/// Linux は /sys/devices/cpu_core/cpus があるとき・Windows は効率の級が最大の CPU 集合(GetSystemCpuSetInformation)・混成でない機械と他の OS では何もしない・LVC_NO_PIN で無効。
/// 並べ替えはスレッドごとに初回だけ行う(繰り返しの呼び出しは直ちに戻る)。LVC_DRIVER_CORE=1 のときはワーカー(この関数)が最初の P コアを音声の駆動系の専用として空ける。
pub fn pin_current_to_perf() {
    pin_perf_inner(false);
}

/// 音声の経路の駆動系(process_block を呼ぶスレッド)の置き方。既定は pin_current_to_perf と同じ・LVC_DRIVER_CORE=1 で最初の P コアへの固定(ワーカーは残りへ)。
pub fn pin_audio_driver() {
    pin_perf_inner(true);
}

fn pin_perf_inner(driver: bool) {
    thread_local! {
        static DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    if DONE.with(|d| d.replace(true)) {
        return;
    }
    let (no_pin, smt) = cfg();
    if no_pin {
        return;
    }
    #[cfg(target_os = "linux")]
    {
        let _ = driver;
        pin_linux(smt);
    }
    #[cfg(target_os = "windows")]
    win::pin_perf(smt, driver);
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    let _ = smt;
}

/// GUI などの重くないスレッドを E コアへ置く(音声のスレッドの P コアを奪わない)。混成でない機械では何もしない。
pub fn pin_current_to_efficiency() {
    thread_local! {
        static DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    if DONE.with(|d| d.replace(true)) {
        return;
    }
    if cfg().0 {
        return;
    }
    #[cfg(target_os = "windows")]
    win::pin_eff();
    #[cfg(not(target_os = "windows"))]
    {}
}

/// Windows のタイマ分解能を 1 ms に上げる(既定は約 15.6 ms・sleep / park_timeout の待ちの尾を作る)。LVC_NO_TIMER で無効。
pub fn raise_timer_resolution() {
    #[cfg(target_os = "windows")]
    win::timer_1ms();
    #[cfg(not(target_os = "windows"))]
    {}
}

fn cfg() -> (bool, bool) {
    static V: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *V.get_or_init(|| (std::env::var_os("LVC_NO_PIN").is_some(), std::env::var_os("LVC_PIN_SMT").is_some()))
}

#[cfg(target_os = "linux")]
fn pin_linux(smt: bool) {
    let Ok(s) = std::fs::read_to_string("/sys/devices/cpu_core/cpus") else { return };
    let mut cpus = Vec::new();
    for part in s.trim().split(',') {
        let (a, b) = part.split_once('-').unwrap_or((part, part));
        let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) else { return };
        cpus.extend(a..=b);
    }
    let first = |c: usize| -> bool {
        let Ok(t) = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")) else { return true };
        t.trim().split([',', '-']).filter_map(|x| x.parse::<usize>().ok()).min().is_none_or(|m| m == c)
    };
    if !smt && cpus.iter().any(|&c| first(c)) {
        cpus.retain(|&c| first(c));
    }
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in &cpus {
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    if !cpus.is_empty() {
        unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    }
}

#[cfg(target_os = "windows")]
mod win {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
    use std::sync::OnceLock;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetSystemCpuSetInformation(info: *mut c_void, len: u32, ret: *mut u32, process: *mut c_void, flags: u32) -> i32;
        fn SetThreadSelectedCpuSets(thread: *mut c_void, ids: *const u32, n: u32) -> i32;
        fn SetThreadAffinityMask(thread: *mut c_void, mask: usize) -> usize;
        fn GetCurrentThread() -> *mut c_void;
    }

    #[link(name = "winmm")]
    unsafe extern "system" {
        fn timeBeginPeriod(ms: u32) -> u32;
    }

    #[link(name = "avrt")]
    unsafe extern "system" {
        fn AvSetMmThreadCharacteristicsW(task: *const u16, index: *mut u32) -> *mut c_void;
    }

    /// SYSTEM_CPU_SET_INFORMATION の可変長の並びから (Id, Group, LogicalProcessorIndex, CoreIndex, EfficiencyClass) を読む。
    fn sets() -> Vec<(u32, u16, u8, u8, u8)> {
        let mut len = 0u32;
        unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut len, std::ptr::null_mut(), 0) };
        if len == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u64; (len as usize).div_ceil(8)];
        if unsafe { GetSystemCpuSetInformation(buf.as_mut_ptr().cast(), len, &mut len, std::ptr::null_mut(), 0) } == 0 {
            return Vec::new();
        }
        let b = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), len as usize) };
        let (mut out, mut o) = (Vec::new(), 0usize);
        while o + 20 <= b.len() {
            let size = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
            let kind = u32::from_le_bytes([b[o + 4], b[o + 5], b[o + 6], b[o + 7]]);
            if size < 20 {
                break;
            }
            if kind == 0 {
                out.push((u32::from_le_bytes([b[o + 8], b[o + 9], b[o + 10], b[o + 11]]), u16::from_le_bytes([b[o + 12], b[o + 13]]), b[o + 14], b[o + 15], b[o + 18]));
            }
            o += size;
        }
        out
    }

    /// 混成の置き方の計画(プロセスで 1 回): P コアの各コア 1 つ・P コアの全て・E コア(先頭の P コアの論理 CPU も駆動系の固定用に)。
    struct Plan {
        perf_core: Vec<u32>,
        perf_smt: Vec<u32>,
        perf_core_rest: Vec<u32>,
        first_lp: Option<u64>,
        eff: Vec<u32>,
    }

    fn plan() -> Option<&'static Plan> {
        static V: OnceLock<Option<Plan>> = OnceLock::new();
        V.get_or_init(|| {
            let all = sets();
            let top = all.iter().map(|s| s.4).max()?;
            if all.iter().all(|s| s.4 == top) {
                return None;
            }
            let (mut perf_core, mut perf_smt, mut eff) = (Vec::new(), Vec::new(), Vec::new());
            let mut seen = std::collections::HashSet::new();
            let mut first_lp = None;
            for &(id, group, lp, core, class) in &all {
                if class == top {
                    perf_smt.push(id);
                    if seen.insert((group, core)) {
                        if first_lp.is_none() {
                            first_lp = Some(1u64 << lp);
                        }
                        perf_core.push(id);
                    }
                } else {
                    eff.push(id);
                }
            }
            let perf_core_rest = if perf_core.len() > 1 { perf_core[1..].to_vec() } else { perf_core.clone() };
            Some(Plan { perf_core, perf_smt, perf_core_rest, first_lp, eff })
        })
        .as_ref()
    }

    static WARNED: AtomicBool = AtomicBool::new(false);

    fn apply(ids: &[u32]) {
        let r = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), ids.as_ptr(), ids.len() as u32) };
        if r == 0 && !WARNED.swap(true, Ordering::Relaxed) {
            eprintln!("SetThreadSelectedCpuSets が失敗(スレッドは既定の置き方のまま)");
        }
    }

    pub(super) fn pin_perf(smt: bool, driver: bool) {
        let reserve = driver_core_on();
        if let Some(p) = plan() {
            if driver && reserve && let Some(mask) = p.first_lp {
                let r = unsafe { SetThreadAffinityMask(GetCurrentThread(), mask as usize) };
                if r != 0 {
                    mmcss();
                    return;
                }
            }
            match (smt, reserve) {
                (false, true) => apply(if driver { &p.perf_core } else { &p.perf_core_rest }),
                (false, false) => apply(&p.perf_core),
                (true, _) => apply(&p.perf_smt),
            }
        }
        mmcss();
    }

    fn driver_core_on() -> bool {
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| std::env::var_os("LVC_DRIVER_CORE").is_some())
    }

    pub(super) fn pin_eff() {
        if let Some(p) = plan() {
            apply(&p.eff);
        }
    }

    /// MMCSS の Pro Audio クラスへ(横取りを減らす・スケジューラの量子 1 個分の尾が主因のため)。LVC_NO_MMCSS で無効。
    pub(super) fn mmcss() {
        static ON: AtomicU8 = AtomicU8::new(0);
        let on = match ON.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => {
                let on = std::env::var_os("LVC_NO_MMCSS").is_none();
                ON.store(if on { 1 } else { 2 }, Ordering::Relaxed);
                on
            }
        };
        if !on {
            return;
        }
        let task: Vec<u16> = "Pro Audio".encode_utf16().chain(std::iter::once(0)).collect();
        let mut idx = 0u32;
        unsafe { AvSetMmThreadCharacteristicsW(task.as_ptr(), &mut idx) };
    }

    pub(super) fn timer_1ms() {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| {
            if std::env::var_os("LVC_NO_TIMER").is_none() {
                unsafe { timeBeginPeriod(1) };
            }
        });
    }
}

/// 受け渡しの待ちを最大 LVC_SPIN_US µs 回してから寝る(D627: 起床の遅れを避ける・既定 500・0 = すぐ寝る。Windows 実機では 0/1000/2000 より 500 が最良だった)。
pub fn spin_us() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LVC_SPIN_US").ok().and_then(|s| s.parse().ok()).unwrap_or(500))
}

pub fn recv_spin<T>(rx: &std::sync::mpsc::Receiver<T>) -> Result<T, std::sync::mpsc::RecvError> {
    let us = spin_us();
    if us > 0 {
        let t = std::time::Instant::now();
        loop {
            match rx.try_recv() {
                Ok(v) => return Ok(v),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return Err(std::sync::mpsc::RecvError),
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if t.elapsed().as_micros() as u64 >= us {
                break;
            }
            std::hint::spin_loop();
        }
    }
    rx.recv()
}

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;

struct TeamShared {
    epoch: AtomicUsize,
    done: AtomicUsize,
    job: AtomicPtr<()>,
    stop: AtomicBool,
}

/// 同じ仕事を n 個のスレッドで分けて回す(呼び出しのスレッド = 0 番・残りは P コアに常駐し LVC_SPIN_US 回し待ちしてから寝る)。
pub struct Team {
    sh: Arc<TeamShared>,
    n: usize,
    hs: Vec<std::thread::JoinHandle<()>>,
}

type Job<'a> = &'a (dyn Fn(usize) + Sync);

impl Team {
    pub fn new(n: usize) -> Self {
        raise_timer_resolution();
        let sh = Arc::new(TeamShared { epoch: AtomicUsize::new(0), done: AtomicUsize::new(0), job: AtomicPtr::new(std::ptr::null_mut()), stop: AtomicBool::new(false) });
        let hs = (1..n.max(1))
            .map(|rank| {
                let sh = sh.clone();
                std::thread::spawn(move || {
                    pin_current_to_perf();
                    let us = spin_us();
                    let mut seen = 0usize;
                    loop {
                        let t0 = std::time::Instant::now();
                        let mut i = 0u32;
                        loop {
                            let g = sh.epoch.load(Ordering::Acquire);
                            if g != seen {
                                seen = g;
                                break;
                            }
                            i = i.wrapping_add(1);
                            if i.is_multiple_of(64) && t0.elapsed().as_micros() as u64 >= us {
                                std::thread::park_timeout(std::time::Duration::from_micros(500));
                            } else {
                                std::hint::spin_loop();
                            }
                        }
                        if sh.stop.load(Ordering::Acquire) {
                            return;
                        }
                        let f = unsafe { &*(sh.job.load(Ordering::Acquire) as *const Job) };
                        f(rank);
                        sh.done.fetch_add(1, Ordering::AcqRel);
                    }
                })
            })
            .collect();
        Self { sh, n: n.max(1), hs }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    /// f(0..n) を並べて回し、全部が終わってから戻る。
    pub fn run(&self, f: Job) {
        if self.n == 1 {
            f(0);
            return;
        }
        let fp: *const Job = &f;
        self.sh.job.store(fp as *mut (), Ordering::Release);
        self.sh.done.store(0, Ordering::Release);
        self.sh.epoch.fetch_add(1, Ordering::AcqRel);
        for h in &self.hs {
            h.thread().unpark();
        }
        f(0);
        while self.sh.done.load(Ordering::Acquire) < self.n - 1 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Team {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::Release);
        self.sh.epoch.fetch_add(1, Ordering::AcqRel);
        for h in self.hs.drain(..) {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}

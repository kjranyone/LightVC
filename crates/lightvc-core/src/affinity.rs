/// 混成の CPU(P / E コア)で呼び出しのスレッドを P コアに置く(D626: E コアに載ると 1 ブロックの尾が伸びる)。
/// 既定は P コアごとに論理 CPU を 1 つだけ(同じコアの 2 つ目のハードウェアのスレッドに重いスレッドが並ばない・段 P94 で p99 4.82 → 4.45 ms)。LVC_PIN_SMT で P コアの全ての論理 CPU。
/// Linux は /sys/devices/cpu_core/cpus があるとき・Windows は効率の級が最大の CPU 集合(GetSystemCpuSetInformation)・混成でない機械と他の OS では何もしない・LVC_NO_PIN で無効。
pub fn pin_current_to_perf() {
    if std::env::var_os("LVC_NO_PIN").is_some() {
        return;
    }
    let smt = std::env::var_os("LVC_PIN_SMT").is_some();
    #[cfg(target_os = "linux")]
    {
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
    win::pin_perf(smt);
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    let _ = smt;
}

#[cfg(target_os = "windows")]
mod win {
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetSystemCpuSetInformation(info: *mut c_void, len: u32, ret: *mut u32, process: *mut c_void, flags: u32) -> i32;
        fn SetThreadSelectedCpuSets(thread: *mut c_void, ids: *const u32, n: u32) -> i32;
        fn GetCurrentThread() -> *mut c_void;
    }

    /// SYSTEM_CPU_SET_INFORMATION の可変長の並びから (Id, Group, CoreIndex, EfficiencyClass) を読む。
    fn sets() -> Vec<(u32, u16, u8, u8)> {
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
                out.push((u32::from_le_bytes([b[o + 8], b[o + 9], b[o + 10], b[o + 11]]), u16::from_le_bytes([b[o + 12], b[o + 13]]), b[o + 15], b[o + 18]));
            }
            o += size;
        }
        out
    }

    pub(super) fn pin_perf(smt: bool) {
        let all = sets();
        let Some(top) = all.iter().map(|s| s.3).max() else { return };
        if all.iter().all(|s| s.3 == top) {
            return;
        }
        let mut ids = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &(id, group, core, class) in &all {
            if class == top && (smt || seen.insert((group, core))) {
                ids.push(id);
            }
        }
        if !ids.is_empty() {
            unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), ids.as_ptr(), ids.len() as u32) };
        }
    }
}

/// 受け渡しの待ちを最大 LVC_SPIN_US µs 回してから寝る(D627: 起床の遅れを避ける・既定 1000・0 = すぐ寝る)。
pub fn spin_us() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("LVC_SPIN_US").ok().and_then(|s| s.parse().ok()).unwrap_or(1000))
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

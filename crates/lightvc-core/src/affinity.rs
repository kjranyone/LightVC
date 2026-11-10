/// 混成の CPU(P / E コア)で呼び出しのスレッドを P コアに置く(D626: E コアに載ると 1 ブロックの尾が伸びる)。
/// Linux は /sys/devices/cpu_core/cpus があるときだけ・混成でない機械と他の OS では何もしない・LVC_NO_PIN で無効。
pub fn pin_current_to_perf() {
    if std::env::var_os("LVC_NO_PIN").is_some() {
        return;
    }
    #[cfg(target_os = "linux")]
    {
        let Ok(s) = std::fs::read_to_string("/sys/devices/cpu_core/cpus") else { return };
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        let mut n = 0usize;
        for part in s.trim().split(',') {
            let (a, b) = part.split_once('-').unwrap_or((part, part));
            let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) else { return };
            for c in a..=b {
                unsafe { libc::CPU_SET(c, &mut set) };
                n += 1;
            }
        }
        if n > 0 {
            unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
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

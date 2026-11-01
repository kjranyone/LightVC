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

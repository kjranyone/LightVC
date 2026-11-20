//! 段ごとの時間の内訳(試験の構成だけ・製品の構成では `prof!` は中身をそのまま実行する)。
//! `prof!(ID, 式)` が式の実行時間を ID の席に足す(全てのスレッドの和)。組ごとの席は `prof!(member ID, 式)`(`slot(i)` でスレッドに組の番号を付ける)。
//! 測るのは `enable(true)` の間だけ(既定は切 = 他の試験の時間に効かない)。内訳の試験 = snv_stream の `profile_k3g`。

#[cfg(test)]
pub(crate) use imp::*;

/// 式の実行時間を席 `$id` に足す(試験の構成だけ)。`prof!(member K, 式)` は組ごとの席。
macro_rules! prof {
    (member $id:ident, $e:expr) => {{
        #[cfg(test)]
        let __t = $crate::prof::start();
        let __r = $e;
        #[cfg(test)]
        $crate::prof::stop($crate::prof::member($crate::prof::$id), __t);
        __r
    }};
    ($id:ident, $e:expr) => {{
        #[cfg(test)]
        let __t = $crate::prof::start();
        let __r = $e;
        #[cfg(test)]
        $crate::prof::stop($crate::prof::$id, __t);
        __r
    }};
}
pub(crate) use prof;

#[cfg(test)]
mod imp {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Instant;

    pub(crate) const GATE_FEAT: usize = 0;
    pub(crate) const GATE_NET: usize = 1;
    pub(crate) const MASK_NET: usize = 2;
    pub(crate) const MASK_HEAD: usize = 3;
    pub(crate) const F_BUF: usize = 4;
    pub(crate) const F_CMNDF: usize = 5;
    pub(crate) const F_LOWBAND: usize = 6;
    pub(crate) const F_DP: usize = 7;
    pub(crate) const F_POWER: usize = 8;
    pub(crate) const F_ENV: usize = 9;
    pub(crate) const F_PER: usize = 10;
    pub(crate) const FIXENV: usize = 11;
    pub(crate) const MODEL: usize = 12;
    pub(crate) const MIX: usize = 13;
    pub(crate) const EXC: usize = 14;
    pub(crate) const VOC: usize = 15;
    pub(crate) const VOC_PRE: usize = 16;
    pub(crate) const VOC_UP: usize = 17;
    pub(crate) const VOC_RB: usize = 18;
    pub(crate) const VOC_WAIT: usize = 19;
    pub(crate) const VOC_POST: usize = 20;
    pub(crate) const TOTAL: usize = 21;
    pub(crate) const GATE: usize = 22;
    pub(crate) const FRONT: usize = 23;
    /// 組ごとの席(組 i は MEMBER + i·PER_MEMBER から)。
    pub(crate) const MEMBER: usize = 32;
    pub(crate) const PER_MEMBER: usize = 8;
    pub(crate) const M_ALL: usize = 0;
    pub(crate) const M_CPC: usize = 1;
    pub(crate) const M_PLAY: usize = 2;
    pub(crate) const M_CONV: usize = 3;
    pub(crate) const M_ATTN: usize = 4;
    pub(crate) const N: usize = MEMBER + 4 * PER_MEMBER;

    static ON: AtomicBool = AtomicBool::new(false);
    static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];

    thread_local! {
        static SLOT: Cell<usize> = const { Cell::new(0) };
    }

    pub(crate) fn enable(on: bool) {
        ON.store(on, Ordering::Release);
    }

    pub(crate) fn reset() {
        NS.iter().for_each(|v| v.store(0, Ordering::Release));
    }

    /// 席ごとの合計の ns。
    pub(crate) fn snapshot() -> Vec<u64> {
        NS.iter().map(|v| v.load(Ordering::Acquire)).collect()
    }

    /// このスレッドが今から回す組の番号。
    pub(crate) fn slot(i: usize) {
        SLOT.with(|s| s.set(i));
    }

    /// 組ごとの席の番号。
    pub(crate) fn member(k: usize) -> usize {
        MEMBER + SLOT.with(|s| s.get()) * PER_MEMBER + k
    }

    pub(crate) fn start() -> Option<Instant> {
        ON.load(Ordering::Relaxed).then(Instant::now)
    }

    pub(crate) fn stop(id: usize, t: Option<Instant>) {
        if let Some(t) = t {
            NS[id].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

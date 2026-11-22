//! 出力の番(paper_snv 段 P95 step c・training/snv_out_guard.py と同じ計算)。網が見る値は変えず、出力の波形だけに掛ける。足す遅延 0・発話の統計なし。
//! 大きさの番: ブロックごとの入力と出力の RMS(dB)から、入力の動きが裏付けない大きさの出力を下げる。
//!   ref = 入力の RMS の直近 REF_N ブロックの最大(出力は入力より 2 ブロック遅れるので、先の 2 ブロックを含む)。
//!   番組の水準 P_in・P_out = 立ち上がりの速い・下がりの遅い追跡。許す水準 = P_out + (ref − P_in) + MARGIN。
//! 尖頭値の番: ブロックの尖頭値が CEIL を超えたら利得を直ちに下げ、1 ブロックあたり LIM_REL dB で戻す。残った超過(|y| > CEIL)は tanh で 1.0 の内側へ丸める。

const REF_N: usize = 9;
const ATTACK: f32 = 0.3;
const REL_ACTIVE: f32 = 2.0 * 0.005;
const REL_IDLE: f32 = 0.2 * 0.005;
const ACTIVE_DB: f32 = 30.0;
const MARGIN: f32 = 10.0;
const G_MIN: f32 = -40.0;
const UP_DB: f32 = 3.0;
const RAMP: usize = 48;
const CEIL: f32 = 0.9;
const LIM_REL: f32 = 0.3;
pub const P_NOMINAL: f32 = -26.0;

pub fn level_db(x: &[f32]) -> f32 {
    let ms = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len().max(1) as f64;
    (10.0 * ms.max(1e-14).log10()) as f32
}

/// 発話の組の声の水準(ブロックの RMS の 90 % 点・dB)。登録の波形から P_in の初期値を作る。
pub fn speech_level(utts: &[&[f32]], hop: usize) -> f32 {
    let mut l: Vec<f32> = utts.iter().flat_map(|u| u.chunks_exact(hop).map(level_db)).collect();
    if l.is_empty() {
        return P_NOMINAL;
    }
    l.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    l[((l.len() - 1) as f64 * 0.9).round() as usize]
}

pub fn soft_clip(v: f32) -> f32 {
    let a = v.abs();
    if a <= CEIL {
        v
    } else {
        v.signum() * (CEIL + (1.0 - CEIL) * ((a - CEIL) / (1.0 - CEIL)).tanh())
    }
}

pub struct OutGuard {
    hist: [f32; REF_N],
    i: usize,
    p_in: f32,
    p_out: f32,
    g: f32,
    gl: f32,
    gt: f32,
    last: [f32; 5],
}

impl Default for OutGuard {
    fn default() -> Self {
        Self::new(P_NOMINAL, P_NOMINAL)
    }
}

impl OutGuard {
    pub fn new(p_in: f32, p_out: f32) -> Self {
        Self { hist: [-140.0; REF_N], i: 0, p_in, p_out, g: 0.0, gl: 0.0, gt: 0.0, last: [0.0; 5] }
    }

    pub fn set_program(&mut self, p_in: f32, p_out: f32) {
        self.p_in = p_in;
        self.p_out = p_out;
    }

    /// 直前のブロックの [ref − P_in, 出力の RMS(番の前), 許す水準, 大きさの番の利得 dB, 尖頭値の番の利得 dB]。
    pub fn last(&self) -> [f32; 5] {
        self.last
    }

    /// x = 入力のブロック・y = 同じ呼び出しの出力のブロック(その場で書き換える)。
    pub fn process(&mut self, x: &[f32], y: &mut [f32]) {
        self.hist[self.i] = level_db(x);
        self.i = (self.i + 1) % REF_N;
        let r = self.hist.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lo = level_db(y);
        let rel = if r > self.p_in - ACTIVE_DB { REL_ACTIVE } else { REL_IDLE };
        self.p_in = if r > self.p_in { self.p_in + ATTACK * (r - self.p_in) } else { self.p_in - rel };
        let allow = self.p_out + (r - self.p_in) + MARGIN;
        let want = (allow - lo).clamp(G_MIN, 0.0);
        let g = if want < self.g { want } else { want.min(self.g + UP_DB) };
        let lg = lo + g;
        self.p_out = if lg > self.p_out { self.p_out + ATTACK * (lg - self.p_out) } else { self.p_out - rel };
        let pk = y.iter().fold(0f32, |a, v| a.max(v.abs())) * 10f32.powf(g / 20.0);
        let need = if pk > CEIL { 20.0 * (CEIL / pk).log10() } else { 0.0 };
        let gl = need.min((self.gl + LIM_REL).min(0.0));
        let gt = g + gl;
        let (a0, a1) = (10f32.powf(self.gt / 20.0), 10f32.powf(gt / 20.0));
        let n = if gt < self.gt { RAMP.min(y.len()) } else { y.len() };
        if a0 != 1.0 || a1 != 1.0 {
            for (j, v) in y.iter_mut().enumerate() {
                let a = if j < n { a0 + (a1 - a0) * (j + 1) as f32 / n as f32 } else { a1 };
                *v *= a;
            }
        }
        for v in y.iter_mut() {
            *v = soft_clip(*v);
        }
        self.gl = gl;
        self.gt = gt;
        self.g = g;
        self.last = [r - self.p_in, lo, allow, g, gl];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_output_follows_input() {
        let mut g = OutGuard::new(-20.0, -20.0);
        let x: Vec<f32> = (0..240 * 40).map(|i| 0.1 * ((i as f32) * 0.05).sin()).collect();
        let mut y = x.clone();
        for (xb, yb) in x.chunks_exact(240).zip(y.chunks_exact_mut(240)) {
            g.process(xb, yb);
        }
        assert!(x.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn burst_in_silence_is_lowered_and_peak_bounded() {
        let mut g = OutGuard::new(-20.0, -20.0);
        let x = vec![1e-4f32; 240];
        let mut worst = 0f32;
        for k in 0..20 {
            let mut y: Vec<f32> = (0..240).map(|i| 1.6 * ((i + k) as f32 * 0.3).sin()).collect();
            g.process(&x, &mut y);
            if k > 0 {
                worst = worst.max(y.iter().fold(0f32, |a, v| a.max(v.abs())));
            }
        }
        assert!(worst < 0.05, "{worst}");
        let mut y = [1.7f32, -1.7, 0.5];
        y.iter_mut().for_each(|v| *v = soft_clip(*v));
        assert!(y[0] <= 1.0 && y[0] > 0.9 && y[1] >= -1.0 && y[2] == 0.5);
    }
}

//! 入力のノイズ除去(変換の前・48 kHz・240 サンプルのブロックごと・因果)。
//!
//! - `Spectral`: 自作のスペクトルの Wiener(雑音は MCRA で追い、最小値の 2 倍で頭を抑える・判定指向の事前 SNR・下限 −12 dB)。
//!   利得は隣 ±2 ビンの最大に広げる(倍音の裾を削らない)。窓 480(sqrt Hann)・送り 240・FFT 512。遅延 +240 サンプル(5 ms)。ファン・空調・ヒスなどの定常の雑音向け。
//! - `Rnnoise`: RNNoise(nnnoiseless・BSD-3-Clause)。10 ms の枠ごと。遅延 +720 サンプル(15 ms = 枠を溜める 240 + 内部 480)。
//!   キーボード・物音など変化する雑音も抑えるが、吐息・ささやきも弱めることがある。

use lightvc_core::fft::fft;
use lightvc_core::snv_stream::SnvStream;

const HOP: usize = SnvStream::BLOCK;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Denoise {
    #[default]
    Off,
    Spectral,
    Rnnoise,
}

impl Denoise {
    pub const ALL: [Denoise; 3] = [Denoise::Off, Denoise::Spectral, Denoise::Rnnoise];

    pub fn label(self) -> &'static str {
        match self {
            Denoise::Off => "切",
            Denoise::Spectral => "定常の雑音",
            Denoise::Rnnoise => "RNNoise",
        }
    }

    /// 足される遅延(48 kHz のサンプル数)。
    pub fn latency_samples(self) -> usize {
        match self {
            Denoise::Off => 0,
            Denoise::Spectral => HOP,
            Denoise::Rnnoise => HOP + nnnoiseless::DenoiseState::FRAME_SIZE,
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(Denoise::Off),
            "spectral" => Ok(Denoise::Spectral),
            "rnnoise" => Ok(Denoise::Rnnoise),
            _ => Err(format!("off / spectral / rnnoise のどれか: {s}")),
        }
    }
}

pub struct Denoiser {
    kind: Denoise,
    spectral: Option<Box<Spectral>>,
    rnn: Option<Rnn>,
}

impl Denoiser {
    pub fn new(kind: Denoise) -> Self {
        Self {
            kind,
            spectral: (kind == Denoise::Spectral).then(|| Box::new(Spectral::new())),
            rnn: (kind == Denoise::Rnnoise).then(Rnn::new),
        }
    }

    pub fn kind(&self) -> Denoise {
        self.kind
    }

    /// 1 ブロック(HOP サンプル)をその場で処理する。
    pub fn process(&mut self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), HOP);
        if let Some(s) = self.spectral.as_mut() {
            s.process(x);
        } else if let Some(r) = self.rnn.as_mut() {
            r.process(x);
        }
    }
}

const W: usize = 2 * HOP;
const N: usize = 512;
const K: usize = N / 2 + 1;
/// MCRA の最小値を取り直す間隔(フレーム・0.5 秒)。
const MIN_FRAMES: u32 = 100;

struct Spectral {
    win: [f64; W],
    inbuf: [f32; W],
    tail: [f64; HOP],
    re: [f64; N],
    im: [f64; N],
    noise: [f64; K],
    smooth: [f64; K],
    smin: [f64; K],
    stmp: [f64; K],
    presence: [f64; K],
    prev: [f64; K],
    frames: u32,
}

impl Spectral {
    fn new() -> Self {
        let mut win = [0.0; W];
        for (n, w) in win.iter_mut().enumerate() {
            *w = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / W as f64).cos()).sqrt();
        }
        Self {
            win,
            inbuf: [0.0; W],
            tail: [0.0; HOP],
            re: [0.0; N],
            im: [0.0; N],
            noise: [0.0; K],
            smooth: [0.0; K],
            smin: [0.0; K],
            stmp: [0.0; K],
            presence: [0.0; K],
            prev: [1.0; K],
            frames: 0,
        }
    }

    fn process(&mut self, x: &mut [f32]) {
        const ALPHA_S: f64 = 0.7;
        const ALPHA_P: f64 = 0.2;
        const ALPHA_D: f64 = 0.95;
        const DELTA: f64 = 5.0;
        const DD: f64 = 0.98;
        const XI_MIN: f64 = 0.003;
        const G_MIN: f64 = 0.25;
        const EPS: f64 = 1e-12;
        const SPREAD: usize = 2;
        const BIAS: f64 = 2.0;

        self.inbuf.copy_within(HOP.., 0);
        self.inbuf[HOP..].copy_from_slice(x);
        for n in 0..N {
            self.re[n] = if n < W { self.inbuf[n] as f64 * self.win[n] } else { 0.0 };
            self.im[n] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        let mut p = [0.0f64; K];
        for (k, v) in p.iter_mut().enumerate() {
            *v = self.re[k] * self.re[k] + self.im[k] * self.im[k];
        }
        let first = self.frames == 0;
        self.frames += 1;
        let renew = self.frames.is_multiple_of(MIN_FRAMES);
        let (mut gamma, mut raw) = ([0.0f64; K], [0.0f64; K]);
        for k in 0..K {
            let pf = 0.25 * p[k.saturating_sub(1)] + 0.5 * p[k] + 0.25 * p[(k + 1).min(K - 1)];
            if first {
                (self.noise[k], self.smooth[k], self.smin[k], self.stmp[k]) = (p[k], pf, pf, pf);
            }
            self.smooth[k] = ALPHA_S * self.smooth[k] + (1.0 - ALPHA_S) * pf;
            if renew {
                self.smin[k] = self.stmp[k].min(self.smooth[k]);
                self.stmp[k] = self.smooth[k];
            } else {
                self.smin[k] = self.smin[k].min(self.smooth[k]);
                self.stmp[k] = self.stmp[k].min(self.smooth[k]);
            }
            let speech = if self.smooth[k] > DELTA * self.smin[k].max(EPS) { 1.0 } else { 0.0 };
            self.presence[k] = ALPHA_P * self.presence[k] + (1.0 - ALPHA_P) * speech;
            let ad = ALPHA_D + (1.0 - ALPHA_D) * self.presence[k];
            self.noise[k] = (ad * self.noise[k] + (1.0 - ad) * p[k]).min(BIAS * self.smin[k]);

            gamma[k] = p[k] / self.noise[k].max(EPS);
            let xi = (DD * self.prev[k] + (1.0 - DD) * (gamma[k] - 1.0).max(0.0)).max(XI_MIN);
            raw[k] = xi / (1.0 + xi);
        }
        for k in 0..K {
            let g = raw[k.saturating_sub(SPREAD)..(k + SPREAD + 1).min(K)].iter().fold(G_MIN, |a, &b| a.max(b));
            self.prev[k] = g * g * gamma[k];
            self.re[k] *= g;
            self.im[k] *= g;
            if k > 0 && k < N / 2 {
                self.re[N - k] *= g;
                self.im[N - k] *= g;
            }
        }
        for v in self.im.iter_mut() {
            *v = -*v;
        }
        fft(&mut self.re, &mut self.im);
        let inv = 1.0 / N as f64;
        for (n, v) in x.iter_mut().enumerate() {
            *v = (self.tail[n] + self.re[n] * inv * self.win[n]) as f32;
            self.tail[n] = self.re[n + HOP] * inv * self.win[n + HOP];
        }
    }
}

struct Rnn {
    state: Box<nnnoiseless::DenoiseState<'static>>,
    inbuf: Vec<f32>,
    outbuf: std::collections::VecDeque<f32>,
    frame: Vec<f32>,
}

impl Rnn {
    const SCALE: f32 = 32768.0;

    fn new() -> Self {
        let f = nnnoiseless::DenoiseState::FRAME_SIZE;
        Self {
            state: nnnoiseless::DenoiseState::new(),
            inbuf: Vec::with_capacity(2 * f),
            outbuf: std::iter::repeat_n(0.0, HOP).collect(),
            frame: vec![0.0; f],
        }
    }

    fn process(&mut self, x: &mut [f32]) {
        let f = nnnoiseless::DenoiseState::FRAME_SIZE;
        self.inbuf.extend(x.iter().map(|v| v * Self::SCALE));
        while self.inbuf.len() >= f {
            self.state.process_frame(&mut self.frame, &self.inbuf[..f]);
            self.outbuf.extend(self.frame.iter().map(|v| v / Self::SCALE));
            self.inbuf.drain(..f);
        }
        for v in x.iter_mut() {
            *v = self.outbuf.pop_front().unwrap_or(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, amp: f32, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*seed >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0) * amp
            })
            .collect()
    }

    fn run(kind: Denoise, x: &[f32]) -> Vec<f32> {
        let mut d = Denoiser::new(kind);
        let mut y = x.to_vec();
        for b in y.as_chunks_mut::<HOP>().0 {
            d.process(b);
        }
        y
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn off_is_identity() {
        let x = noise(10 * HOP, 0.1, &mut 1);
        assert_eq!(run(Denoise::Off, &x), x);
    }

    #[test]
    fn spectral_passes_signal_with_its_latency_and_cuts_steady_noise() {
        let n = 400 * HOP;
        let floor = noise(n, 0.001, &mut 5);
        let burst: Vec<f32> = (0..n)
            .map(|i| {
                let on = (i / 9_600) % 2 == 0;
                floor[i] + if on { 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() } else { 0.0 }
            })
            .collect();
        let y = run(Denoise::Spectral, &burst);
        let d = Denoise::Spectral.latency_samples();
        let tail = n / 2;
        let err: f32 = (tail..n).map(|i| (y[i] - burst[i - d]).powi(2)).sum::<f32>() / (n - tail) as f32;
        assert!(err.sqrt() < 0.02 * rms(&burst[tail..]), "burst err {}", err.sqrt());

        let x = noise(n, 0.05, &mut 7);
        let y = run(Denoise::Spectral, &x);
        let r = rms(&y[tail..]) / rms(&x[tail..]);
        assert!(r < 0.4, "steady noise ratio {r}");
    }

    #[test]
    fn rnnoise_keeps_block_length_and_cuts_noise() {
        let n = 400 * HOP;
        let x = noise(n, 0.05, &mut 3);
        let y = run(Denoise::Rnnoise, &x);
        assert_eq!(y.len(), x.len());
        let r = rms(&y[n / 2..]) / rms(&x[n / 2..]);
        assert!(r < 0.5, "noise ratio {r}");
    }
}

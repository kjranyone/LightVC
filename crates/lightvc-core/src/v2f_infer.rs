//! v2f 出荷経路の一括推論（WAV → mel/f0/prior → SIMD 幹 → cistft → WAV）。
//!
//! `examples/e2e_v2f.rs` で PyTorch と **SNR 95.2 dB** の一致を確認した経路の関数化。
//! streaming（`TrunkState` ＋ ゼロ初期化キャッシュ）が Python の層ごと左ゼロ詰めと
//! 等価なので、一括入力でも内部はブロック実行で回す。

use crate::ship_front as sf;
use crate::simd::{pointwise, Ten, TrunkState, V2fWeights};

pub struct V2fEngine {
    pub w: V2fWeights,
    pub ch: usize,
    pub fb: Vec<f32>,
    pub w2l: Vec<f32>,
}

impl V2fEngine {
    pub fn load(weights: &std::path::Path, fb: &std::path::Path, w2l: &std::path::Path,
                ch: usize, layers: usize) -> std::io::Result<Self> {
        let rd = |p: &std::path::Path| -> std::io::Result<Vec<f32>> {
            Ok(std::fs::read(p)?.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let mut w = V2fWeights::from_flat(&rd(weights)?, 4, ch, layers, 7, 3);
        for l in w.layers.iter_mut() {
            l.norm_mode = 1;              // cummean（出荷モデルの正規化）
        }
        Ok(Self { w, ch, fb: rd(fb)?, w2l: rd(w2l)? })
    }

    /// 一括処理。`noise_seed` は prior の雑音（決定的にするため注入）。
    pub fn process(&self, x: &[f32], noise: &[f32]) -> Vec<f32> {
        let n = x.len();
        let nb = sf::NBIN_S;
        let mel80 = sf::mel(x, &self.fb);
        let (f0, _) = sf::causal_f0(x);
        let t_syn = sf::n_frames(n, sf::HOP_S, sf::NFFT_S);
        let exc = sf::excitation(&f0, n, noise, 0.3);
        let (er, ei) = sf::cstft(&exc, sf::NFFT_S, sf::HOP_S);
        let t = t_syn.min(er[0].len());
        let mut feat = Ten::zeros(4, t, nb);
        for b in 0..nb {
            for ti in 0..t {
                let j = sf::to_frames_idx(ti).min(mel80[0].len() - 1);
                let mut ml = 0.0f32;
                for m in 0..sf::N_MEL {
                    ml += self.w2l[b * sf::N_MEL + m] * mel80[m][j];
                }
                let h = (ml - sf::MEL_REF).exp();
                let (r, i) = (er[b][ti] * h, ei[b][ti] * h);
                feat.at_mut(0, ti)[b] = ml;
                feat.at_mut(1, ti)[b] = r;
                feat.at_mut(2, ti)[b] = i;
                feat.at_mut(3, ti)[b] = ((r * r + i * i).sqrt() + 1e-5).ln();
            }
        }
        let h = pointwise(&feat, &self.w.inp_w, &self.w.inp_b, self.ch);
        let emit = 2usize;
        let mut st = TrunkState::with_emit(&self.w.layers, nb, emit);
        let mut cur = Ten::zeros(self.ch, t, nb);
        let mut b0 = 0usize;
        while b0 + emit <= t {
            let mut xb = Ten::zeros(self.ch, emit, nb);
            for c in 0..self.ch {
                for ti in 0..emit {
                    xb.at_mut(c, ti).copy_from_slice(h.at(c, b0 + ti));
                }
            }
            let y = st.step(&self.w.layers, &xb);
            for c in 0..self.ch {
                for ti in 0..emit {
                    let src = y.at(c, ti).to_vec();
                    cur.at_mut(c, b0 + ti).copy_from_slice(&src);
                }
            }
            b0 += emit;
        }
        let o = pointwise(&cur, &self.w.out_w, &self.w.out_b, 2);
        let (mut sr, mut si) = (vec![vec![0.0f32; t]; nb], vec![vec![0.0f32; t]; nb]);
        for b in 0..nb {
            for ti in 0..t {
                sr[b][ti] = o.at(0, ti)[b];
                si[b][ti] = o.at(1, ti)[b];
            }
        }
        sf::cistft(&sr, &si, n, sf::NFFT_S, sf::HOP_S)
    }
}


/// ストリーミング実行（2.4a-2 の状態機械）。**ブロック＝ HOP_A = 256 サンプル**
/// （解析 1 フレーム ＝ 合成 2 フレーム）。一括 `process()` と同一出力。
///
/// 保持する状態:
///   - 入力リング（NFFT_A、解析窓）
///   - f0 の直近 2 フレーム（因果アップサンプル用）＋ 直近有声 f0（fill）
///   - 位相アキュムレータ（閉形式励起）
///   - 励起の合成窓リング（NFFT_S − HOP_S ＝ 過去 384 サンプル）
///   - `TrunkState`（層ごと左文脈）＋ cummean 走行統計
///   - OLA リング（cistft の重畳残り）
pub struct V2fStream {
    eng: std::sync::Arc<V2fEngine>,
    pool: Option<crate::simd::Pool>,
    in_ring: Vec<f32>,
    n_in: usize,
    front: crate::ship_front::FrontState,
    /// 直近 2 本の解析 mel（合成フレーム 2k, 2k+1 は解析 k−1, k を見る＝to_frames と同一）
    mel_prev: Vec<f32>,
    /// fill 済み f0 の履歴 [k−2, k−1]（因果アップサンプルの端点）
    f0_fill: [f32; 2],
    last_voiced: Option<f32>,
    phase_acc: f64,
    exc_ring: Vec<f32>,
    trunk: TrunkState,
    ola_y: Vec<f32>,
    ola_w: Vec<f32>,
    /// OLA 窓和が閉じるまでの残り無音サンプル (起動スパイク対策・固有遅延区間)
    warmup: usize,
    started: bool,
    dbg: [f32; 3],
    pub dbg_exc_blk: Vec<f32>,
    pub dbg_phi_blk: Vec<f32>,
    pub dbg_f_blk: Vec<f32>,
    pub dbg_feat: Vec<f32>,
    dbg_mel: f32,
    dbg_exc: f32,
    dbg_o: f32,
}

impl V2fStream {
    pub const BLOCK: usize = sf::HOP_A;                 // 256 サンプル ＝ 5.8 ms

    pub fn new(eng: std::sync::Arc<V2fEngine>) -> Self {
        Self::with_threads(eng, 2)
    }

    /// `threads`: 実効スレッド数（thread_budget）。2 なら常駐ワーカ 1 ＋ 自分。
    pub fn with_threads(eng: std::sync::Arc<V2fEngine>, threads: usize) -> Self {
        let trunk = TrunkState::with_emit(&eng.w.layers, sf::NBIN_S, 2);
        let pool = if threads >= 2 { Some(crate::simd::Pool::new(threads - 1)) } else { None };
        Self {
            eng,
            pool,
            in_ring: vec![0.0; sf::NFFT_A],
            n_in: 0,
            front: crate::ship_front::FrontState::new(),
            mel_prev: vec![(1e-5f32).ln(); sf::N_MEL],
            f0_fill: [sf::PRE_VOICED_HZ; 2],
            last_voiced: None,
            phase_acc: 0.0,
            exc_ring: vec![0.0; sf::NFFT_S - sf::HOP_S],
            trunk,
            ola_y: vec![0.0; sf::NFFT_S],
            ola_w: vec![0.0; sf::NFFT_S],
            warmup: sf::NFFT_S - sf::HOP_S,
            started: false,
            dbg: [0.0; 3],
            dbg_exc_blk: Vec::new(),
            dbg_phi_blk: Vec::new(),
            dbg_f_blk: Vec::new(),
            dbg_feat: Vec::new(),
            dbg_mel: 0.0,
            dbg_exc: 0.0,
            dbg_o: 0.0,
        }
    }

    /// 全ストリーム状態を初期化（開始/停止をまたぐ持ち越しを断つ）。
    pub fn reset(&mut self) {
        self.in_ring.iter_mut().for_each(|v| *v = 0.0);
        self.n_in = 0;
        self.front = crate::ship_front::FrontState::new();
        self.mel_prev.iter_mut().for_each(|v| *v = (1e-5f32).ln());
        self.f0_fill = [sf::PRE_VOICED_HZ; 2];
        self.last_voiced = None;
        self.phase_acc = 0.0;
        self.exc_ring.iter_mut().for_each(|v| *v = 0.0);
        self.trunk = TrunkState::with_emit(&self.eng.w.layers, sf::NBIN_S, 2);
        self.ola_y.iter_mut().for_each(|v| *v = 0.0);
        self.ola_w.iter_mut().for_each(|v| *v = 0.0);
        self.warmup = sf::NFFT_S - sf::HOP_S;
        self.started = false;
    }

    /// 256 入力 → 256 出力。`noise` は 256 サンプルの雑音（None なら内蔵 xorshift）。
    pub fn process_block(&mut self, x: &[f32]) -> Vec<f32> {
        self.process_block_with(x, None)
    }

    pub fn process_block_with(&mut self, x: &[f32], noise: Option<&[f32]>) -> Vec<f32> {
        debug_assert_eq!(x.len(), Self::BLOCK);
        // 入力リングを進める（左寄せ: 最右が現在）
        self.in_ring.copy_within(Self::BLOCK.., 0);
        let n0 = sf::NFFT_A - Self::BLOCK;
        self.in_ring[n0..].copy_from_slice(x);

        // 解析 1 フレーム（起動直後は実サンプルのみ＝ゼロ頭。cstft と同じ）
        let (mel1, f0_now, _voi) = self.front.step(&self.in_ring, &self.eng.fb);

        // fill（走行）: 直近有声、無ければ PRE_VOICED
        if f0_now > 50.0 {
            self.last_voiced = Some(f0_now.max(50.0));
        }
        let f0f = self.last_voiced.unwrap_or(sf::PRE_VOICED_HZ);
        self.synth_block(&mel1, f0f, noise)
    }

    /// 合成コア: 条件 mel（1 解析フレーム）と fill 済み f0 から 256 サンプル。
    /// 変換経路（VcStream）は G の出力 mel とシフト済み f0 をここに注入する。
    pub fn synth_block(&mut self, mel1: &[f32], f0f: f32, noise: Option<&[f32]>) -> Vec<f32> {
        let nb = sf::NBIN_S;
        self.n_in += Self::BLOCK;
        self.dbg_mel = mel1.iter().sum::<f32>() / mel1.len() as f32;

        // 励起 256 サンプル。**一括版の frame_upsample_causal と同一**:
        // ブロック k のサンプル i は fill[k−2] → fill[k−1] を (i+1)/256 で補間。
        let (fa, fb_) = (self.f0_fill[0], self.f0_fill[1]);
        self.f0_fill = [self.f0_fill[1], f0f];
        let mut exc = vec![0.0f32; Self::BLOCK];
        let mut st_noise = 0x9e3779b97f4a7c15u64 ^ (self.n_in as u64);
        for (i, e) in exc.iter_mut().enumerate() {
            // 補間は f64 (ship_front::frame_upsample_causal_f64 と同じ理由)
            let fr = (i as f64 + 1.0) / sf::HOP_A as f64;
            let f64v = (fa as f64 * (1.0 - fr) + fb_ as f64 * fr).max(0.0);
            let f = (f64v as f32).max(1e-3);
            self.phase_acc += f64v;
            let phi = (2.0 * std::f64::consts::PI * self.phase_acc / sf::SR as f64)
                .rem_euclid(2.0 * std::f64::consts::PI);
            if i == 0 {
                self.dbg_phi_blk.clear();
                self.dbg_f_blk.clear();
            }
            self.dbg_phi_blk.push(phi as f32);
            self.dbg_f_blk.push(f);
            // K は Python f32 マスクと同一判定 (ship_front::excitation と同じ形)
            let nyqf = sf::SR / 2.0;
            let mut kmax = ((nyqf as f64) / (f as f64)).floor() as usize;
            if kmax >= 1 && (kmax as f32) * f >= nyqf {
                kmax -= 1;
            }
            if ((kmax + 1) as f32) * f < nyqf {
                kmax += 1;
            }
            let kmax = kmax.min(sf::KMAX);
            let v = if kmax == 0 { 0.0 } else {
                let half = (0.5 * phi).sin();
                let sum = if half.abs() < 1e-6 { kmax as f64 }
                          else { ((kmax as f64 + 0.5) * phi).sin() / (2.0 * half) - 0.5 };
                (sum as f32) / (kmax as f32).sqrt()
            };
            let z = match noise {
                Some(nz) => nz[i],
                None => {
                    st_noise ^= st_noise << 13;
                    st_noise ^= st_noise >> 7;
                    st_noise ^= st_noise << 17;
                    ((st_noise >> 40) as f32 / 8388608.0) - 1.0
                }
            };
            *e = 0.7 * v + 0.3 * z;
        }
        self.dbg_exc_blk = exc.clone();

        // 励起の合成フレーム 2 本（過去 384 ＋ 今回 256 → 窓 512 を 2 回）
        let mut ebuf = vec![0.0f32; (sf::NFFT_S - sf::HOP_S) + Self::BLOCK];
        ebuf[..sf::NFFT_S - sf::HOP_S].copy_from_slice(&self.exc_ring);
        ebuf[sf::NFFT_S - sf::HOP_S..].copy_from_slice(&exc);
        let ekeep = ebuf.len() - (sf::NFFT_S - sf::HOP_S);
        self.exc_ring.copy_from_slice(&ebuf[ekeep..]);
        let w = crate::ship_front::hann(sf::NFFT_S);
        let mut feat = Ten::zeros(4, 2, nb);
        let (mut pre, mut pim) = (vec![[0.0f32; 2]; nb], vec![[0.0f32; 2]; nb]);
        for t in 0..2usize {
            let seg = &ebuf[t * sf::HOP_S..t * sf::HOP_S + sf::NFFT_S];
            let (fr_, fi_) = crate::ship_front::frame_rfft_pub(seg, &w);
            // to_frames と同一: 合成 2k は解析 k−1、2k+1 は解析 k を見る
            let msrc: &[f32] = if t == 0 { &self.mel_prev } else { mel1 };
            for b in 0..nb {
                let mut ml = 0.0f32;
                for m in 0..sf::N_MEL {
                    ml += self.eng.w2l[b * sf::N_MEL + m] * msrc[m];
                }
                let h = (ml - sf::MEL_REF).exp();
                let (r, i) = (fr_[b] as f32 * h, fi_[b] as f32 * h);
                pre[b][t] = r;
                pim[b][t] = i;
                feat.at_mut(0, t)[b] = ml;
                feat.at_mut(1, t)[b] = r;
                feat.at_mut(2, t)[b] = i;
                feat.at_mut(3, t)[b] = ((r * r + i * i).sqrt() + 1e-5).ln();
            }
        }
        let _ = (pre, pim);
        self.dbg_feat = feat.d.clone();
        let h0 = pointwise(&feat, &self.eng.w.inp_w, &self.eng.w.inp_b, self.eng.ch);
        let y2 = match &self.pool {
            Some(pl) => unsafe { self.trunk.step_par(&self.eng.w.layers, &h0, pl) },
            None => self.trunk.step(&self.eng.w.layers, &h0),
        };
        let o = pointwise(&y2, &self.eng.w.out_w, &self.eng.w.out_b, 2);

        self.dbg_o = o.d.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // cistft ストリーム: 合成フレーム 2 本を OLA リングへ
        let mut out = vec![0.0f32; Self::BLOCK];
        for t in 0..2usize {
            let mut fr_ = vec![0.0f64; sf::NFFT_S];
            let mut fi_ = vec![0.0f64; sf::NFFT_S];
            for k in 0..nb {
                fr_[k] = o.at(0, t)[k] as f64;
                fi_[k] = -(o.at(1, t)[k] as f64);
            }
            for k in nb..sf::NFFT_S {
                fr_[k] = o.at(0, t)[sf::NFFT_S - k] as f64;
                fi_[k] = o.at(1, t)[sf::NFFT_S - k] as f64;
            }
            crate::ship_front::fft(&mut fr_, &mut fi_);
            // OLA: hop ぶん出して残りを持ち越す
            for i in 0..sf::NFFT_S {
                let v = (fr_[i] / sf::NFFT_S as f64) as f32 * w[i];
                self.ola_y[i] += v;
                self.ola_w[i] += w[i] * w[i];
            }
            for i in 0..sf::HOP_S {
                out[t * sf::HOP_S + i] = if self.warmup > 0 {
                    self.warmup -= 1;
                    0.0
                } else {
                    self.ola_y[i] / self.ola_w[i].max(1e-8)
                };
            }
            self.ola_y.copy_within(sf::HOP_S.., 0);
            self.ola_w.copy_within(sf::HOP_S.., 0);
            for i in sf::NFFT_S - sf::HOP_S..sf::NFFT_S {
                self.ola_y[i] = 0.0;
                self.ola_w[i] = 0.0;
            }
        }
        self.mel_prev.copy_from_slice(mel1);
        self.started = true;
        self.dbg = [self.dbg_mel, self.dbg_exc, self.dbg_o];
        out
    }

    pub fn debug_stats(&self) -> [f32; 3] {
        self.dbg
    }
}

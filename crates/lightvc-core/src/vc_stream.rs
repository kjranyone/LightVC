//! バ美声変換ストリーム: mic → front(mel,f0) → E → G(content, f0シフト) → V。
//!
//! Python 参照系 (training/convert_vc.py / eval_x.py) と同一の系列構成:
//!   - E 入力 = 製品 front の mel80 (×32768 スケール音声)
//!   - G 入力 = [content(768), ln(max(f0*2^(s/12),50)/200), ln(max(rms,1e-4))]
//!     energy は **[-1,1] スケール**波形の HOP512 非重畳 RMS を因果リサンプル
//!     (×32768 のまま取ると log 空間 +10.397 がのる — R-X 第 1 走の実バグ)
//!   - 励起 f0 = シフト済み f0 の走行 fill (発話統計なし)
//!   - f0 シフトは半音単位の固定ノブ (cartridge 定数)
//!
//! 追加遅延は E/G の CTX (左文脈) ぶんのみ = 先読み 0。

use crate::eg::{Eg1d, Eg1dH, EgStream};

/// V は [-1,1] スケール波形で学習 (train_gvoc.to_gpu /32767)。E/G 系は x32768
/// 慣習の mel なので、V へ渡す直前にこの定数を引く (training/eval_g1.V_MEL_ADAPT
/// と同一値・実測 copy-syn PESQ +0.40)。出力は [-1,1] スケール。
pub const V_MEL_ADAPT: f32 = 10.397_207_f32; // ln(32768)
use crate::ship_front as sf;
use crate::v2f_infer::V2fStream;
use std::sync::Arc;

/// E/G の精度: f32 か f16（f16 は帯域半減で RTF ~2 倍。判定は proxy 再検証済みのみ出荷）
pub enum EgAny {
    F32(Arc<Eg1d>),
    F16(Arc<Eg1dH>),
}

impl EgAny {
    fn stream(&self) -> EgStream {
        match self {
            EgAny::F32(n) => EgStream::new(n),
            EgAny::F16(n) => EgStream::new_h(n),
        }
    }

    fn step(&self, st: &mut EgStream, x: &[f32]) -> Vec<f32> {
        match self {
            EgAny::F32(n) => st.step(n, x),
            EgAny::F16(n) => st.step_h(n, x),
        }
    }
}

pub struct VcStream {
    e: EgAny,
    g: EgAny,
    es: EgStream,
    gs: EgStream,
    v: V2fStream,
    front: sf::FrontState,
    in_ring: Vec<f32>,
    ratio: f32,
    last_voiced: Option<f32>,
    // energy (HOP512 非重畳 RMS, [-1,1] スケール)
    rms_acc: f64,
    rms_n: usize,
    last_rms: f32,
    fb: Vec<f32>,
}

impl VcStream {
    pub const BLOCK: usize = V2fStream::BLOCK; // 256 = 5.8ms

    pub fn new(e: Arc<Eg1d>, g: Arc<Eg1d>, v: V2fStream, fb: Vec<f32>, shift_semitones: f32) -> Self {
        Self::new_any(EgAny::F32(e), EgAny::F32(g), v, fb, shift_semitones)
    }

    pub fn new_any(e: EgAny, g: EgAny, v: V2fStream, fb: Vec<f32>, shift_semitones: f32) -> Self {
        let es = e.stream();
        let gs = g.stream();
        VcStream {
            e,
            g,
            es,
            gs,
            v,
            front: sf::FrontState::new(),
            in_ring: vec![0.0; sf::NFFT_A],
            // torch と同じ丸め: f64 で 2^(s/12) を計算してから f32 へ
            // (2f32.powf は最終ビットが違い、全有声フレームの系統ずれ→位相積分で増幅)
            ratio: (2.0f64.powf(shift_semitones as f64 / 12.0)) as f32,
            last_voiced: None,
            rms_acc: 0.0,
            rms_n: 0,
            last_rms: 0.0,
            fb,
        }
    }

    /// x: 256 サンプル (×32768 スケール = 学習系の慣習)。返り: 256 サンプル。
    pub fn process_block(&mut self, x: &[f32]) -> Vec<f32> {
        self.process_block_with(x, None)
    }

    pub fn process_block_with(&mut self, x: &[f32], noise: Option<&[f32]>) -> Vec<f32> {
        debug_assert_eq!(x.len(), Self::BLOCK);
        // 入力リング + front (V2fStream と同一の左寄せ 1 フレーム/ブロック)
        self.in_ring.copy_within(Self::BLOCK.., 0);
        let n0 = sf::NFFT_A - Self::BLOCK;
        self.in_ring[n0..].copy_from_slice(x);
        let (mel1, f0_now, _voi) = self.front.step(&self.in_ring, &self.fb);

        // energy: [-1,1] スケールの HOP512 非重畳 RMS。mel フレーム t は
        // 「窓が閉じた最新の RMS フレーム」を見る (resample_to と同一・頭は無音規約)
        for &v in x {
            let s = (v / 32768.0) as f64;
            self.rms_acc += s * s;
            self.rms_n += 1;
            if self.rms_n == 512 {
                self.last_rms = ((self.rms_acc / 512.0) + 1e-12).sqrt() as f32;
                self.rms_acc = 0.0;
                self.rms_n = 0;
            }
        }

        // E: content 768
        let content = self.e.step(&mut self.es, &mel1);

        // G 入力 [770]
        let f0s = if f0_now > 50.0 { f0_now * self.ratio } else { f0_now };
        let mut gin = content;
        gin.push((f0s.max(50.0) / 200.0).ln());
        gin.push(self.last_rms.max(1e-4).ln());
        let mut mel_g = self.g.step(&mut self.gs, &gin);
        for v in mel_g.iter_mut() {
            *v -= V_MEL_ADAPT;
        }

        // 励起 f0: シフト済み値の走行 fill (Python _fill(f0s) と同一)
        if f0s > 50.0 {
            self.last_voiced = Some(f0s.max(50.0));
        }
        let f0f = self.last_voiced.unwrap_or(sf::PRE_VOICED_HZ);

        let mut y = self.v.synth_block(&mel_g, f0f, noise);
        for v in y.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        y
    }

    pub fn dbg_exc(&self) -> &[f32] {
        &self.v.dbg_exc_blk
    }

    pub fn reset(&mut self) {
        self.es.reset();
        self.gs.reset();
        self.v.reset();
        self.front = sf::FrontState::new();
        self.in_ring.iter_mut().for_each(|v| *v = 0.0);
        self.last_voiced = None;
        self.rms_acc = 0.0;
        self.rms_n = 0;
        self.last_rms = 0.0;
    }

    pub fn dbg_phi(&self) -> &[f32] {
        &self.v.dbg_phi_blk
    }

    pub fn dbg_fu(&self) -> &[f32] {
        &self.v.dbg_f_blk
    }

    pub fn dbg_f0f(&self) -> f32 {
        self.last_voiced.unwrap_or(sf::PRE_VOICED_HZ)
    }
}

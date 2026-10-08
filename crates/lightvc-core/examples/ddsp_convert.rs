//! DDSP-VC の wav 変換（Rust 製品経路のストリーミング推論をそのまま使う・240 サンプル単位）。
//!
//! cargo run --release -p lightvc-core --example ddsp_convert -- \
//!     --model results/ddsp_vc/export --ref target.wav --src-enroll me.wav --in in.wav --out out.wav
//!
//! 入出力は 48kHz。f0 レジスタの定数（log f0 の中央値・標準偏差）は目標の参照と元話者の登録音声から因果 YIN で求める
//! （変換中の発話の統計は使わない）。--src-enroll を省くと男声の既定値（中央 120Hz・σ 0.15）。

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use lightvc_core::ddsp_vc::{map_register, register_from_pcm as register, CausalYin, DdspVc, HOP, SR};

fn read_wav(p: &PathBuf) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("read {p:?}"))?;
    if &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        bail!("not a WAV: {p:?}");
    }
    let (mut fmt, mut ch, mut sr, mut bits) = (0u16, 0u16, 0u32, 0u16);
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let body = &b[i + 8..(i + 8 + sz).min(b.len())];
        if id == b"fmt " {
            fmt = u16::from_le_bytes([body[0], body[1]]);
            ch = u16::from_le_bytes([body[2], body[3]]);
            sr = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            if sr as usize != SR {
                bail!("{p:?} は {sr}Hz(48kHz が必要)");
            }
            let n = ch as usize;
            let frames: Vec<f32> = match (fmt, bits) {
                (1, 16) => body.chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0).collect(),
                (3, 32) | (0xFFFE, 32) => body.chunks_exact(4).map(|s| f32::from_le_bytes([s[0], s[1], s[2], s[3]])).collect(),
                _ => bail!("unsupported wav format {fmt}/{bits}bit"),
            };
            return Ok(frames.chunks(n).map(|c| c.iter().sum::<f32>() / n as f32).collect());
        }
        i += 8 + sz + (sz & 1);
    }
    bail!("no data chunk: {p:?}")
}

fn write_wav(p: &PathBuf, x: &[f32]) -> Result<()> {
    let mut b = Vec::with_capacity(44 + 4 * x.len());
    let data = (4 * x.len()) as u32;
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&(SR as u32).to_le_bytes());
    b.extend_from_slice(&(SR as u32 * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for v in x {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(p, b)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(PathBuf::from);
    let model = get("--model").context("--model")?;
    let refp = get("--ref").context("--ref")?;
    let inp = get("--in").context("--in")?;
    let outp = get("--out").context("--out")?;
    let x = read_wav(&inp)?;
    let rf = read_wav(&refp)?;
    let (mu_t, sd_t) = register(&rf);
    let (mu_s, sd_s) = match get("--src-enroll") {
        Some(e) => register(&read_wav(&e)?),
        None => ((120f64).ln() as f32, 0.15),
    };
    let mut vc = DdspVc::load(&model)?;
    vc.set_target_ref(&rf[..rf.len().min(3 * SR)]);
    let mut yin = CausalYin::new(0.45);
    let blocks = x.len() / HOP;
    let mut y = vec![0f32; blocks * HOP];
    let mut times = Vec::with_capacity(blocks);
    for i in 0..blocks {
        let blk = &x[i * HOP..(i + 1) * HOP];
        let t0 = Instant::now();
        let f0 = map_register(yin.push(blk), mu_s, sd_s, mu_t, sd_t);
        vc.process_block(blk, f0, &mut y[i * HOP..(i + 1) * HOP]);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let peak = y.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > 0.95 {
        y.iter_mut().for_each(|v| *v *= 0.95 / peak);
    }
    write_wav(&outp, &y)?;
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| times[((times.len() as f64 - 1.0) * q) as usize];
    eprintln!(
        "{} blocks | f0 register src {:.1}Hz σ{:.2} → tgt {:.1}Hz σ{:.2} | per-block ms p50 {:.2} p95 {:.2} p99 {:.2} (予算 {:.1} ms) | algorithmic latency {} ms",
        blocks, mu_s.exp(), sd_s, mu_t.exp(), sd_t, p(0.5), p(0.95), p(0.99), 1000.0 * HOP as f64 / SR as f64,
        1000 * lightvc_core::ddsp_vc::DELAY / SR
    );
    Ok(())
}

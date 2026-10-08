//! nvoc のコピー合成（Rust 製品経路のストリーミング推論・240 サンプル単位・f0 は因果 YIN）。
//!
//! cargo run --release -p lightvc-core --example nvoc_resynth -- --model results/nvoc1/export --in in.wav --out out.wav
//!
//! 入出力は 48kHz。出力は DELAY(240 サンプル)を詰めて入力と時刻を揃えて書く（試聴用）。1 ブロックの処理時間は YIN 込み
//! （調波源なしの重みでは YIN を回さない）。

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use lightvc_core::ddsp_vc::CausalYin;
use lightvc_core::nvoc::{NVoc, DELAY, HOP, SR};

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
    let inp = get("--in").context("--in")?;
    let outp = get("--out").context("--out")?;
    let x = read_wav(&inp)?;
    let mut voc = NVoc::load(&model)?;
    let mut yin = CausalYin::new(0.45);
    let blocks = x.len() / HOP + 1;
    let mut xp = x.clone();
    xp.resize(blocks * HOP, 0.0);
    let mut y = vec![0f32; blocks * HOP];
    let mut times = Vec::with_capacity(blocks);
    for i in 0..blocks {
        let blk = &xp[i * HOP..(i + 1) * HOP];
        let t0 = Instant::now();
        let f0 = if voc.needs_f0() { yin.push(blk) } else { 0.0 };
        voc.process_block(blk, f0, &mut y[i * HOP..(i + 1) * HOP]);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let mut out: Vec<f32> = y[DELAY..].iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    out.truncate(x.len());
    write_wav(&outp, &out)?;
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| times[((times.len() as f64 - 1.0) * q) as usize];
    eprintln!("{} blocks | per-block ms p50 {:.2} p95 {:.2} p99 {:.2} (予算 {:.1} ms) | algorithmic latency {} ms",
              blocks, p(0.5), p(0.95), p(0.99), 1000.0 * HOP as f64 / SR as f64, 1000 * (HOP + DELAY) / SR);
    Ok(())
}

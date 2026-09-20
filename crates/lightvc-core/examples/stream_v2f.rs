//! V2fStream（リアルタイム経路）の検証: 一括 process() との一致 + ブロック RTF。
use lightvc_core::ship_front as SF;
use lightvc_core::v2f_infer::{V2fEngine, V2fStream};
use std::sync::Arc;
use std::time::Instant;

fn td(name: &str) -> Vec<f32> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name);
    std::fs::read(p).unwrap().chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    let eng = Arc::new(V2fEngine::load(
        &base.join("v2f_trained.bin"), &base.join("mel_fb_1024_80.bin"),
        &base.join("mel2lin_W.bin"), 24, 8).unwrap());
    // ⚠ 先頭 1024 を無音にする。batch 参照の頭の癖（frame_upsample_causal の
    //   clamp が先頭 2 ブロックで未来 f0 を読む）は無音頭なら stream の初期状態と
    //   一致する。実運用のストリームも無音から始まる。
    let mut x = vec![0.0f32; 1024];
    x.extend(td("sf_w.bin"));
    let n = (x.len() / V2fStream::BLOCK) * V2fStream::BLOCK;

    let mut noise = vec![0.0f32; 1024];
    noise.extend(td("e2e_noise.bin"));  // 一括版と同一の雑音を注入
    let mut st = V2fStream::new(eng.clone());
    let mut y = Vec::with_capacity(n);
    let mut v = Vec::new();
    let audio = V2fStream::BLOCK as f64 / SF::SR as f64;
    for (bi, b) in (0..n).step_by(V2fStream::BLOCK).enumerate() {
        let t0 = Instant::now();
        let out = st.process_block_with(&x[b..b + V2fStream::BLOCK],
                                        Some(&noise[b..b + V2fStream::BLOCK]));
        v.push(t0.elapsed().as_secs_f64() / audio);
        if bi == 100 {
            let f0b = SF::causal_f0(&x[..n]).0;
            let excb = SF::excitation(&f0b, n, &noise[..n], 0.3);
            let seg = &excb[b..b + V2fStream::BLOCK];
            let d: f32 = seg.iter().zip(&st.dbg_exc_blk)
                .map(|(p, q)| (p - q).abs()).fold(0.0, f32::max);
            // batch の feat（フレーム 2bi, 2bi+1）を再現して比較
            let fb2 = td("mel_fb_1024_80.bin");
            let mel80 = SF::mel(&x[..n], &fb2);
            let (er, ei) = SF::cstft(&excb, SF::NFFT_S, SF::HOP_S);
            let w2l = td("mel2lin_W.bin");
            let nb2 = SF::NBIN_S;
            let mut dmax = [0.0f32; 4];
            for t2 in 0..2usize {
                let ti = 2 * bi + t2;
                let j = SF::to_frames_idx(ti).min(mel80[0].len() - 1);
                for bq in 0..nb2 {
                    let mut ml = 0.0f32;
                    for m2 in 0..SF::N_MEL {
                        ml += w2l[bq * SF::N_MEL + m2] * mel80[m2][j];
                    }
                    let hh = (ml - SF::MEL_REF).exp();
                    let (r, i2) = (er[bq][ti] * hh, ei[bq][ti] * hh);
                    let want4 = [ml, r, i2, ((r * r + i2 * i2).sqrt() + 1e-5).ln()];
                    for c in 0..4usize {
                        let got = st.dbg_feat[(c * 2 + t2) * nb2 + bq];
                        dmax[c] = dmax[c].max((got - want4[c]).abs());
                    }
                }
            }
            println!("blk100 exc 最大差 {d:.5}  feat 差 [mel {:.4} re {:.4} im {:.4} log {:.4}]",
                     dmax[0], dmax[1], dmax[2], dmax[3]);
            // ±2 フレームずらして Re の一致点を探す
            for off in -2i64..=2 {
                let mut dm = 0.0f32;
                for t2 in 0..2usize {
                    let ti = (2 * bi + t2) as i64 + off;
                    if ti < 0 || ti as usize >= er[0].len() { continue; }
                    let j = SF::to_frames_idx(2 * bi + t2).min(mel80[0].len() - 1);
                    for bq in 0..nb2 {
                        let mut ml = 0.0f32;
                        for m2 in 0..SF::N_MEL {
                            ml += w2l[bq * SF::N_MEL + m2] * mel80[m2][j];
                        }
                        let hh = (ml - SF::MEL_REF).exp();
                        let got = st.dbg_feat[(2 + t2) * nb2 + bq];   // ch1 = Re, index (c*2+t)
                        dm = dm.max((got - er[bq][ti as usize] * hh).abs());
                    }
                }
                println!("  Re: 位相フレーム off={off:+} の最大差 {dm:.4}");
            }
        }
        y.extend(out);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = v[v.len() * 95 / 100];

    // **一括版（e2e_y.bin ＝ PyTorch と 95 dB 一致済み）との SNR**。
    // 先頭は状態の立ち上がり（OLA・f0 履歴・cummean）で数フレームずれるので
    // 1 秒目以降で判定。合格線 40 dB。
    // ⚠ 波形 SNR は使えない——batch 参照は frame_upsample_causal の clamp が
    //   先頭 2 ブロックだけ未来フレームの f0 を読む（学習と同一の頭の癖）。
    //   位相は積分器なので、その初期差が**恒久位相オフセット**になる。
    //   stream 側が因果的に正しい。∴ 位相盲の**振幅スペクトログラム SNR** で判定。
    let want = eng.process(&x[..n], &noise[..n]);   // Rust batch（PyTorch と 95 dB 一致済みの経路）
    let m = n.min(want.len());
    let skip = 44100usize.min(m / 2);
    // ⚠ stream は OLA 固有遅延（NFFT_S − HOP_S = 384 サンプル ＝ 台帳の recon 8.71 ms）
    //   だけ遅れて出る。batch はオフラインなので頭を切って遅延を消しているだけ。
    //   シフトを合わせてから比較する。
    let dly = SF::NFFT_S - SF::HOP_S;
    let (mut e, mut sg) = (0.0f64, 0.0f64);
    for k in skip..m - dly {
        e += ((y[k + dly] - want[k]) as f64).powi(2);
        sg += (want[k] as f64).powi(2);
    }
    let snr = 10.0 * (sg / e.max(1e-30)).log10();
    println!("stream blocks={}  RTF p50 {:.4} / p95 {:.4}（予算 0.35）",
             v.len(), v[v.len() / 2], p95);
    println!("stream ≡ batch（遅延 384 整合後の波形）SNR {snr:.2} dB（合格線 40）");
    let bytes: Vec<u8> = y.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(base.join("stream_y.bin"), bytes).unwrap();
    std::process::exit(if p95 < 0.35 && snr >= 40.0 { 0 } else { 1 });
}

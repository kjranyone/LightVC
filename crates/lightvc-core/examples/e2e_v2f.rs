//! End-to-end: 実音声 1 秒 → 出荷 front → v2f 幹（学習済み重み）→ cistft。
//! Python 参照（testdata/e2e_y.bin）との SNR を出す。**納品物の最終同一性検査。**
use lightvc_core::ship_front as SF;
use lightvc_core::simd::*;

fn td(name: &str) -> Vec<f32> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name);
    std::fs::read(p).unwrap().chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let (ch, nl, nb) = (24usize, 8usize, 257usize);
    let w = td("sf_w.bin");
    let n = w.len();
    let fb = td("mel_fb_1024_80.bin");
    let w2l = td("mel2lin_W.bin");            // [257][80]
    let noise = td("e2e_noise.bin");
    let mut wt = V2fWeights::from_flat(&td("v2f_trained.bin"), 4, ch, nl, 7, 3);
    for l in wt.layers.iter_mut() {
        l.norm_mode = 1;                       // cummean（出荷モデルの正規化）
    }

    // 解析: mel(80) と f0
    let mel80 = SF::mel(&w, &fb);              // [80][Ta]
    let (f0, _) = SF::causal_f0(&w);
    let t_syn = SF::n_frames(n, SF::HOP_S, SF::NFFT_S);
    // 合成格子へ写像 + 線形軸 257 へ
    let mut mel_lin = vec![vec![0.0f32; t_syn]; nb];
    for t in 0..t_syn {
        let j = SF::to_frames_idx(t).min(mel80[0].len() - 1);
        for b in 0..nb {
            let mut acc = 0.0f32;
            for m in 0..80 {
                acc += w2l[b * 80 + m] * mel80[m][j];
            }
            mel_lin[b][t] = acc;
        }
    }
    // 調波 prior
    let exc = SF::excitation(&f0, n, &noise, 0.3);
    let (er, ei) = SF::cstft(&exc, SF::NFFT_S, SF::HOP_S);
    let t = t_syn.min(er[0].len());
    // 特徴 [4][t][257] と P
    let mut feat = Ten::zeros(4, t, nb);
    let (mut pre, mut pim) = (vec![vec![0.0f32; t]; nb], vec![vec![0.0f32; t]; nb]);
    for b in 0..nb {
        for ti in 0..t {
            let h = (mel_lin[b][ti] - SF::MEL_REF).exp();
            let (r, i) = (er[b][ti] * h, ei[b][ti] * h);
            pre[b][ti] = r;
            pim[b][ti] = i;
            feat.at_mut(0, ti)[b] = mel_lin[b][ti];
            feat.at_mut(1, ti)[b] = r;
            feat.at_mut(2, ti)[b] = i;
            feat.at_mut(3, ti)[b] = ((r * r + i * i).sqrt() + 1e-5).ln();
        }
    }
    // 幹（**streaming**。ゼロ初期化キャッシュが Python の層ごと左ゼロ詰めと等価。
    //   一括チェーンは文脈を消費して T が縮むので Python と形が合わない）
    let emit = 2usize;
    let mut st = TrunkState::with_emit(&wt.layers, nb, emit);
    let mut cur = Ten::zeros(ch, t, nb);
    {
        let h = pointwise(&feat, &wt.inp_w, &wt.inp_b, ch);
        let mut b0 = 0usize;
        while b0 + emit <= t {
            let mut xb = Ten::zeros(ch, emit, nb);
            for c in 0..ch {
                for ti in 0..emit {
                    xb.at_mut(c, ti).copy_from_slice(h.at(c, b0 + ti));
                }
            }
            let y = st.step(&wt.layers, &xb);
            for c in 0..ch {
                for ti in 0..emit {
                    let src = y.at(c, ti).to_vec();
                    cur.at_mut(c, b0 + ti).copy_from_slice(&src);
                }
            }
            b0 += emit;
        }
    }
    let o = pointwise(&cur, &wt.out_w, &wt.out_b, 2);
    let (mut sr, mut si) = (vec![vec![0.0f32; t]; nb], vec![vec![0.0f32; t]; nb]);
    for b in 0..nb {
        for ti in 0..t {
            sr[b][ti] = o.at(0, ti)[b];
            si[b][ti] = o.at(1, ti)[b];
        }
    }
    let y = SF::cistft(&sr, &si, n, SF::NFFT_S, SF::HOP_S);
    let want = td("e2e_y.bin");
    let m = n - SF::NFFT_S;                    // 末尾の不良条件領域は除外
    let (mut e, mut sg) = (0.0f64, 0.0f64);
    for k in 0..m {
        e += ((y[k] - want[k]) as f64).powi(2);
        sg += (want[k] as f64).powi(2);
    }
    let snr = 10.0 * (sg / e.max(1e-30)).log10();
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/e2e_rust_y.bin");
    let bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, bytes).unwrap();
    println!("e2e torch ≡ rust  SNR {snr:.2} dB（合格線 40）");
    std::process::exit(if snr >= 40.0 { 0 } else { 1 });
}

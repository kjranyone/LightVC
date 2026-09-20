//! VcStream (Rust フル変換チェーン) vs Python batch 変換の parity と RTF。
//!
//!   cargo run --release -p lightvc-core --example vc_parity -- <scratch_dir>

use lightvc_core::eg::{Eg1d, EgStream};
use lightvc_core::ship_front as sfr;
use lightvc_core::v2f_infer::{V2fEngine, V2fStream};
use lightvc_core::vc_stream::VcStream;
use std::sync::Arc;
use std::time::Instant;

fn rd(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"))
        .chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("scratch dir");
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let base = root.join("crates/lightvc-core/testdata");

    let e = Arc::new(Eg1d::from_flat(&rd(&format!("{dir}/e1_test.bin")), 80, 768, 256, 8));
    let g = Arc::new(Eg1d::from_flat(&rd(&format!("{dir}/g1_test.bin")), 770, 80, 256, 6));
    let eng = Arc::new(V2fEngine::load(
        &root.join("models/v2f.bin"), &base.join("mel_fb_1024_80.bin"),
        &base.join("mel2lin_W.bin"), 24, 8).unwrap());
    let fb = rd(base.join("mel_fb_1024_80.bin").to_str().unwrap());

    let x = rd(&format!("{dir}/vcp_x.bin"));
    let z = rd(&format!("{dir}/vcp_z.bin"));
    let y_ref = rd(&format!("{dir}/vcp_y.bin"));
    let n = (x.len() / VcStream::BLOCK) * VcStream::BLOCK;

    // 中間信号の突き合わせ: front→E→G の各段を独立に再走して Python ダンプと比較
    {
        let fb2 = rd(base.join("mel_fb_1024_80.bin").to_str().unwrap());
        let mut front = sfr::FrontState::new();
        let mut ring = vec![0.0f32; sfr::NFFT_A];
        let mut es = EgStream::new(&e);
        let mut gs = EgStream::new(&g);
        let ratio = (2.0f64.powf(15.0 / 12.0)) as f32;
        let (mut lv, mut rms_acc, mut rms_n, mut last_rms) = (0.0f32, 0f64, 0usize, 0f32);
        let _ = lv;
        let mut mels = Vec::new();
        let mut lf0s = Vec::new();
        let mut ens = Vec::new();
        let mut cs = Vec::new();
        let mut mgs = Vec::new();
        for b in 0..n / 256 {
            let xx = &x[b * 256..(b + 1) * 256];
            ring.copy_within(256.., 0);
            let l0 = sfr::NFFT_A - 256;
            ring[l0..].copy_from_slice(xx);
            let (mel1, f0_now, _) = front.step(&ring, &fb2);
            for &v in xx {
                let sv = (v / 32768.0) as f64;
                rms_acc += sv * sv;
                rms_n += 1;
                if rms_n == 512 {
                    last_rms = ((rms_acc / 512.0) + 1e-12).sqrt() as f32;
                    rms_acc = 0.0;
                    rms_n = 0;
                }
            }
            let content = es.step(&e, &mel1);
            let f0s = if f0_now > 50.0 { f0_now * ratio } else { f0_now };
            let lf0 = (f0s.max(50.0) / 200.0).ln();
            let en = last_rms.max(1e-4).ln();
            let mut gin = content.clone();
            gin.push(lf0);
            gin.push(en);
            let mg = gs.step(&g, &gin);
            mels.push(mel1);
            lf0s.push(lf0);
            ens.push(en);
            cs.push(content);
            mgs.push(mg);
            let _ = b;
        }
        let cmp_t = |name: &str, ours: &[Vec<f32>], py: &[f32], c: usize| {
            let t = ours.len().min(py.len() / c);
            let mut mx = 0f32;
            let mut mxi = 0usize;
            for ti in 0..t {
                for ci in 0..c {
                    let d = (ours[ti][ci] - py[ci * t + ti]).abs();
                    if d > mx {
                        mx = d;
                        mxi = ti;
                    }
                }
            }
            println!("  {name}: max|diff| {mx:.5} @frame {mxi} / {t}");
        };
        let cmp_1 = |name: &str, ours: &[f32], py: &[f32]| {
            let t = ours.len().min(py.len());
            let mut mx = 0f32;
            let mut mxi = 0usize;
            for ti in 0..t {
                let d = (ours[ti] - py[ti]).abs();
                if d > mx {
                    mx = d;
                    mxi = ti;
                }
            }
            println!("  {name}: max|diff| {mx:.5} @frame {mxi} / {t}");
        };
        cmp_t("mel  ", &mels, &rd(&format!("{dir}/vcp_mel.bin")), 80);
        cmp_1("lf0  ", &lf0s, &rd(&format!("{dir}/vcp_lf0.bin")));
        cmp_1("en   ", &ens, &rd(&format!("{dir}/vcp_en.bin")));
        cmp_t("cont ", &cs, &rd(&format!("{dir}/vcp_content.bin")), 768);
        cmp_t("mel_g", &mgs, &rd(&format!("{dir}/vcp_melg.bin")), 80);
    }

    let v = V2fStream::with_threads(eng, 2);
    let mut vc = VcStream::new(e, g, v, fb, 15.0);
    let mut y = Vec::with_capacity(n);
    let mut exc_all = Vec::with_capacity(n);
    let mut f0f_all: Vec<f32> = Vec::new();
    let mut phi_all: Vec<f32> = Vec::new();
    let mut fu_all: Vec<f32> = Vec::new();
    let mut times = Vec::new();
    for b in 0..n / VcStream::BLOCK {
        let s = b * VcStream::BLOCK;
        let t0 = Instant::now();
        y.extend(vc.process_block_with(&x[s..s + VcStream::BLOCK],
                                       Some(&z[s..s + VcStream::BLOCK])));
        times.push(t0.elapsed().as_secs_f64());
        exc_all.extend_from_slice(vc.dbg_exc());
        phi_all.extend_from_slice(vc.dbg_phi());
        fu_all.extend_from_slice(vc.dbg_fu());
        f0f_all.push(vc.dbg_f0f());
    }
    {
        let (pp, pf) = (rd(&format!("{dir}/vcp_phi.bin")), rd(&format!("{dir}/vcp_f0u.bin")));
        let cmp = |name: &str, a: &[f32], b: &[f32]| {
            let mm = a.len().min(b.len());
            let mut mx = 0f32;
            let mut mxi = 0usize;
            for i in 0..mm {
                let mut d2 = (a[i] - b[i]).abs();
                if name == "phi " && d2 > 3.14 {
                    d2 = (6.283_185_5 - d2).abs();
                }
                if d2 > mx {
                    mx = d2;
                    mxi = i;
                }
            }
            println!("  {name}: max|diff| {mx:.6} @{mxi}");
        };
        cmp("f0u ", &pf, &fu_all);
        cmp("phi ", &pp, &phi_all);
    }
    {
        let pf = rd(&format!("{dir}/vcp_fill.bin"));
        let mm = pf.len().min(f0f_all.len());
        let mut mx = 0f32;
        let mut mxi = 0usize;
        let mut nbad = 0usize;
        for i in 0..mm {
            let d2 = (pf[i] - f0f_all[i]).abs();
            if d2 > 1e-3 {
                nbad += 1;
            }
            if d2 > mx {
                mx = d2;
                mxi = i;
            }
        }
        println!("  fill: max|diff| {mx:.3} Hz @frame {mxi}  |diff|>1e-3 が {nbad}/{mm} frames");
        if mx > 0.0 {
            let lo = mxi.saturating_sub(2);
            for i in lo..(mxi + 3).min(mm) {
                println!("    frame {i}: py {:.3}  rust {:.3}", pf[i], f0f_all[i]);
            }
        }
    }
    {
        let pe = rd(&format!("{dir}/vcp_exc.bin"));
        let m2 = pe.len().min(exc_all.len());
        let mut mx = 0f32;
        let mut mxi = 0usize;
        for i in 0..m2 {
            let d2 = (pe[i] - exc_all[i]).abs();
            if d2 > mx {
                mx = d2;
                mxi = i;
            }
        }
        let (mut se2, mut sr2) = (0f64, 0f64);
        for i in 0..m2 {
            se2 += (pe[i] as f64 - exc_all[i] as f64).powi(2);
            sr2 += (pe[i] as f64).powi(2);
        }
        println!("  exc : max|diff| {mx:.5} @{mxi}  SNR {:.1} dB",
                 10.0 * (sr2 / se2.max(1e-30)).log10());
        // (a) 式の差: Python の f0s 系列から Rust batch excitation を計算して比較
        let f0s_py = rd(&format!("{dir}/vcp_f0s.bin"));
        let zz = rd(&format!("{dir}/vcp_z.bin"));
        let eb = sfr::excitation(&f0s_py, m2, &zz, 0.3);
        let snr_of = |a: &[f32], b: &[f32]| {
            let mm = a.len().min(b.len());
            let (mut se3, mut sr3) = (0f64, 0f64);
            for i in 0..mm {
                se3 += (a[i] as f64 - b[i] as f64).powi(2);
                sr3 += (a[i] as f64).powi(2);
            }
            10.0 * (sr3 / se3.max(1e-30)).log10()
        };
        println!("  exc 式差 (py明示和 vs rust閉形式, 同一f0): {:.1} dB",
                 snr_of(&pe, &eb));
        println!("  exc 状態差 (rust batch vs stream): {:.1} dB", snr_of(&eb, &exc_all));
    }
    // V 段の単独切り分け: Python の特徴量ダンプを Rust V (batch trunk) に食わせる
    {
        use lightvc_core::simd::*;
        let feat_py = rd(&format!("{dir}/vcp_feat.bin"));
        let nb = 257usize;
        let tt = feat_py.len() / (4 * nb);
        let mut feat = Ten::zeros(4, tt, nb);
        for c in 0..4 {
            for ti in 0..tt {
                for b in 0..nb {
                    feat.at_mut(c, ti)[b] = feat_py[(c * nb + b) * tt + ti];
                }
            }
        }
        let engv = lightvc_core::v2f_infer::V2fEngine::load(
            &root.join("models/v2f.bin"), &base.join("mel_fb_1024_80.bin"),
            &base.join("mel2lin_W.bin"), 24, 8).unwrap();
        let h = pointwise(&feat, &engv.w.inp_w, &engv.w.inp_b, 24);
        let mut st = TrunkState::with_emit(&engv.w.layers, nb, 2);
        let mut cur = Ten::zeros(24, tt, nb);
        for ti in (0..tt / 2 * 2).step_by(2) {
            let mut blkh = Ten::zeros(24, 2, nb);
            for c in 0..24 {
                for k in 0..2 {
                    blkh.at_mut(c, k).copy_from_slice(h.at(c, ti + k));
                }
            }
            let o2 = st.step(&engv.w.layers, &blkh);
            for c in 0..24 {
                for k in 0..2 {
                    cur.at_mut(c, ti + k).copy_from_slice(o2.at(c, k));
                }
            }
        }
        let o = pointwise(&cur, &engv.w.out_w, &engv.w.out_b, 2);
        // cistft 相当は省き、複素スペクトル S 自体を Python 出力 S と比べたいが
        // ダンプが波形なので、ここでは cistft を通して波形比較する
        let mut re = vec![vec![0f32; tt]; nb];
        let mut im = vec![vec![0f32; tt]; nb];
        for b in 0..nb {
            for ti in 0..tt {
                re[b][ti] = o.at(0, ti)[b];
                im[b][ti] = o.at(1, ti)[b];
            }
        }
        let yb = lightvc_core::ship_front::cistft(&re, &im, n, 512, 128);
        let (mut se, mut sr) = (0f64, 0f64);
        for i in 2048..n - 2048 {
            se += (y_ref[i] as f64 - yb[i] as f64).powi(2);
            sr += (y_ref[i] as f64).powi(2);
        }
        println!("  V単独 (py特徴→rust V batch) vs py出力: SNR {:.2} dB",
                 10.0 * (sr / se.max(1e-30)).log10());
    }

    // 固有遅延 384 (OLA) を整合して SNR — 掃引で真の遅延も確認
    for dd in [128usize, 256, 320, 384, 448, 512, 640] {
        let (mut se, mut sr) = (0f64, 0f64);
        let mm = n - dd;
        for i in 2048..mm {
            let (a, b) = (y_ref[i] as f64, y[i + dd] as f64);
            se += (a - b) * (a - b);
            sr += a * a;
        }
        println!("  delay {dd}: SNR {:.2} dB", 10.0 * (sr / se.max(1e-30)).log10());
    }
    let d = 384usize;
    let (mut se, mut sr) = (0f64, 0f64);
    let m = n - d;
    // 立ち上がり (無音頭 1024) を除いた区間で比較
    for i in 2048..m {
        let (a, b) = (y_ref[i] as f64, y[i + d] as f64);
        se += (a - b) * (a - b);
        sr += a * a;
    }
    let snr = 10.0 * (sr / se.max(1e-30)).log10();
    // 1 秒ごとの SNR プロファイル (発散の局在を見る)
    for sec in 0..(m / 44100) {
        let (mut e2, mut r2) = (0f64, 0f64);
        for i in sec * 44100..((sec + 1) * 44100).min(m) {
            let (a, b) = (y_ref[i] as f64, y[i + d] as f64);
            e2 += (a - b) * (a - b);
            r2 += a * a;
        }
        println!("  sec {sec}: SNR {:.1} dB  ref_rms {:.4}", 10.0 * (r2 / e2.max(1e-30)).log10(),
                 (r2 / 44100.0).sqrt());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let blk_s = VcStream::BLOCK as f64 / 44100.0;
    println!("stream ≡ batch(Python) SNR {:.2} dB（合格線 40）", snr);
    println!("full-chain RTF p50 {:.4} / p95 {:.4}",
             times[times.len() / 2] / blk_s, times[times.len() * 95 / 100] / blk_s);
}

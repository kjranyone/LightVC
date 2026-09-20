//! E1/G1 の Rust 実装 vs PyTorch の parity(batch/stream)と RTF。
//!
//!   cargo run --release -p lightvc-core --example eg_parity -- <scratch_dir>

use lightvc_core::eg::{Eg1d, Eg1dH, EgPool, EgStream};

fn rd(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn snr(a: &[f32], b: &[f32]) -> f64 {
    let mut se = 0f64;
    let mut sr = 0f64;
    for (x, y) in a.iter().zip(b) {
        se += (*x as f64 - *y as f64).powi(2);
        sr += (*x as f64).powi(2);
    }
    10.0 * (sr / se.max(1e-30)).log10()
}

fn main() {
    let dir = std::env::args().nth(1).expect("scratch dir");
    let t = 400usize;

    let e = Eg1d::from_flat(&rd(&format!("{dir}/e1_test.bin")), 80, 768, 256, 8);
    let g = Eg1d::from_flat(&rd(&format!("{dir}/g1_test.bin")), 770, 80, 256, 6);

    // [C][T] (Python dump) -> [T][C]
    let tr = |v: &[f32], c: usize| -> Vec<f32> {
        let mut o = vec![0f32; v.len()];
        for i in 0..c {
            for ti in 0..t {
                o[ti * c + i] = v[i * t + ti];
            }
        }
        o
    };
    let mel = tr(&rd(&format!("{dir}/par_mel.bin")), 80);
    let e_ref = tr(&rd(&format!("{dir}/par_e_out.bin")), 768);
    let g_in = tr(&rd(&format!("{dir}/par_g_in.bin")), 770);
    let g_ref = tr(&rd(&format!("{dir}/par_g_out.bin")), 80);

    let e_out = e.process(&mel, t);
    let g_out = g.process(&g_in, t);
    println!("batch parity: E {:.1} dB  G {:.1} dB", snr(&e_ref, &e_out), snr(&g_ref, &g_out));

    let mut es = EgStream::new(&e);
    let mut so = Vec::with_capacity(t * 768);
    for ti in 0..t {
        so.extend(es.step(&e, &mel[ti * 80..(ti + 1) * 80]));
    }
    let bit = so.iter().zip(&e_out).filter(|(a, b)| a != b).count();
    println!("stream vs batch: E maxdiff {} 個 / SNR {:.1} dB", bit, snr(&e_out, &so));

    // RTF: 10 秒相当 (172fps -> 1723 フレーム) をストリームで
    let n = 1723usize;
    let mut gs = EgStream::new(&g);
    let mut es2 = EgStream::new(&e);
    let frame_e = vec![0.1f32; 80];
    let frame_g = vec![0.1f32; 770];
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(es2.step(&e, &frame_e));
    }
    let te = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(gs.step(&g, &frame_g));
    }
    let tg = t1.elapsed().as_secs_f64();
    println!("stream RTF 1t: E {:.4} (予算0.10)  G {:.4} (予算0.05)", te / 10.0, tg / 10.0);

    // 2 スレッド (出力分割・ビット一致)
    let pool = EgPool::new();
    let mut es3 = EgStream::new(&e);
    let mut so2 = Vec::with_capacity(t * 768);
    for ti in 0..t {
        so2.extend(es3.step_par(&e, &mel[ti * 80..(ti + 1) * 80], Some(&pool)));
    }
    let bit2 = so2.iter().zip(&so).filter(|(a, b)| a != b).count();
    let mut es4 = EgStream::new(&e);
    let mut gs4 = EgStream::new(&g);
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(es4.step_par(&e, &frame_e, Some(&pool)));
    }
    let te2 = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(gs4.step_par(&g, &frame_g, Some(&pool)));
    }
    let tg2 = t1.elapsed().as_secs_f64();
    println!("stream RTF 2t: E {:.4}  G {:.4}  (1t とのビット差 {} 個)",
             te2 / 10.0, tg2 / 10.0, bit2);

    // f16 重み: parity (vs f32 stream) と RTF
    let eh = Eg1dH::from_net(&e);
    let gh = Eg1dH::from_net(&g);
    let mut es5 = EgStream::new_h(&eh);
    let mut so3 = Vec::with_capacity(t * 768);
    for ti in 0..t {
        so3.extend(es5.step_h(&eh, &mel[ti * 80..(ti + 1) * 80]));
    }
    println!("f16 stream vs f32 stream: E SNR {:.1} dB", snr(&so, &so3));
    let mut es6 = EgStream::new_h(&eh);
    let mut gs6 = EgStream::new_h(&gh);
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(es6.step_h(&eh, &frame_e));
    }
    let te3 = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(gs6.step_h(&gh, &frame_g));
    }
    let tg3 = t1.elapsed().as_secs_f64();
    println!("stream RTF f16 1t: E {:.4} (予算0.10)  G {:.4} (予算0.05)", te3 / 10.0, tg3 / 10.0);
}

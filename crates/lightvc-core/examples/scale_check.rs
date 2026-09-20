//! CLI の入力スケール ([-1,1] vs x32768) が v2f resynth 品質に与える影響を実証。
use lightvc_core::v2f_infer::V2fEngine;

fn rd(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn snr(a: &[f32], b: &[f32]) -> f64 {
    let (mut se, mut sr) = (0f64, 0f64);
    let m = a.len().min(b.len());
    for i in 8192..m {
        se += (a[i] as f64 - b[i] as f64).powi(2);
        sr += (a[i] as f64).powi(2);
    }
    10.0 * (sr / se.max(1e-30)).log10()
}

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let base = root.join("crates/lightvc-core/testdata");
    let eng = V2fEngine::load(&root.join("models/v2f.bin"), &base.join("mel_fb_1024_80.bin"),
                              &base.join("mel2lin_W.bin"), 24, 8).unwrap();
    let x = rd(&base.join("sf_w.bin")); // x32768 スケールの実音声ダンプ
    let mut st = 0x9e3779b97f4a7c15u64;
    let noise: Vec<f32> = (0..x.len()).map(|_| {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        ((st >> 40) as f32 / 8388608.0) - 1.0
    }).collect();
    let y_ok = eng.process(&x, &noise);
    let x_small: Vec<f32> = x.iter().map(|v| v / 32768.0).collect();
    let y_bad: Vec<f32> = eng.process(&x_small, &noise).iter().map(|v| v * 32768.0).collect();
    let _ = snr(&x, &y_ok);
    // mel 域 L1 (log-mel): 位相非保存の vocoder に波形 SNR は無意味
    let fb = rd(&base.join("mel_fb_1024_80.bin"));
    let ml = |v: &[f32]| lightvc_core::ship_front::mel(v, &fb);
    let (mx, mo, mb) = (ml(&x), ml(&y_ok), ml(&y_bad));
    let l1 = |a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>| -> f64 {
        let t = a[0].len().min(b[0].len());
        let mut s = 0f64;
        for m in 0..80 { for ti in 8..t { s += (a[m][ti] as f64 - b[m][ti] as f64).abs(); } }
        s / (80.0 * (t - 8) as f64)
    };
    println!("resynth mel-L1 vs input: x32768 {:.3} / [-1,1] {:.3}", l1(&mx, &mo), l1(&mx, &mb));
}

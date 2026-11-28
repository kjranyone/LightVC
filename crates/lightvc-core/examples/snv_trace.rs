//! 製品の経路をブロックごとに追う診断: 入力と登録は生の f32(48 kHz・LE)。出力の波形と、源と模型の出力の Y(<trace.csv>.ysrc・.yout = [フレーム][29] の生の f32)と、ブロックごとの表(入力の RMS・門・有声・lf・源と出力の c0・出力の RMS・出力の番の値)を書く。
//! LVC_COPY = 写しの描画・LVC_NO_GUARD = 出力の番を切る・LVC_ESM = 描画器の界面の包絡の平滑を入れる・LVC_EK = 平滑の重みの差し替え(t・t−1・t−2・t−3)・LVC_EK0 = 0 行目(音量)だけの重み・LVC_E1E = 0 行目だけ動きに応じる平滑(a0,θ)・LVC_C0X = 音量の速い成分を源から取る(時定数・フレーム)。
//!     cargo run --release -p lightvc-core --example snv_trace -- <in.f32> <out.f32> <trace.csv> <半音> <操作者のフォルダ> <目標のフォルダ>

use std::io::Write;
use std::path::Path;

use lightvc_core::snv_stream::SnvStream;

fn rd(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn dir(p: &str) -> Vec<Vec<f32>> {
    let mut v: Vec<_> = std::fs::read_dir(p).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "f32")).collect();
    v.sort();
    v.iter().map(|p| rd(p)).collect()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let mut s = SnvStream::load(
        Path::new(&env("LVC_FRONT", "results/snv_p126_gate_export")),
        Path::new(&env("LVC_K3G", "results/snv_p126_k3g16kv_export")),
        Path::new(&env("LVC_VOC", "results/snv_p123_s/export_20k")),
    )
    .unwrap();
    s.enable_parallel(4);
    s.set_semitones(a[3].parse().unwrap());
    s.set_copy(std::env::var_os("LVC_COPY").is_some());
    s.set_guard(std::env::var_os("LVC_NO_GUARD").is_none());
    s.set_env_smooth(std::env::var_os("LVC_ESM").is_some());
    if let Ok(k) = std::env::var("LVC_EK") {
        let w: Vec<f32> = k.split(',').map(|v| v.parse().unwrap()).collect();
        s.set_env_kernel(Some([w[0], w[1], w[2], w[3]]));
    }
    if let Ok(k) = std::env::var("LVC_EK0") {
        let w: Vec<f32> = k.split(',').map(|v| v.parse().unwrap()).collect();
        s.set_env_kernel0(Some([w[0], w[1], w[2], w[3]]));
    }
    if let Ok(k) = std::env::var("LVC_E1E") {
        let w: Vec<f32> = k.split(',').map(|v| v.parse().unwrap()).collect();
        s.set_env_adaptive0(Some((w[0], w[1])));
    }
    if let Ok(k) = std::env::var("LVC_C0X") {
        s.set_c0_from_source(Some(k.parse().unwrap()));
    }
    let (op, tg) = (dir(&a[4]), dir(&a[5]));
    s.enroll_operator(&op.iter().map(|x| x.as_slice()).collect::<Vec<_>>()).unwrap();
    s.enroll_target(&tg.iter().map(|x| x.as_slice()).collect::<Vec<_>>()).unwrap();
    let x = rd(Path::new(&a[0]));
    let n = x.len() / SnvStream::BLOCK;
    let mut y = vec![0f32; n * SnvStream::BLOCK];
    let (mut ys, mut yo) = (Vec::new(), Vec::new());
    let mut csv = std::io::BufWriter::new(std::fs::File::create(&a[2]).unwrap());
    writeln!(csv, "t,in_rms,open,v,lf,src_c0,out_c0,out_rms,g_ref,g_lo,g_allow,g_db,g_lim").unwrap();
    for i in 0..n {
        let r = i * SnvStream::BLOCK..(i + 1) * SnvStream::BLOCK;
        s.process_block(&x[r.clone()], &mut y[r.clone()]);
        let c = s.last_cond();
        ys.extend_from_slice(s.last_src_y());
        yo.extend_from_slice(s.last_y());
        let g = s.last_guard().unwrap_or([0.0; 5]);
        writeln!(csv, "{:.3},{:.6},{},{},{:.4},{:.4},{:.4},{:.6},{:.2},{:.2},{:.2},{:.2},{:.2}", i as f32 * 0.005, rms(&x[r.clone()]), s.gate_open() as u8, c[26], c[25], s.last_src_y()[0], c[0], rms(&y[r]), g[0], g[1], g[2], g[3], g[4]).unwrap();
    }
    std::fs::write(&a[1], y.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let raw = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::write(format!("{}.ysrc", a[2]), raw(&ys)).unwrap();
    std::fs::write(format!("{}.yout", a[2]), raw(&yo)).unwrap();
}

use lightvc_core::ys1_codec::CodecDecoder;
use std::fs;
use std::time::Instant;

fn main() {
    let mut d = CodecDecoder::load("models/ys1_decoder.bin", "models/ys1_decoder.json").unwrap();
    let frames = fs::read("/tmp/opencode/parity_z.bin").unwrap().chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<_>>();
    let z = &frames[..32];
    // warm-up
    d.reset();
    for _ in 0..2000 { d.decode_step(z); }
    // 10,000 step・1 frame/step
    let mut ts = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let t0 = Instant::now();
        d.decode_step(z);
        ts.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| -> f64 {
        let idx = ((q * (ts.len() - 1) as f64).round()) as usize;
        ts[idx]
    };
    let mean: f64 = ts.iter().sum::<f64>() / ts.len() as f64;
    let rtf = mean / 10.0;
    let out = format!(
        "{{\n  \"schema\": 1,\n  \"cpu_name\": \"{}\",\n  \"thread_count\": 1,\n  \"runs\": {},\n  \"warmup_steps\": 2000,\n  \"p50_ms\": {:.4},\n  \"p95_ms\": {:.4},\n  \"p99_ms\": {:.4},\n  \"max_ms\": {:.4},\n  \"mean_ms\": {:.4},\n  \"mean_rtf\": {:.4},\n  \"weight_bytes\": {},\n  \"state_bytes\": {},\n  \"hop\": 480,\n  \"sample_rate\": 48000\n}}",
        std::env::var("HOSTNAME").unwrap_or_default(),
        ts.len(), p(0.5), p(0.95), p(0.99), *ts.last().unwrap(), mean, rtf,
        fs::metadata("models/ys1_decoder.bin").unwrap().len(),
        0
    );
    fs::write("results/ys1/s1_4_probe.json", &out).unwrap();
    println!("{}", out);
}

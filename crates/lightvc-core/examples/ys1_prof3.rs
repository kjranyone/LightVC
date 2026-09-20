use lightvc_core::ys1_codec::*;
use std::time::Instant;

fn main() {
    let d = CodecDecoder::load("models/ys1_decoder.bin", "models/ys1_decoder.json").unwrap();
    // stage0 res0: c=256, hidden=128, w1 = [128][256*7=1792]
    let st = &d.stages[0];
    let ru = &st.res[0];
    let mut y = vec![0f32; ru.hidden];
    let x = vec![0.1f32; 256 * 7];
    for _ in 0..500 { lightvc_core::ys1_codec::fma_h(&ru.w1, &x, &ru.bias1, &mut y, 256 * 7); }
    let t = Instant::now();
    for _ in 0..2000 { lightvc_core::ys1_codec::fma_h(&ru.w1, &x, &ru.bias1, &mut y, 256 * 7); }
    println!("res conv1 (stage0): {:.4} ms/call", t.elapsed().as_secs_f64()*1000.0/2000.0);
    let t2 = Instant::now();
    let mut y2 = vec![0f32; ru.c];
    let x2 = vec![0.1f32; ru.hidden];
    for _ in 0..2000 { lightvc_core::ys1_codec::fma_h(&ru.w2, &x2, &ru.bias2, &mut y2, ru.hidden); }
    println!("res conv2 (stage0): {:.4} ms/call", t2.elapsed().as_secs_f64()*1000.0/2000.0);
    let t3 = Instant::now();
    let mut y3 = vec![0f32; 512];
    let x3 = vec![0.1f32; 224];
    for _ in 0..2000 { lightvc_core::ys1_codec::fma_h(&d.pre_w, &x3, &d.pre_b, &mut y3, 224); }
    println!("pre: {:.4} ms", t3.elapsed().as_secs_f64()*1000.0/2000.0);
}

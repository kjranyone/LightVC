use lightvc_core::ys1_codec::*;
use std::fs;
use std::time::Instant;

fn main() {
    let mut d = CodecDecoder::load("models/ys1_decoder.bin", "models/ys1_decoder.json").unwrap();
    let frames = fs::read("/tmp/opencode/parity_z.bin").unwrap().chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<_>>();
    let z = &frames[..32];
    for _ in 0..500 { d.decode_step(z); }
    // 全体
    let t = Instant::now();
    for _ in 0..2000 { d.decode_step(z); }
    println!("decode_step total: {:.3} ms", t.elapsed().as_secs_f64()*1000.0/2000.0);
    // 段別近似: 各 stage の up dot と res fma を単体計測
    for (si, st) in d.stages.iter().enumerate() {
        let n_in = [1usize, 3, 4, 5][si];
        let w2 = st.cout * 2 * st.stride;
        let mut buf = vec![0f32; w2];
        let xin = vec![0.1f32; st.cin];
        let t0 = Instant::now();
        for _ in 0..2000 {
            for _ in 0..n_in {
                unsafe { dot_f16_out(&st.up_wt2, &xin, st.cin, &mut buf) };
            }
        }
        let t_up = t0.elapsed().as_secs_f64()*1000.0/2000.0;
        println!("stage{} up(×{}): {:.3} ms", si, n_in, t_up);
    }
}

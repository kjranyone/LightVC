use lightvc_core::ys1_codec::CodecDecoder;
use std::fs;

fn main() {
    let d = CodecDecoder::load("models/ys1_decoder.bin", "models/ys1_decoder.json").unwrap();
    let zb = fs::read("/tmp/opencode/parity_z.bin").unwrap();
    let frames: Vec<f32> = zb.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    // pre 層 1 frame 手動再現
    let z = &frames[0..32];
    // channel-major: cm[c*7+k] = (zero pad 6, z[c])
    let mut cm = [0f64; 32 * 7];
    for c in 0..32 { cm[c * 7 + 6] = z[c] as f64; }
    let mut h = vec![0f64; 512];
    for o in 0..512 {
        let w = &d.pre_w[o * 32 * 7..(o + 1) * 32 * 7];
        let acc = d.pre_b[o] as f64
            + cm.iter().zip(w.iter()).map(|(x, w)| x * (*w as f64)).sum::<f64>();
        h[o] = acc;
    }
    println!("rust pre[0..5]: {:?}", &h[..5]);
    // stage0 up 1 frame: raw[k*cout+o] = sum_ci w[ci*cout*2r + o*2r + k] * h[ci]
    let st = &d.stages[0];
    let (cin, cout, r) = (st.cin, st.cout, st.stride);
    let mut raw = vec![0f64; 2 * r * cout];
    for ci in 0..cin {
        let xc = h[ci];
        let w = &st.up_w[ci * cout * 2 * r..(ci + 1) * cout * 2 * r];
        for o in 0..cout {
            for k in 0..2 * r {
                raw[k * cout + o] += xc * (w[o * 2 * r + k] as f64);
            }
        }
    }
    println!("rust up0 t0 ch0..5: {:?}", &raw[..5]);
    println!("rust up0 t1 ch0..5: {:?}", &raw[cout..cout + 5]);
}

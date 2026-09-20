use lightvc_core::ys1_codec::{CodecDecoder, snake_pub};
use std::fs;

fn main() {
    let mut d = CodecDecoder::load("models/ys1_decoder.bin", "models/ys1_decoder.json").unwrap();
    let zb = fs::read("/tmp/opencode/parity_z.bin").unwrap();
    let frames: Vec<f32> = zb.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let z = &frames[0..32];
    // pre 再現 (channel-major)
    let mut cm = [0f32; 32 * 7];
    for c in 0..32 { cm[c * 7 + 6] = z[c]; }
    let mut h = vec![0f32; 512];
    for o in 0..512 {
        let w = &d.pre_w[o * 32 * 7..(o + 1) * 32 * 7];
        h[o] = d.pre_b[o] + cm.iter().zip(w.iter()).map(|(x, w)| x * w).sum::<f32>();
    }
    // stage0 up (1入力時刻 → 3出力時刻)
    let st = &mut d.stages[0];
    let (cin, cout, r) = (st.cin, st.cout, st.stride);
    let mut raw = vec![0f64; 2 * r * cout];
    for ci in 0..cin {
        let xc = h[ci] as f64;
        let w = &st.up_w[ci * cout * 2 * r..(ci + 1) * cout * 2 * r];
        for o in 0..cout {
            for k in 0..2 * r {
                raw[k * cout + o] += xc * (w[o * 2 * r + k] as f64);
            }
        }
    }
    let mut up_out = vec![0f32; 3 * cout];
    for (i, v) in up_out.iter_mut().enumerate() { *v = raw[i] as f32 + st.up_b[i % cout]; }
    // res0 (dil=1) を 3 時刻分。st1 = 0 初期化。
    // Python: 系列全体(3時刻)を一度に通す → st1=6点pad。時刻 t の窓 = 過去(6-k)+現在
    let ru = &mut st.res[0];
    let c = ru.c;
    let mut ser: Vec<Vec<f32>> = (0..3).map(|t| (0..c).map(|ch| up_out[t * c + ch]).collect()).collect();
    // snake1 適用後の系列を作ってから conv
    let mut out_all = vec![0f32; 3 * c];
    for t in 0..3 {
        for ch in 0..c {
            let s = snake_pub(ser[t][ch], ru.a1[ch], ru.b1[ch]);
            // dil=1: 窓 = [hist..., s]。hist は過去 t 時刻分(初回は pad ゼロ)
            let mut y1 = vec![0f32; ru.hidden];
            for o in 0..ru.hidden {
                let w = &ru.w1[o * c * 7..(o + 1) * c * 7];
                let mut acc = ru.bias1[o];
                for ch2 in 0..c {
                    for k in 0..7 {
                        let pos_t = t as isize - (6 - k) as isize;
                        let val = if pos_t < 0 { 0.0 }
                            else { snake_pub(ser[pos_t as usize][ch2], ru.a1[ch2], ru.b1[ch2]) };
                        acc += w[ch2 * 7 + k] * val;
                    }
                }
                y1[o] = acc;
            }
            let mut o_v = ser[t][ch];
            for o in 0..c {
                let mut acc = ru.bias2[o];
                let w = &ru.w2[o * ru.hidden..(o + 1) * ru.hidden];
                for (hh, &v) in y1.iter().enumerate() {
                    acc += w[hh] * snake_pub(v, ru.a2[hh], ru.b2[hh]);
                }
                o_v += acc;
            }
            out_all[t * c + ch] = o_v;
        }
    }
    println!("rust res0 t0 ch0..5: {:?}", &out_all[..5]);
}

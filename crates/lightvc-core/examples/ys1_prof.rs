use lightvc_core::ys1_codec::*;
use std::fs;
use std::time::Instant;

fn main() {
    let mut d = CodecDecoder::load("models/ys1_decoder.bin", "models/ys1_decoder.json").unwrap();
    let frames = fs::read("/tmp/opencode/parity_z.bin").unwrap().chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<_>>();
    let z = &frames[..32];
    for _ in 0..200 { d.decode_step(z); }
    // decode_step は一体なので、段別の計測は構造体内部が必要。
    // ここでは簡易に: pre+bias相当(pre_w 512x224), up0(512x256x6)を直接回して相対比較
    let t0 = Instant::now();
    for _ in 0..1000 {
        let mut h = vec![0f32; 512];
        fma_h_pub(&d.pre_w, &vec![0.1f32; 224], &d.pre_b, &mut h, 224);
    }
    let t_pre = t0.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    println!("pre equiv: {:.3} ms", t_pre);
    let st = &d.stages[0];
    let wt = &st.up_wt;
    let t1 = Instant::now();
    for _ in 0..1000 {
        let mut raw = vec![0f32; 6 * 256];
        unsafe {
            use std::arch::x86_64::*;
            for ci in 0..512 {
                let xc = 0.1f32;
                let bx = _mm256_set1_ps(xc);
                let wr = wt.as_ptr().add(ci * 1536);
                let dst = raw.as_mut_ptr();
                let mut o = 0;
                while o < 1536 {
                    let wh = _mm_loadu_si128(wr.add(o) as *const __m128i);
                    let wf = _mm256_cvtph_ps(wh);
                    let acc = _mm256_fmadd_ps(bx, wf, _mm256_loadu_ps(dst.add(o)));
                    _mm256_storeu_ps(dst.add(o), acc);
                    o += 8;
                }
            }
        }
    }
    let t_up = t1.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    println!("up0 equiv: {:.3} ms", t_up);
}

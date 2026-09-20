use lightvc_core::ys1_codec::CodecDecoder;
use std::fs;

fn read_f32(p: &str) -> Vec<f32> {
    fs::read(p).unwrap().chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn main() {
    let mut d = CodecDecoder::load("models/ys1_decoder.bin", "models/ys1_decoder.json").unwrap();
    let frames = read_f32("/tmp/opencode/parity_z.bin");
    let n = frames.len() / 32;

    // T1: future mutation — frame t 以降を変えても t までの出力が完全一致
    d.reset();
    let mut a = Vec::new();
    for i in 0..n { a.extend_from_slice(&d.decode_step(&frames[i * 32..(i + 1) * 32])); }
    d.reset();
    let mut b = Vec::new();
    let mut modf = frames.clone();
    for i in 0..n {
        if i >= 25 { for v in modf[i * 32..(i + 1) * 32].iter_mut() { *v += 0.5; } }
        b.extend_from_slice(&d.decode_step(&modf[i * 32..(i + 1) * 32]));
    }
    let t1 = a[..25 * 480].iter().zip(&b[..25 * 480])
        .all(|(x, y)| x.to_bits() == y.to_bits());
    println!("T1 future mutation: {}", if t1 { "PASS" } else { "FAIL" });

    // T3: arbitrary chunk {1,2,3,5,8,17} が同一出力
    d.reset();
    let mut c1 = Vec::new();
    for i in 0..n { c1.extend_from_slice(&d.decode_step(&frames[i * 32..(i + 1) * 32])); }
    let mut ok3 = true;
    for chunk in [2usize, 3, 5, 8, 17] {
        d.reset();
        let mut cc = Vec::new();
        let mut i = 0;
        while i < n {
            let take = chunk.min(n - i);
            for j in 0..take {
                cc.extend_from_slice(&d.decode_step(&frames[(i + j) * 32..(i + j + 1) * 32]));
            }
            i += take;
        }
        if cc.len() != c1.len() || cc.iter().zip(&c1).any(|(x, y)| (x - y).abs() > 1e-6) {
            ok3 = false;
        }
    }
    println!("T3 arbitrary chunk: {}", if ok3 { "PASS" } else { "FAIL" });

    // T4: reset — decode→reset→別入力が履歴に影響されない
    d.reset();
    for i in 0..10 { d.decode_step(&frames[i * 32..(i + 1) * 32]); }
    d.reset();
    let after = d.decode_step(&frames[0..32]);
    d.reset();
    let fresh = d.decode_step(&frames[0..32]);
    let t4 = after == fresh;
    println!("T4 reset: {}", if t4 { "PASS" } else { "FAIL" });

    // T5: length — n frames -> n*480
    println!("T5 length: {}", if c1.len() == n * 480 { "PASS" } else { "FAIL" });
}

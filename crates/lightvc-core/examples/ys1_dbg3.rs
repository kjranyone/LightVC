use lightvc_core::ys1_codec::CodecDecoder;
use std::fs;

fn main() {
    let mut d = CodecDecoder::load("models/ys1_decoder.bin", "models/ys1_decoder.json").unwrap();
    d.reset();
    let zb = fs::read("/tmp/opencode/parity_z.bin").unwrap();
    let frames: Vec<f32> = zb.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let z = &frames[0..32];
    let y = d.decode_step(z);
    // 最終出力の最初の3サンプルは post(32ch)由来。代わりに内部公開が無いので
    // 全体の最初 frame 出力だけ
    println!("rust y0[0..5]: {:?}", &y[..5]);
}

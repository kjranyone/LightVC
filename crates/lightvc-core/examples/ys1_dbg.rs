use lightvc_core::ys1_codec::CodecDecoder;
use std::fs;

fn main() {
    let mut d = CodecDecoder::load("models/ys1_decoder.bin", "models/ys1_decoder.json").unwrap();
    d.reset();
    let zb = fs::read("/tmp/opencode/parity_z.bin").unwrap();
    let frames: Vec<f32> = zb.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    // pre 層の出力を 50 frame 分自前で再現(内部関数を pub にする代わり decode_step の
    // 最初段だけ比較するため、ここでは公開 API が無いのでスキップ。
    // 代わりに decode_step 全体の最初の frame を Python stream と比較する数値を出す
    let y0 = d.decode_step(&frames[0..32]);
    println!("rust y0[..8]: {:?}", &y0[..8]);
}

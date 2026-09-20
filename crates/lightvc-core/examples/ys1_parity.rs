use std::fs;

fn main() {
    let mut d = lightvc_core::ys1_codec::CodecDecoder::load(
        "models/ys1_decoder.bin", "models/ys1_decoder.json").unwrap();
    d.reset();
    let zb = fs::read("/tmp/opencode/parity_z.bin").unwrap();
    let frames: Vec<f32> = zb.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let mut y = Vec::new();
    for i in 0..50 {
        y.extend_from_slice(&d.decode_step(&frames[i * 32..(i + 1) * 32]));
    }
    // Python reference
    let ref_raw = fs::read("models/ys1_parity_y.npy").unwrap();
    // npy parse: header skip
    let (off, _) = {
        let magic = &ref_raw[..6];
        assert_eq!(magic, b"\x93NUMPY");
        let hlen = u16::from_le_bytes([ref_raw[8], ref_raw[9]]) as usize;
        (10 + hlen, ())
    };
    let ref_f: Vec<f32> = ref_raw[off..].chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let n = y.len().min(ref_f.len());
    let mut err = 0f64;
    let mut ref_pow = 0f64;
    for i in 0..n {
        err += ((y[i] - ref_f[i]) as f64).powi(2);
        ref_pow += (ref_f[i] as f64).powi(2);
    }
    let snr = 10.0 * (ref_pow / err).log10();
    let maxabs = (0..n).map(|i| (y[i] - ref_f[i]).abs())
        .fold(0f32, f32::max);
    println!("frames 50 -> {} samples", y.len());
    println!("parity: max abs {:.2e} / SNR {:.1} dB (gate: <=1e-4, >=80dB)", maxabs, snr);
}

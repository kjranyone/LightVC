//! オフライン変換の実行部(GUI の Offline タブから別スレッドで呼ぶ)。

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use lightvc_core::snv_stream::SnvStream;

use crate::state::AppState;

/// 240 サンプルずつ流し、経路の遅れ(LATENCY_SAMPLES)を落として入力と時刻を揃える。
pub fn convert(stream: &mut SnvStream, x: &[f32]) -> Vec<f32> {
    const HOP: usize = SnvStream::BLOCK;
    let lat = stream.latency_samples();
    let n = (x.len() + lat).div_ceil(HOP);
    let mut xin = x.to_vec();
    xin.resize(n * HOP, 0.0);
    let mut y = vec![0f32; n * HOP];
    for (xi, yi) in xin.chunks_exact(HOP).zip(y.chunks_exact_mut(HOP)) {
        stream.process_block(xi, yi);
    }
    y.drain(..lat);
    y.truncate(x.len());
    let pk = y.iter().fold(0f32, |a, v| a.max(v.abs()));
    if pk > 0.99 {
        y.iter_mut().for_each(|v| *v *= 0.99 / pk);
    }
    y
}

pub fn run_offline_conversion(state: Arc<Mutex<AppState>>, source: PathBuf, reference: PathBuf) {
    let cfg = state.lock().unwrap().settings.clone();
    let result = (|| -> anyhow::Result<Vec<f32>> {
        let mut stream = crate::state::build_stream(&cfg, &[reference])?;
        let x = crate::state::load_48k(&source)?;
        Ok(convert(&mut stream, &x))
    })();

    let mut s = state.lock().unwrap();
    match result {
        Ok(samples) => {
            s.status = format!("Converted: {} samples", samples.len());
            s.error = None;
            s.offline_result = Some(samples);
        }
        Err(e) => {
            s.error = Some(format!("Conversion failed: {e}"));
            s.status = "Error".to_string();
        }
    }
}

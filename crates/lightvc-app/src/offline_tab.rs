//! オフライン変換の実行部(GUI のファイル変換から別スレッドで呼ぶ)。
//!
//! 段: 読み込み → 話し手と目標の声を測る(登録)→ 変換。画面は [`Job`] から進み具合と残り時間を読み、中止もできる。
//! 登録の段は途中の進み具合が取れないので、前回の速さ(音声 1 秒あたりの所要秒)から見積もる。

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use lightvc_core::snv_stream::SnvStream;

use crate::state::AppState;

const HOP: usize = SnvStream::BLOCK;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Stage {
    Loading,
    Measuring,
    Converting,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Loading => "音声を読み込んでいます",
            Stage::Measuring => "話し手と目標の声を測っています",
            Stage::Converting => "変換しています",
        }
    }
}

/// 速さの覚え(音声 1 秒あたりの所要秒)。初回は既定、変換を終えるたびに測り直す。
#[derive(Clone, Copy)]
pub struct Rates {
    pub measure: f32,
    pub convert: f32,
}

impl Default for Rates {
    fn default() -> Self {
        Self { measure: 0.5, convert: 0.5 }
    }
}

struct Info {
    stage: Stage,
    stage_started: Instant,
    /// 変換元と目標の参照の長さ(秒)。
    source_s: f32,
    target_s: f32,
    /// 登録の段に実際にかかった秒(変換の段に入ってから)。
    measured_s: Option<f32>,
}

/// 1 回のファイル変換の進み具合(画面と変換の糸で共有)。
pub struct Job {
    started: Instant,
    rates: Rates,
    info: Mutex<Info>,
    done: AtomicUsize,
    total: AtomicUsize,
    cancel: AtomicBool,
}

/// 画面に出す進み具合。
pub struct Progress {
    pub stage: Stage,
    pub frac: f32,
    pub elapsed_s: f32,
    pub remaining_s: Option<f32>,
}

impl Job {
    pub fn new(rates: Rates) -> Arc<Self> {
        let now = Instant::now();
        Arc::new(Self {
            started: now,
            rates,
            info: Mutex::new(Info { stage: Stage::Loading, stage_started: now, source_s: 0.0, target_s: 0.0, measured_s: None }),
            done: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
        })
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    fn set_stage(&self, stage: Stage) {
        let mut i = self.info.lock().unwrap();
        if stage == Stage::Converting {
            i.measured_s = Some(i.stage_started.elapsed().as_secs_f32());
        }
        i.stage = stage;
        i.stage_started = Instant::now();
    }

    pub fn progress(&self) -> Progress {
        let i = self.info.lock().unwrap();
        let elapsed_s = self.started.elapsed().as_secs_f32();
        let in_stage = i.stage_started.elapsed().as_secs_f32();
        let measure_est = (i.source_s + i.target_s) * self.rates.measure;
        let convert_est = i.source_s * self.rates.convert;
        let (frac, remaining_s) = match i.stage {
            Stage::Loading => (0.0, None),
            Stage::Measuring => {
                let left = (measure_est - in_stage).max(0.5);
                let total = in_stage + left + convert_est;
                ((in_stage / total).min(0.95), Some(left + convert_est))
            }
            Stage::Converting => {
                let (done, total) = (self.done.load(Ordering::Relaxed), self.total.load(Ordering::Relaxed).max(1));
                let measured = i.measured_s.unwrap_or(0.0);
                let left = if done > 0 { in_stage / done as f32 * (total - done) as f32 } else { convert_est };
                ((measured + in_stage) / (measured + in_stage + left).max(1e-3), Some(left))
            }
        };
        Progress { stage: i.stage, frac: frac.clamp(0.0, 1.0), elapsed_s, remaining_s }
    }

    /// 測った速さ(次の見積もりに使う)。
    fn rates_measured(&self) -> Option<Rates> {
        let i = self.info.lock().unwrap();
        let audio = i.source_s + i.target_s;
        let measured = i.measured_s?;
        (audio > 1.0 && i.source_s > 1.0).then(|| Rates { measure: measured / audio, convert: i.stage_started.elapsed().as_secs_f32() / i.source_s })
    }
}

/// 240 サンプルずつ流し、経路の遅れ(latency_samples)を落として入力と時刻を揃える。中止されたら None。
fn convert(stream: &mut SnvStream, x: &[f32], job: &Job) -> Option<Vec<f32>> {
    let lat = stream.latency_samples();
    let n = (x.len() + lat).div_ceil(HOP);
    let mut xin = x.to_vec();
    xin.resize(n * HOP, 0.0);
    let mut y = vec![0f32; n * HOP];
    job.total.store(n, Ordering::Relaxed);
    for (k, (xi, yi)) in xin.as_chunks::<HOP>().0.iter().zip(y.as_chunks_mut::<HOP>().0.iter_mut()).enumerate() {
        if k % 64 == 0 {
            if job.cancelled() {
                return None;
            }
            job.done.store(k, Ordering::Relaxed);
        }
        stream.process_block(xi, yi);
    }
    job.done.store(n, Ordering::Relaxed);
    y.drain(..lat);
    y.truncate(x.len());
    let pk = y.iter().fold(0f32, |a, v| a.max(v.abs()));
    if pk > 0.99 {
        y.iter_mut().for_each(|v| *v *= 0.99 / pk);
    }
    Some(y)
}

/// 変換元の話し手は操作者とは限らないので、話し手の特徴は変換元そのものから測る(リアルタイムの話者登録は使わない)。
pub fn run_offline_conversion(state: Arc<Mutex<AppState>>, job: Arc<Job>, source: PathBuf, reference: PathBuf) {
    let cfg = state.lock().unwrap().settings.clone();
    let result = (|| -> anyhow::Result<Option<Vec<f32>>> {
        let x = crate::state::load_48k(&source)?;
        let target_s = crate::state::load_48k(&reference).map(|t| t.len() as f32 / 48_000.0).unwrap_or(0.0);
        {
            let mut i = job.info.lock().unwrap();
            (i.source_s, i.target_s) = (x.len() as f32 / 48_000.0, target_s);
        }
        job.set_stage(Stage::Measuring);
        let mut stream = crate::state::build_stream_with(&cfg, std::slice::from_ref(&x), &[reference])?;
        if job.cancelled() {
            return Ok(None);
        }
        job.set_stage(Stage::Converting);
        Ok(convert(&mut stream, &x, &job))
    })();

    let mut s = state.lock().unwrap();
    match result {
        Ok(Some(samples)) => {
            if let Some(r) = job.rates_measured() {
                s.offline_rates = r;
            }
            s.status = format!("変換しました({:.1} 秒かかりました)", job.started.elapsed().as_secs_f32());
            s.error = None;
            s.offline_result = Some(samples);
        }
        Ok(None) => s.status = "変換を中止しました".to_string(),
        Err(e) => {
            s.error = Some(format!("変換に失敗しました: {e}"));
            s.status = "変換に失敗しました".to_string();
        }
    }
    s.offline_done = true;
}

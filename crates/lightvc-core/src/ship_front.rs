//! Left-aligned causal front-end. Mirrors `training/ship_front.py`.
//!
//! Zero lookahead by construction: every frame's rightmost sample is
//! `t*hop + hop - 1`, and the head is **zero**-padded, never reflected
//! (reflect reads `x[1..n_fft-hop+1]` at t=0 — that is 768 samples of
//! lookahead, `PROCEDURE.md` 2.1 item #9).
//!
//! No utterance-wide statistics anywhere (`CLAUDE.md` shipping gate):
//! the voicing threshold is a calibrated constant, the envelope reference is
//! a constant, and `fill` uses a running last-voiced value with a fixed
//! pre-voiced default.

use std::f32::consts::PI;

pub const SR: f32 = 44100.0;
pub const NFFT_A: usize = 1024;
pub const HOP_A: usize = 256;
pub const NFFT_S: usize = 512;
pub const HOP_S: usize = 128;
pub const N_MEL: usize = 80;
pub const NBIN_S: usize = NFFT_S / 2 + 1;

pub const F0_MIN: f32 = 60.0;
pub const F0_MAX: f32 = 600.0;
/// Calibrated absolute voicing threshold — replaces `voi / voi.median()`.
pub const VOI_ABS: f32 = 1.1505;
/// Log-mel reference: `H = exp(mel_lin - MEL_REF)`. Replaces `H / H.amax()`.
/// Recalibrated 2026-08-12: the old 1.49 left the prior 21.85 dB too quiet
/// (52 utterances, log-gain sd 0.084 — a constant offset, not variation).
/// Must move together with `training/ship_front.py` (`CLAUDE.md`).
pub const MEL_REF: f32 = -1.0254;
/// f0 used before the first voiced frame. Historical effective value; changing
/// it shifts the phase accumulator and invalidates trained weights.
pub const PRE_VOICED_HZ: f32 = 50.0;
pub const EPS: f32 = 1e-8;

pub fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos()).collect()
}

/// In-place radix-2 complex FFT (`n` must be a power of two).
///
/// Written out rather than pulled in as a dependency: the front-end is the
/// only user, `n` is always 512 or 1024, and a new crate would have to be
/// license-audited (MIT-only project).
pub fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two() && im.len() == n);
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    re[i + k + len / 2] * cr - im[i + k + len / 2] * ci,
                    re[i + k + len / 2] * ci + im[i + k + len / 2] * cr,
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// One left-aligned analysis frame -> (re, im) for bins `0..=n/2`.
pub fn frame_rfft_pub(seg: &[f32], w: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let n = seg.len();
    let mut re: Vec<f64> = seg.iter().zip(w).map(|(s, ww)| (s * ww) as f64).collect();
    let mut im = vec![0.0f64; n];
    fft(&mut re, &mut im);
    re.truncate(n / 2 + 1);
    im.truncate(n / 2 + 1);
    (re, im)
}

/// Left-aligned STFT magnitude+phase via direct DFT. `out[f][t]` = (re, im).
///
/// Head is zero-padded by `nfft - hop`; the tail is zero-padded so the overlap
/// tail-off falls outside `[..n]`.
pub fn cstft(x: &[f32], nfft: usize, hop: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let w = hann(nfft);
    let head = nfft - hop;
    let q = ((hop - x.len() % hop) % hop) + head;
    let mut xp = vec![0.0f32; head + x.len() + q];
    xp[head..head + x.len()].copy_from_slice(x);
    let t = (xp.len() - nfft) / hop + 1;
    let nb = nfft / 2 + 1;
    let mut re = vec![vec![0.0f32; t]; nb];
    let mut im = vec![vec![0.0f32; t]; nb];
    for ti in 0..t {
        let off = ti * hop;
        let (fr, fi) = frame_rfft_pub(&xp[off..off + nfft], &w);
        for (k, (rr, ii)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
            rr[ti] = fr[k] as f32;
            ii[ti] = fi[k] as f32;
        }
    }
    (re, im)
}

/// Frames `cstft` actually emits for `n` samples.
pub fn n_frames(n: usize, hop: usize, nfft: usize) -> usize {
    (n + ((hop - n % hop) % hop) + (nfft - hop)) / hop
}

/// Analysis grid -> synthesis grid. **Local map only** — an index that depends
/// on the total length would not close until the utterance ends (6.30 ms of
/// measured lookahead came from exactly that).
pub fn to_frames_idx(t: usize) -> usize {
    let v = t * HOP_S + HOP_S;
    v.saturating_sub(HOP_A) / HOP_A
}

/// Running fill with the last voiced f0. No `any()` over the utterance.
pub fn fill(f0: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(f0.len());
    let mut last: Option<f32> = None;
    for &v in f0 {
        if v > 50.0 {
            last = Some(v);
        }
        out.push(last.map(|x| x.max(50.0)).unwrap_or(PRE_VOICED_HZ));
    }
    out
}

/// Frame -> sample using only past frames. Frame `j` is usable once
/// `j*hop + hop - 1` has been emitted, so sample `n` interpolates `j-1, j`.
/// 補間は **f64** で行う (Python は v.double() で補間して cumsum)。f32 補間だと
/// 位相 cumsum に ~1e-7 相対の誤差が毎サンプル積もり、7.5 秒で ~2e-3 rad の
/// ドリフト → 倍音和 sin((K+1/2)φ) で ×K 増幅されて励起 −54dB / 出力 −23dB の
/// 床になった (シフト後 f0 の変換で顕在化。実測)。
pub fn frame_upsample_causal_f64(v: &[f32], n: usize, hop: usize) -> Vec<f64> {
    let last = v.len().saturating_sub(2) as f64;
    (0..n)
        .map(|s| {
            let t = (s as f64 + 1.0) / hop as f64 - 2.0;
            let i = t.floor().clamp(0.0, last);
            let fr = (t - i).clamp(0.0, 1.0);
            let j = i as usize;
            v[j] as f64 * (1.0 - fr) + v[(j + 1).min(v.len() - 1)] as f64 * fr
        })
        .collect()
}

pub fn frame_upsample_causal(v: &[f32], n: usize, hop: usize) -> Vec<f32> {
    frame_upsample_causal_f64(v, n, hop).into_iter().map(|x| x as f32).collect()
}

/// Running phase accumulator — identical offline and streaming.
pub fn phase_of(f0: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let f0u64 = frame_upsample_causal_f64(&fill(f0), n, HOP_A);
    let mut acc = 0.0f64;
    let mut phi = Vec::with_capacity(n);
    let mut f0u = Vec::with_capacity(n);
    for &f in &f0u64 {
        acc += f.max(0.0);
        phi.push(((2.0 * std::f64::consts::PI * acc / SR as f64)
            .rem_euclid(2.0 * std::f64::consts::PI)) as f32);
        f0u.push(f as f32);
    }
    (phi, f0u)
}

/// Causal running median over the last `k` frames.
pub fn causal_median(v: &[f32], k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len());
    for i in 0..v.len() {
        let mut w: Vec<f32> = (0..k)
            .map(|j| v[i.saturating_sub(k - 1 - j)])
            .collect();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out.push(w[k / 2]);
    }
    out
}

/// Two-lobe harmonic-sum f0 on the left-aligned analysis grid.
///
/// The negative lobe (penalising `k+0.5`) stays — it is what suppresses the
/// octave error on weak fundamentals.
/// f0 候補テーブル（torch `F0_MIN * 2**(arange*10/1200)` の f32 ビット列を焼き込み）。
/// libm powf とは 9/399 本が 1 ULP 違い、位相積分で増幅されて励起 53.7dB /
/// 出力 23dB の言語間差の根だった。生成: training で本ファイル冒頭のコマンド。
pub const F0_CAND_BITS: [i32; 399] = [
    1114636288, 1114727404, 1114819046, 1114911219, 1115003927, 1115097172, 1115190958, 1115285286,
    1115380160, 1115475585, 1115571562, 1115668095, 1115725025, 1115773853, 1115822963, 1115872358,
    1115922039, 1115972008, 1116022265, 1116072815, 1116123657, 1116174794, 1116226227, 1116277957,
    1116329987, 1116382319, 1116434954, 1116487894, 1116541141, 1116594696, 1116648561, 1116702738,
    1116757230, 1116812037, 1116867161, 1116922604, 1116978370, 1117034458, 1117090871, 1117147610,
    1117204678, 1117262077, 1117319808, 1117377874, 1117436277, 1117495018, 1117554098, 1117613521,
    1117673289, 1117733402, 1117793864, 1117854676, 1117915840, 1117977358, 1118039234, 1118101468,
    1118164061, 1118227018, 1118290339, 1118354027, 1118418084, 1118482512, 1118547314, 1118612489,
    1118678044, 1118743978, 1118810294, 1118876995, 1118944081, 1119011556, 1119079423, 1119147681,
    1119216336, 1119285388, 1119354840, 1119424695, 1119494954, 1119565621, 1119636697, 1119708184,
    1119780086, 1119852404, 1119925141, 1119998299, 1120071882, 1120145890, 1120220327, 1120295196,
    1120370498, 1120446236, 1120522413, 1120599031, 1120676094, 1120753602, 1120831560, 1120909969,
    1120988833, 1121068153, 1121147933, 1121228174, 1121308881, 1121390055, 1121471700, 1121553818,
    1121636411, 1121719483, 1121803035, 1121887072, 1121971596, 1122056609, 1122142115, 1122228117,
    1122314616, 1122401616, 1122489120, 1122577132, 1122665654, 1122754688, 1122844237, 1122934305,
    1123024896, 1123116012, 1123207654, 1123299827, 1123392535, 1123485780, 1123579564, 1123673894,
    1123768768, 1123864193, 1123960170, 1124056703, 1124113634, 1124162461, 1124211571, 1124260966,
    1124310647, 1124360616, 1124410873, 1124461422, 1124512265, 1124563402, 1124614834, 1124666565,
    1124718596, 1124770927, 1124823562, 1124876502, 1124929749, 1124983304, 1125037169, 1125091346,
    1125145837, 1125200645, 1125255769, 1125311212, 1125366978, 1125423066, 1125479479, 1125536218,
    1125593286, 1125650686, 1125708417, 1125766483, 1125824885, 1125883626, 1125942706, 1126002129,
    1126061896, 1126122010, 1126182472, 1126243283, 1126304448, 1126365967, 1126427842, 1126490076,
    1126552670, 1126615626, 1126678947, 1126742635, 1126806692, 1126871120, 1126935922, 1127001097,
    1127066652, 1127132586, 1127198902, 1127265602, 1127332689, 1127400165, 1127468031, 1127536290,
    1127604944, 1127673996, 1127743448, 1127813303, 1127883562, 1127954229, 1128025305, 1128096792,
    1128168694, 1128241012, 1128313748, 1128386908, 1128460490, 1128534499, 1128608935, 1128683804,
    1128759106, 1128834844, 1128911021, 1128987639, 1129064702, 1129142210, 1129220167, 1129298576,
    1129377440, 1129456761, 1129536541, 1129616782, 1129697490, 1129778663, 1129860308, 1129942426,
    1130025019, 1130108091, 1130191643, 1130275680, 1130360204, 1130445217, 1130530723, 1130616724,
    1130703223, 1130790225, 1130877729, 1130965741, 1131054262, 1131143296, 1131232845, 1131322913,
    1131413504, 1131504620, 1131596262, 1131688437, 1131781143, 1131874388, 1131968172, 1132062502,
    1132157375, 1132252801, 1132348776, 1132445311, 1132502241, 1132551069, 1132600178, 1132649574,
    1132699255, 1132749224, 1132799482, 1132850030, 1132900874, 1132952010, 1133003443, 1133055173,
    1133107204, 1133159534, 1133212170, 1133265110, 1133318357, 1133371911, 1133425777, 1133479955,
    1133534445, 1133589254, 1133644377, 1133699821, 1133755586, 1133811674, 1133868086, 1133924826,
    1133981894, 1134039294, 1134097024, 1134155091, 1134213492, 1134272234, 1134331315, 1134390737,
    1134450505, 1134510618, 1134571080, 1134631891, 1134693057, 1134754574, 1134816450, 1134878683,
    1134941278, 1135004233, 1135067555, 1135131242, 1135195300, 1135259729, 1135324530, 1135389707,
    1135455260, 1135521195, 1135587510, 1135654211, 1135721296, 1135788773, 1135856638, 1135924898,
    1135993551, 1136062604, 1136132055, 1136201911, 1136272171, 1136342837, 1136413914, 1136485400,
    1136557303, 1136629620, 1136702357, 1136775515, 1136849098, 1136923105, 1136997543, 1137072411,
    1137147714, 1137223451, 1137299629, 1137376248, 1137453310, 1137530820, 1137608775, 1137687186,
    1137766048, 1137845370, 1137925148, 1138005390, 1138086096, 1138167271, 1138248915, 1138331034,
    1138413626, 1138496699, 1138580252, 1138664288, 1138748813, 1138833825, 1138919332, 1139005332,
    1139091833, 1139178832, 1139266337, 1139354347, 1139442870, 1139531902, 1139621453, 1139711520,
    1139802112, 1139893228, 1139984870, 1140077045, 1140169751, 1140262996, 1140356780, 1140451110,
    1140545983, 1140641409, 1140737384, 1140833919, 1140890849, 1140939677, 1140988786, 1141038182,
    1141087863, 1141137832, 1141188090, 1141238638, 1141289482, 1141340618, 1141392051, 1141443781,
    1141495812, 1141548142, 1141600778, 1141653718, 1141706965, 1141760519, 1141814385, 1141868563,
    1141923053, 1141977862, 1142032985, 1142088429, 1142144194, 1142200282, 1142256694,
];

pub fn causal_f0(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (re, im) = cstft(x, NFFT_A, HOP_A);
    let nb = re.len();
    let t = re[0].len();
    let mag: Vec<Vec<f32>> = (0..nb)
        .map(|f| (0..t).map(|i| (re[f][i] * re[f][i] + im[f][i] * im[f][i]).sqrt()).collect())
        .collect();
    let binhz = SR / NFFT_A as f32;
    let ncand = F0_CAND_BITS.len();
    let cand: Vec<f32> = F0_CAND_BITS.iter().map(|&b| f32::from_bits(b as u32)).collect();

    let take = |f: f32, ti: usize| -> f32 {
        if !(f > 0.0 && f < SR / 2.0 - binhz) {
            return 0.0;
        }
        let b = (f / binhz).clamp(0.0, nb as f32 - 2.0);
        let lo = b as usize;
        let fr = b - lo as f32;
        mag[lo][ti] * (1.0 - fr) + mag[lo + 1][ti] * fr
    };

    let mut colsum = vec![0.0f32; t];
    for row in &mag {
        for (ti, s) in colsum.iter_mut().enumerate() {
            *s += row[ti];
        }
    }
    let mut f0 = vec![0.0f32; t];
    let mut voi = vec![0.0f32; t];
    for ti in 0..t {
        let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
        for (ci, &c) in cand.iter().enumerate() {
            let mut s = 0.0f32;
            for k in 1..=20u32 {
                let w = 1.0 / (k as f32).sqrt();
                s += w * take(c * k as f32, ti) - 0.5 * w * take(c * (k as f32 + 0.5), ti);
            }
            if s > best {
                best = s;
                bi = ci;
            }
        }
        f0[ti] = cand[bi];
        voi[ti] = best / (colsum[ti] / (nb as f32).sqrt() + EPS);
    }
    let f0m = causal_median(&f0, 5);
    let out = (0..t).map(|i| if voi[i] > VOI_ABS { f0m[i] } else { 0.0 }).collect();
    (out, voi)
}

/// Left-aligned causal log-mel. `fb` is the librosa slaney filterbank
/// (`[N_MEL, NFFT_A/2+1]`, row-major) — shipped as data rather than
/// reimplemented, so the Python and Rust banks cannot drift.
///
/// Matches `causal_mel._mel`: magnitude `sqrt(|S|^2 + 1e-9)`, then
/// `log(clamp(mel, min=1e-5))`.
pub fn mel(x: &[f32], fb: &[f32]) -> Vec<Vec<f32>> {
    // ⚠ **`cstft` を使わない。** `causal_mel._mel` は `right_pad=0` で、
    //    `cstft` の「末尾 (nfft-hop) 余分」を持たない。フレーム数が食い違う
    //    （44100 サンプルで 176 対 172）。mel は解析側の実体に合わせる。
    let w = hann(NFFT_A);
    let head = NFFT_A - HOP_A;
    let mut xp = vec![0.0f32; head + x.len()];
    xp[head..].copy_from_slice(x);
    let nb = NFFT_A / 2 + 1;
    if xp.len() < NFFT_A {
        return vec![vec![]; N_MEL];
    }
    let t = (xp.len() - NFFT_A) / HOP_A + 1;
    let mut spec = vec![vec![0.0f32; t]; nb];
    for ti in 0..t {
        let off = ti * HOP_A;
        // f64 の radix-2 FFT。素朴 DFT(f32) だと 1024 項の丸めで mel が 0.257 ずれた。
        let (fr, fi) = frame_rfft_pub(&xp[off..off + NFFT_A], &w);
        for (k, row) in spec.iter_mut().enumerate() {
            row[ti] = ((fr[k] * fr[k] + fi[k] * fi[k] + 1e-9).sqrt()) as f32;
        }
    }
    let mut out = vec![vec![0.0f32; t]; N_MEL];
    for m in 0..N_MEL {
        let row = &fb[m * nb..(m + 1) * nb];
        for ti in 0..t {
            let mut acc = 0.0f32;
            for (f, &c) in row.iter().enumerate() {
                if c != 0.0 {
                    acc += c * spec[f][ti];
                }
            }
            out[m][ti] = acc.max(1e-5).ln();
        }
    }
    out
}

/// Highest harmonic index considered. The *effective* count is decided by the
/// per-sample Nyquist mask, not by an utterance-median f0 (that would be an
/// utterance-wide statistic — `CLAUDE.md` shipping gate).
pub const KMAX: usize = (SR as usize / 2) / (F0_MIN as usize);

/// Harmonic excitation: impulse train (zero relative phase) + white noise.
///
/// Normalised by `sqrt(active harmonic count)` so the level does not move when
/// f0 moves. No envelope `amax()`, no utterance RMS — `H = exp(mel_lin - MEL_REF)`
/// carries the absolute level.
pub fn excitation(f0: &[f32], n: usize, noise: &[f32], noise_mix: f32) -> Vec<f32> {
    // ⚠ 倍音を 1 本ずつ cos() で積むと **O(n × K)**（f0=110 Hz で K≈200、
    //   実測 0.24 ms ＝ RTF 0.041。f0 が低いほど悪化）。
    //   総和には閉形式がある（Dirichlet 核）:
    //     sum_{k=1..K} cos(kφ) = sin((K+1/2)φ) / (2 sin(φ/2)) − 1/2
    //   毎サンプル O(1)。K は per-sample の Nyquist マスクそのもの
    //   （発話統計を使わない設計は変わらない）。数学的に恒等なので
    //   Python 参照（明示和）との parity 検査がそのまま同一性の証明になる。
    let (phi, f0u) = phase_of(f0, n);
    let nyq = SR / 2.0;
    (0..n)
        .map(|s| {
            let f = f0u[s].max(1e-3);
            // K は Python 明示和のマスク `(k as f32) * f < nyq` と**同一の f32 判定**で
            // 数える。単純な floor(nyq/f) だと丸みの境界で ±1 本ずれ、シフト後の
            // 高 f0 (K~50) では 1 本 = 1/sqrt(K) ≈ 0.13 の段差が −30dB の床を作る
            // (未シフト男声 K~180 では 95dB に埋まっていた)。
            let mut kmax = ((nyq as f64) / (f as f64)).floor() as usize;
            if kmax >= 1 && (kmax as f32) * f >= nyq {
                kmax -= 1;
            }
            if ((kmax + 1) as f32) * f < nyq {
                kmax += 1;
            }
            let kmax = kmax.min(KMAX);
            let v = if kmax == 0 {
                0.0
            } else {
                let p = phi[s] as f64;
                let half = (0.5 * p).sin();
                let sum = if half.abs() < 1e-6 {
                    kmax as f64                       // φ→0 の極限は K
                } else {
                    ((kmax as f64 + 0.5) * p).sin() / (2.0 * half) - 0.5
                };
                (sum as f32) / (kmax as f32).sqrt()
            };
            (1.0 - noise_mix) * v + noise_mix * noise[s]
        })
        .collect()
}

/// Prior spectrum on the synthesis grid: `E * H`, where `H = exp(mel_lin - MEL_REF)`.
/// `mel_lin` is the log-mel mapped onto the linear bin axis, `[NBIN_S, T]`.
pub fn nhv_spec(mel_lin: &[Vec<f32>], f0: &[f32], n: usize, noise: &[f32],
                noise_mix: f32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let e = excitation(f0, n, noise, noise_mix);
    let (re, im) = cstft(&e, NFFT_S, HOP_S);
    let t = re[0].len().min(mel_lin[0].len());
    let mut or_ = vec![vec![0.0f32; t]; NBIN_S];
    let mut oi = vec![vec![0.0f32; t]; NBIN_S];
    for b in 0..NBIN_S {
        for ti in 0..t {
            let h = (mel_lin[b][ti] - MEL_REF).exp();
            or_[b][ti] = re[b][ti] * h;
            oi[b][ti] = im[b][ti] * h;
        }
    }
    (or_, oi)
}

/// Streaming front-end: **1 ブロック = 解析 1 フレーム**。
///
/// `K=2` 合成フレーム = 256 サンプル = `HOP_A` なので、1 ブロックで新しく閉じる
/// 解析フレームはちょうど 1 本。ブロックごとにリング全体を測り直すと
/// 5 倍の無駄になる（実測 full graph RTF 0.496 -> 下記で 0.1 台）。
pub struct FrontState {
    w: Vec<f32>,
    f0_hist: Vec<f32>,
    cand: Vec<f32>,
}

impl Default for FrontState {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontState {
    pub fn new() -> Self {
        let ncand = ((F0_MAX / F0_MIN).log2() * 1200.0 / 10.0) as usize + 1;
        Self {
            w: hann(NFFT_A),
            f0_hist: Vec::new(),
            cand: F0_CAND_BITS.iter().map(|&b| f32::from_bits(b as u32)).collect(),
        }
    }

    /// `win` は最新の `NFFT_A` サンプル（左寄せ＝最右が現在）。
    /// 戻り値は (log-mel 1 フレーム, f0, voi)。
    pub fn step(&mut self, win: &[f32], fb: &[f32]) -> (Vec<f32>, f32, f32) {
        debug_assert_eq!(win.len(), NFFT_A);
        let (fr, fi) = frame_rfft_pub(win, &self.w);
        let nb = NFFT_A / 2 + 1;
        let mag: Vec<f32> = (0..nb)
            .map(|k| ((fr[k] * fr[k] + fi[k] * fi[k]).sqrt()) as f32)
            .collect();
        let spec: Vec<f32> = (0..nb)
            .map(|k| ((fr[k] * fr[k] + fi[k] * fi[k] + 1e-9).sqrt()) as f32)
            .collect();

        let mut mel_out = vec![0.0f32; N_MEL];
        for (m, o) in mel_out.iter_mut().enumerate() {
            let row = &fb[m * nb..(m + 1) * nb];
            let mut acc = 0.0f32;
            for (f, &c) in row.iter().enumerate() {
                if c != 0.0 {
                    acc += c * spec[f];
                }
            }
            *o = acc.max(1e-5).ln();
        }

        let binhz = SR / NFFT_A as f32;
        let take = |f: f32| -> f32 {
            if !(f > 0.0 && f < SR / 2.0 - binhz) {
                return 0.0;
            }
            let b = (f / binhz).clamp(0.0, nb as f32 - 2.0);
            let lo = b as usize;
            let fr = b - lo as f32;
            mag[lo] * (1.0 - fr) + mag[lo + 1] * fr
        };
        let colsum: f32 = mag.iter().sum();
        let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
        for (ci, &c) in self.cand.iter().enumerate() {
            let mut s = 0.0f32;
            for k in 1..=20u32 {
                let w = 1.0 / (k as f32).sqrt();
                s += w * take(c * k as f32) - 0.5 * w * take(c * (k as f32 + 0.5));
            }
            if s > best {
                best = s;
                bi = ci;
            }
        }
        let voi = best / (colsum / (nb as f32).sqrt() + EPS);
        self.f0_hist.push(self.cand[bi]);
        // 因果 5 点メディアン（直近 5 本だけ保持すれば足りる）
        let n = self.f0_hist.len();
        let mut w5: Vec<f32> = (0..5).map(|j| self.f0_hist[n.saturating_sub(5 - j).max(0)]).collect();
        w5.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let f0m = w5[2];
        if self.f0_hist.len() > 8 {
            self.f0_hist.drain(..self.f0_hist.len() - 8);
        }
        (mel_out, if voi > VOI_ABS { f0m } else { 0.0 }, voi)
    }
}

/// Left-aligned iSTFT（`ship_front.cistft` と同じ）。
///
/// 固有遅延 `nfft − hop`（重畳が閉じるまで）で先読みではない。
/// `torch.istft` は center=False の端で窓和ゼロを拒否するので自前 OLA。
/// `re/im`: `[NBIN][T]`。返り値は `n` サンプル。
pub fn cistft(re: &[Vec<f32>], im: &[Vec<f32>], n: usize, nfft: usize, hop: usize) -> Vec<f32> {
    let t = re[0].len();
    let w = hann(nfft);
    let l = (t - 1) * hop + nfft;
    let mut y = vec![0.0f32; l];
    let mut ws = vec![0.0f32; l];
    let nb = nfft / 2 + 1;
    let mut fr = vec![0.0f64; nfft];
    let mut fi = vec![0.0f64; nfft];
    for ti in 0..t {
        // 逆 rFFT: 共役対称に展開して逆 FFT（符号反転 + 1/N）
        for k in 0..nb {
            fr[k] = re[k][ti] as f64;
            fi[k] = -(im[k][ti] as f64);
        }
        for k in nb..nfft {
            fr[k] = re[nfft - k][ti] as f64;
            fi[k] = im[nfft - k][ti] as f64;
        }
        fft(&mut fr, &mut fi);
        let off = ti * hop;
        for i in 0..nfft {
            let v = (fr[i] / nfft as f64) as f32 * w[i];
            y[off + i] += v;
            ws[off + i] += w[i] * w[i];
        }
    }
    let head = nfft - hop;
    (0..n)
        .map(|i| {
            let j = head + i;
            if j < l { y[j] / ws[j].max(1e-8) } else { 0.0 }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// C.3 parity: f0 は Python 実装と ±5 cent / voiced-unvoiced 一致
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn td(name: &str) -> Vec<f32> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name);
        std::fs::read(p).expect("testdata")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn cistft_matches_python() {
        let t = 24usize;
        let nb = NFFT_S / 2 + 1;
        let rd = td("ci_re.bin");
        let id = td("ci_im.bin");
        let re: Vec<Vec<f32>> = (0..nb).map(|k| rd[k * t..(k + 1) * t].to_vec()).collect();
        let im: Vec<Vec<f32>> = (0..nb).map(|k| id[k * t..(k + 1) * t].to_vec()).collect();
        let n = t * HOP_S;
        let got = cistft(&re, &im, n, NFFT_S, HOP_S);
        let want = td("ci_y.bin");
        // ⚠ 末尾 nfft は OLA の立ち下がり（窓和 ~1e-10）にかかる。クランプ除算が
        //   丸め差を 1e8 倍に増幅する不良条件領域で、実運用では cstft が余分に
        //   フレームを取って [:n] の外へ押し出す。検査は良条件領域だけで行う。
        let m = n - NFFT_S;
        let e = got[..m].iter().zip(&want[..m])
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(e <= 1e-3, "cistft 最大絶対誤差 {e}");
    }

    #[test]
    fn fill_matches_python() {
        let f0 = td("sf_f0.bin");
        let got = fill(&f0);
        let want = td("sf_fill.bin");
        let e = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(e <= 1e-4, "fill max abs err {e}");
    }

    #[test]
    fn phase_matches_python() {
        let f0 = td("sf_f0.bin");
        let n = td("sf_w.bin").len();
        let (phi, f0u) = phase_of(&f0, n);
        let we = td("sf_f0u.bin");
        let eu = f0u.iter().zip(&we).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(eu <= 1e-3, "f0u max abs err {eu}");
        // 位相は cumsum なので f32 の丸めが効く。角度差で見る。
        let wp = td("sf_phi.bin");
        let ep = phi.iter().zip(&wp)
            .map(|(a, b)| {
                let d = (a - b).abs();
                d.min(2.0 * PI - d)
            })
            .fold(0.0f32, f32::max);
        assert!(ep <= 2e-2, "phase max angle err {ep}");
    }

    #[test]
    fn mel_matches_python() {
        let w = td("sf_w.bin");
        let fb = td("mel_fb_1024_80.bin");
        let got = mel(&w, &fb);
        let want = td("sf_mel.bin");
        let t = got[0].len();
        assert_eq!(got.len() * t, want.len(), "mel の形が違う");
        let mut e = 0.0f32;
        for (m, row) in got.iter().enumerate() {
            for (ti, v) in row.iter().enumerate() {
                e = e.max((v - want[m * t + ti]).abs());
            }
        }
        // 許容 1e-3。差 0.00057 は torch(f32 FFT) と本実装(f64 FFT) の精度差で、
        // 本実装のほうが高精度。実装差ではないので締めない。
        assert!(e <= 1e-3, "mel max abs err {e}");
    }

    #[test]
    fn excitation_matches_python() {
        let f0 = td("sf_f0.bin");
        let z = td("sf_noise.bin");
        let n = z.len();
        let got = excitation(&f0, n, &z, 0.3);
        let want = td("sf_exc.bin");
        let e = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(e <= 1e-3, "excitation max abs err {e}");
    }

    #[test]
    fn streaming_front_equals_offline() {
        // ⚠ ブロック実行が一括実行と一致することを検査する。一致しないと
        //    「学習で見た値」と「製品が作る値」が別物になる。
        let w = td("sf_w.bin");
        let fb = td("mel_fb_1024_80.bin");
        let mel_off = mel(&w, &fb);
        let (f0_off, _) = causal_f0(&w);

        let head = NFFT_A - HOP_A;
        let mut xp = vec![0.0f32; head + w.len()];
        xp[head..].copy_from_slice(&w);
        let t = (xp.len() - NFFT_A) / HOP_A + 1;

        let mut st = FrontState::new();
        let (mut em, mut ef) = (0.0f32, 0usize);
        for ti in 0..t {
            let (m, f0, _) = st.step(&xp[ti * HOP_A..ti * HOP_A + NFFT_A], &fb);
            for (k, v) in m.iter().enumerate() {
                em = em.max((v - mel_off[k][ti]).abs());
            }
            if (f0 > 0.0) != (f0_off[ti] > 0.0) || (f0 - f0_off[ti]).abs() > 1e-3 {
                ef += 1;
            }
        }
        assert!(em <= 1e-5, "streaming mel が offline と違う: {em}");
        assert_eq!(ef, 0, "streaming f0 が offline と {ef} フレーム違う");
    }

    #[test]
    fn kmax_is_not_utterance_derived() {
        // 倍音数は毎サンプルの Nyquist マスクで決まる（発話中央値 f0 から
        // 決めない）。KMAX は F0_MIN 由来の定数であることを固定する。
        assert_eq!(KMAX, 367);
    }

    #[test]
    fn f0_matches_python() {
        let w = td("sf_w.bin");
        let (f0, _) = causal_f0(&w);
        let want = td("sf_f0.bin");
        assert_eq!(f0.len(), want.len(), "frame count");
        let mut vu = 0usize;
        let mut worst_cent = 0.0f32;
        for (a, b) in f0.iter().zip(&want) {
            if (*a > 0.0) != (*b > 0.0) {
                vu += 1;
                continue;
            }
            if *a > 0.0 {
                let c = 1200.0 * (a / b).log2().abs();
                worst_cent = worst_cent.max(c);
            }
        }
        assert_eq!(vu, 0, "voiced/unvoiced が {vu} フレームで食い違う");
        assert!(worst_cent <= 5.0, "f0 が {worst_cent:.2} cent ずれた（許容 5）");
    }
}

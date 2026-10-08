"""高い f0(1kHz 超・叫び・裏返り)に対応した教師の f0 v2(current/f0_range.md)。
A = harvest + stonemask(上限 1000Hz・既存の教師と同一・既存の npy を渡せる)・B = 48kHz の DIO + stonemask(上限 2400Hz)。A を基本にし、次を満たす区間だけ B を採る:
  候補: B ≥ 950Hz。各フレームが次の検査(gate)を全て通る(p = B の付近のスペクトルのピーク周波数・S = ハン窓 1024 点のパワー):
   1. 基音より下が空: ピークが 80Hz〜0.55p の最大より EMPTY_DB 以上大きい(1kHz 超の声は基音より下に成分が無い)
   2. 倍音の谷: 1.5p と 2.5p(±8%)の最大よりピークが VALLEY_DB 以上大きい(第 2 倍音を基音と誤ると 1.5B に第 3 倍音・2.5B に第 5 倍音が出る・
      基音が極端に弱く傾きが急な声でも -14dB 程度は出るので 20dB を課す)
   3. 音の線: ピークが ±25% の中央値より TONAL_DB 以上高い(雑音の帯を落とす)
   4. 第 2 倍音の線: 2p(±7%)が存在し(ピークの HARM2_DB 以内)かつ ±40% の中央値より H2_TONAL_DB 以上高い(倍音の積み重なりの確認)
  区間: gate を通り隣と ±JUMP_CENT 以内で連続する MIN_RUN フレーム以上。区間の両端は最大 EXT フレーム、B ≥ 600 かつ隣と ±JUMP_CENT 以内かつ gate(4 を除く)を通る限り延ばす
  (グライドの出入りで A が無声・1 オクターブ下になる階段を避ける)。区間の前後 CTX フレームの CTX_FRAC 以上で A が B/2 に安定して有声なら、B は第 2 倍音の取り違えとして区間ごと棄却する(基音が弱い声の中の短い誤り)。区間内で A が有声かつ B と KEEP_A_CENT 以内なら A を残す(置換は A が欠ける・外れるフレームだけ)。
区間の外は A とビット一致。フレーム k の中心 = k·HOP(48kHz で 240 サンプル)・窓は中心 ±512。
    uv run python f0hi.py   # 合成音の自己検査
"""
from __future__ import annotations

import numpy as np
import pyworld
from scipy.signal import resample_poly

SR = 48000
HOP = 240
NFFT = 1024
EMPTY_DB = 25.0
VALLEY_DB = 20.0
TONAL_DB = 10.0
HARM2_DB = 35.0
H2_TONAL_DB = 8.0
MIN_RUN = 6
EXT = 4
JUMP_CENT = 300.0
CTX = 8
CTX_FRAC = 0.5
KEEP_A_CENT = 100.0
CEIL_LO, CEIL_HI = 1000.0, 2400.0
FR = np.fft.rfftfreq(NFFT, 1.0 / SR)
WIN = np.hanning(NFFT)


def dio48(x48: np.ndarray, ceil: float) -> np.ndarray:
    """48kHz のまま DIO + stonemask(harvest は 1.2〜1.5kHz 以上で無声と返すことがある・DIO はオクターブ誤りが多いので gate で守る)。"""
    x = x48.astype(np.float64)
    f0, t = pyworld.dio(x, SR, f0_floor=60, f0_ceil=ceil, frame_period=5.0)
    return pyworld.stonemask(x, f0, t, SR)


def harvest(x48: np.ndarray, ceil: float) -> np.ndarray:
    x16 = resample_poly(x48.astype(np.float64), 1, 3)
    f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=ceil, frame_period=5.0)
    return pyworld.stonemask(x16, f0, t, 16000)


def spectra(x48: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """frames [M] → S [M, NFFT/2+1](フレーム k の窓は中心 k·HOP の ±NFFT/2)。"""
    pad = np.concatenate([np.zeros(NFFT // 2, np.float32), x48.astype(np.float32), np.zeros(NFFT, np.float32)])
    idx = frames[:, None] * HOP + np.arange(NFFT)[None]
    return np.abs(np.fft.rfft(pad[idx] * WIN, axis=1)) ** 2 + 1e-12


def gate_row(S: np.ndarray, ft: float, strict: bool = True) -> bool:
    """S [F]・ft(Hz)の候補が『高い基音』か。strict = False は第 2 倍音の線(4)を課さない(区間の縁の延長用)。"""
    near = (FR >= 0.93 * ft) & (FR <= 1.07 * ft)
    if not near.any():
        return False
    ip = np.where(near)[0][np.argmax(S[near])]
    p, pk = FR[ip], S[ip]
    low = (FR >= 80.0) & (FR <= 0.55 * p)
    v15 = (FR >= 1.38 * p) & (FR <= 1.62 * p)
    wide = (FR >= 0.75 * p) & (FR <= 1.25 * p)
    if not (low.any() and v15.any()):
        return False
    db = lambda m: 10 * np.log10(pk / S[m].max())
    ok = db(low) >= EMPTY_DB and db(v15) >= VALLEY_DB and 10 * np.log10(pk / np.median(S[wide])) >= TONAL_DB
    if ok and 2.62 * p < SR / 2 - 1000:
        v25 = (FR >= 2.3 * p) & (FR <= 2.7 * p)
        ok = ok and db(v25) >= VALLEY_DB
    if ok and strict:
        h2 = (FR >= 1.86 * p) & (FR <= 2.14 * p)
        h2w = (FR >= 1.2 * p) & (FR <= 2.8 * p)
        if not h2.any():
            return False
        ok = db(h2) <= HARM2_DB and 10 * np.log10(S[h2].max() / np.median(S[h2w])) >= H2_TONAL_DB
    return bool(ok)


def cents(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1200 * np.log2(np.where(a > 0, a, 1.0) / np.where(b > 0, b, 1.0))


def teacher_f0(x48: np.ndarray, n: int | None = None, A: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """→ (f0 [T] float32・無声 0, replaced [T] bool = A から置換したフレーム)。A を渡せば(既存の npy)harvest を再計算しない。"""
    n = len(x48) // HOP if n is None else n
    if A is None:
        A = harvest(x48, CEIL_LO)
    A = np.pad(A[:n].astype(np.float32), (0, max(0, n - len(A))))
    B = dio48(x48, CEIL_HI)
    B = np.pad(B[:n], (0, max(0, n - len(B))))
    out = A.copy()
    pool = np.where(B >= 600.0)[0]
    if not (B >= 950.0).any():
        return out, np.zeros(n, bool)
    S = dict(zip(pool.tolist(), spectra(x48, pool))) if len(pool) else {}
    strict = np.zeros(n, bool)
    relaxed = np.zeros(n, bool)
    for t in pool:
        relaxed[t] = gate_row(S[t], float(B[t]), strict=False)
        if B[t] >= 950.0 and relaxed[t]:
            strict[t] = gate_row(S[t], float(B[t]), strict=True)
    ok = np.zeros(n, bool)
    i = 0
    while i < n:
        if not strict[i]:
            i += 1
            continue
        j = i + 1
        while j < n and strict[j] and abs(cents(B[j:j + 1], B[j - 1:j])[0]) <= JUMP_CENT:
            j += 1
        if j - i >= MIN_RUN:
            a, b = i, j
            for _ in range(EXT):
                t = a - 1
                if t >= 0 and B[t] >= 600.0 and relaxed[t] and abs(cents(B[t:t + 1], B[a:a + 1])[0]) <= JUMP_CENT:
                    a = t
                else:
                    break
            for _ in range(EXT):
                t = b
                if t < n and B[t] >= 600.0 and relaxed[t] and abs(cents(B[t:t + 1], B[b - 1:b])[0]) <= JUMP_CENT:
                    b = t + 1
                else:
                    break
            lo_, hi_ = max(0, a - CTX), min(n, b + CTX)
            half = lambda sl: float(((A[sl] > 0) & (np.abs(cents(A[sl], B[sl] / 2)) < 100)).mean()) if len(A[sl]) else 0.0
            before, after = half(slice(lo_, a)), half(slice(b, hi_))
            if not (a - lo_ >= 4 and hi_ - b >= 4 and before >= CTX_FRAC and after >= CTX_FRAC):
                ok[a:b] = True
        i = j
    keepA = ok & (A > 0) & (np.abs(cents(A, B)) <= KEEP_A_CENT)
    rep = ok & ~keepA
    out[rep] = B[rep]
    return out.astype(np.float32), rep


def _tone(f0_hz, dur, h1_db=0.0, snr_db=25.0, seed=0, vib=0.02, tilt=1.0, hp_ratio=0.0):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0_hz * (1 + vib * np.sin(2 * np.pi * 5.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.zeros(n)
    for k in range(1, int(12000 / f0_hz) + 1):
        a = k ** -tilt * (10 ** (h1_db / 20) if k == 1 else 1.0)
        x += a * np.sin(k * ph)
    if hp_ratio > 0:
        from scipy.signal import butter, sosfilt
        x = sosfilt(butter(4, hp_ratio * f0_hz, "highpass", fs=SR, output="sos"), x)
    x /= np.abs(x).max()
    x += rng.standard_normal(n) * 10 ** (-snr_db / 20) * x.std()
    env = np.minimum(1, np.minimum(t, dur - t) / 0.04)
    return (x * env * 0.3).astype(np.float32)


def _frames(f, lo=0.06, hi=0.54):
    t = np.arange(len(f)) * HOP / SR
    return (t > lo) & (t < hi)


if __name__ == "__main__":
    print("正例: 高音だけ(1000〜2000Hz・傾き k^-1 / k^-2 / k^-3・SNR 25 / 10dB): 100cent 以内で検出した割合")
    for tilt in (1.0, 2.0, 3.0):
        for f0 in (1000, 1300, 1600, 2000):
            row = []
            for snr in (25, 10):
                x = _tone(f0, 0.6, snr_db=snr, tilt=tilt)
                f, rep = teacher_f0(x)
                v = _frames(f)
                row.append(((np.abs(cents(f, np.full_like(f, f0))) < 100) & (f > 0))[v].mean())
            print(f"  tilt k^-{tilt:.0f} f0 {f0}: SNR25 {row[0]:.2f}  SNR10 {row[1]:.2f}")
    print("負例(普通の声 300〜900Hz): B を採ったフレーム数(0 であるべき)。H1 を 0/−10/−20/−30dB 弱める × 傾き k^-1 / k^-3 / k^-4 × 高域通過(基音を欠く)")
    bad = total = 0
    for tilt in (1.0, 3.0, 4.0):
        for hp in (0.0, 0.8):
            for f0 in (300, 450, 600, 750, 900):
                for h1 in (0, -10, -20, -30, -40):
                    x = _tone(f0, 0.6, h1_db=h1, tilt=tilt, hp_ratio=hp)
                    f, rep = teacher_f0(x)
                    nb = int(rep[_frames(f)].sum())
                    bad += nb
                    total += 1
                    if nb:
                        print(f"  NEG-HIT tilt {tilt} hp {hp} f0 {f0} H1 {h1:+d}dB: B frames {nb}")
    print(f"負例 {total} 条件・B 採用の合計 {bad} フレーム")
    print("強制 B(DIO が返さなくても gate 単体を当てる): 真の基音 500〜900Hz に対し B = 2·f0 を与えて gate が通る条件数")
    forced = fpass = 0
    for tilt in (1.0, 3.0, 4.0):
        for hp in (0.0, 0.8):
            for f0 in (500, 600, 700, 800, 900):
                for h1 in (0, -10, -20, -30, -40, -50):
                    x = _tone(f0, 0.6, h1_db=h1, tilt=tilt, hp_ratio=hp)
                    S = spectra(x, np.arange(40, 90))
                    npass = sum(gate_row(S[i], 2.0 * f0) for i in range(len(S)))
                    forced += 1
                    if npass:
                        fpass += 1
                        print(f"  FORCED-PASS tilt {tilt} hp {hp} f0 {f0} H1 {h1:+d}dB: {npass}/50 frames")
    print(f"強制 B: {forced} 条件のうち gate を通った条件 {fpass}")

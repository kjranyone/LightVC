"""自己官能評価プロセス（ear-less AB）: 提示前の機械的自己判定レポート生成。

背景: 私は聴けないのに「改善したはず」と信念で巨塊サンプルを提示し、
3 回連続で耳棄却された。プロセスを批判的に再設計する:

  P1 較正セットでブラインド自己予測（命中率 = 私の「耳」の精度指標として記録）
  P2 類似 FAIL 検索（候補が過去のどの FAIL に計器上似ているか必ず開示）
  P3 天井との距離表（「まだ天井のどの位置か」を数値で明示）
  P4 最悪区間スライス（巨塊でなく最悪 12 秒 × 2 を提示 = 聴取コスト削減）
  P5 アンカー同梱（in-domain 天井なしの提示は禁止）

    CUDA_VISIBLE_DEVICES="" uv run python kansei_meter.py --judge <wav>...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF

ROOT = Path(__file__).resolve().parent.parent
SR = 44100
CAL = ROOT / "results/kansei_calibration.json"
AXES = ["jitter_pct", "aper", "b6_9k", "b9_22k", "spike", "abrupt_min"]


def battery(w: np.ndarray) -> dict:
    w16 = librosa.resample(w, orig_sr=SR, target_sr=16000).astype(np.float64)
    f0, t = pyworld.harvest(w16, 16000, f0_floor=50, f0_ceil=650)
    ap = pyworld.d4c(w16, f0, t, 16000)
    v = f0 > 0
    out: dict = {}
    if v.sum() > 30:
        vf = f0[v]
        d = np.abs(np.diff(vf)) / vf[:-1]
        out["jitter_pct"] = round(float(np.median(d[d > 0]) * 100), 2) if (d > 0).any() else 0.0
        out["f0_med"] = round(float(np.median(vf)), 1)
        out["aper"] = round(float((1.0 - ap[v]).mean()), 3)
    S = np.abs(librosa.stft(w, n_fft=2048, hop_length=512)) ** 2
    hz = librosa.fft_frequencies(sr=SR, n_fft=2048)
    bands = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 9000), (9000, 22050)]
    B = np.stack([10 * np.log10(S[(hz >= lo) & (hz < hi)].sum(0) + 1e-12) for lo, hi in bands])
    tot = 10 * np.log10(S.sum(0) + 1e-12)
    act = tot > tot.max() - 35
    B2 = B[:, act]
    out["b6_9k"] = round(float(np.median(B2[4])), 1)
    out["b9_22k"] = round(float(np.median(B2[5])), 1)
    hi = B2[5]
    med = np.median(hi)
    out["spike"] = round(float((hi > med + 15).mean()), 3)
    jump = np.abs(np.diff(B, axis=-1)) > 6.0
    out["abrupt_min"] = round(float((jump.sum(0) >= 3).sum() / (B.shape[1] * 512 / SR / 60)), 1)
    return out


def battery_segments(path: Path, seg_sec: float = 12.0) -> list[tuple[float, dict]]:
    """全体を seg_sec 窓で計測し、悪い順に返す（スライス提示用）。"""
    w, _ = librosa.load(str(path), sr=SR, mono=True)
    n = len(w)
    segs = []
    step = int(seg_sec * SR)
    for s0 in range(0, n - SR, step):
        s1 = min(s0 + step, n)
        seg = w[s0:s1]
        if len(seg) < SR * 3:
            break
        m = battery(seg)
        m["_t0"] = round(s0 / SR, 1)
        segs.append(m)
    return segs


def seg_badness(m: dict, cal: list) -> float:
    """較正 FAIL 中央からの相対距離（大きいほど悪い）。軸ごとにFAIL/PASS幅で正規化。"""
    z = 0.0
    for ax in AXES:
        fv = [r[ax] for r in cal if r["label"] == "FAIL" and ax in r]
        pv = [r[ax] for r in cal if r["label"] == "PASS" and ax in r]
        if not fv or not pv or ax not in m:
            continue
        lo, hi = min(fv + pv), max(fv + pv)
        if hi - lo < 1e-6:
            continue
        z += (m[ax] - float(np.median(pv))) / (hi - lo)
    return z


def self_predict(m: dict, cal: list, domain: str | None = None) -> tuple[str, float, str]:
    """P1: 較正セットへの最近傍(k=3)で PASS/FAIL を予測。

    domain='vout' 指定時は V 出力系サンプルのみで投票（TTS 合格サンプルは
    「合成の合格」でドメインが違い、False FAIL を生む実測のため）。
    戻り値: (予測, 距離, ドメイン内PASSサンプルの有無)。V 出力系で PASS 近傍が
    1 つも無い場合は予測を 'UNRESOLVED' にする＝判定に必要な陽性例が無い。
    """
    pool = cal
    if domain == "vout":
        pool = [r for r in cal
                if any(k in r["file"] for k in
                       ("v23_decompose", "diag_ear", "diag_ciptB", "pyv", "namikawa"))]
    ps = [r for r in pool if r["label"] == "PASS"]
    dists = []
    for r in pool:
        d = sum(abs(r[ax] - m[ax]) for ax in AXES if ax in r and ax in m)
        dists.append((d, r["label"], r["file"]))
    dists.sort()
    k3 = dists[:3]
    votes = sum(1 for _, l, _ in k3 if l == "FAIL")
    pred = "FAIL" if votes >= 2 else "PASS"
    if domain == "vout" and not ps:
        pred = "UNRESOLVED"
    return pred, float(np.mean([d for d, _, _ in k3])), "陽性例あり" if ps else "陽性例なし"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", nargs="+", default=None)
    ap.add_argument("--out-slices", default=None)
    a = ap.parse_args()
    cal = json.load(open(CAL))

    if not a.judge:
        return 0

    for p in a.judge:
        path = ROOT / p if not Path(p).is_absolute() else Path(p)
        if not path.exists():
            print(f"  SKIP {p}")
            continue
        w, _ = librosa.load(str(path), sr=SR, mono=True)
        m = battery(w)
        pred, dist, pos = self_predict(m, cal, domain="vout")
        bad = seg_badness(m, cal)
        # P2 類似FAIL検索
        sims = []
        for r in cal:
            if r["label"] != "FAIL":
                continue
            d = sum(abs(r[ax] - m[ax]) for ax in AXES if ax in r and ax in m)
            sims.append((d, r["file"]))
        sims.sort()
        # P3 天井(dom_shift0)との距離
        ceil_row = next((r for r in cal if "dom_shift0" in r["file"]), None)
        print(f"\n=== {path.name} ===")
        print(f"  計器: " + "  ".join(f"{ax}={m.get(ax)}" for ax in AXES))
        print(f"  P1 自己予測(V出力系較正): {pred}（最近傍距離 {dist:.2f}・{pos}）")
        print(f"  P2 類似FAIL: {sims[0][1]} (L1 {sims[0][0]:.2f}) / {sims[1][1]} ({sims[1][0]:.2f})")
        if ceil_row:
            dd = {ax: round(m[ax] - ceil_row[ax], 2) for ax in AXES if ax in m and ax in ceil_row}
            print(f"  P3 天井差: {dd}")
        print(f"  P4 badness(FAIL方向スコア): {bad:+.2f}")
        if a.out_slices:
            segs = battery_segments(path)
            segs.sort(key=lambda s: -seg_badness(s, cal))
            od = Path(a.out_slices)
            od.mkdir(parents=True, exist_ok=True)
            for i, s in enumerate(segs[:2]):
                t0 = int(s["_t0"])
                seg = w[t0 * SR: (t0 + 12) * SR]
                seg = seg / (np.sqrt((seg ** 2).mean()) + 1e-9) * 0.1   # RMS 揃え
                sf_path = od / f"{path.stem}_worst{i}_at{t0}.wav"
                soundfile.write(sf_path, seg, SR)
                print(f"  P4 worst{i}: {sf_path.name} (@{t0}s badness {seg_badness(s, cal):+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""f0 経路の検査(a2vc.md §5 F1): 因果 f0 の候補を非因果 pYIN(参照)と照合する。0-GPU。

候補:
  raw     artic_dsp.causal_yin(voi_max 0.45)そのまま
  fix     f0_fix.fix_f0(履歴の中央値へ折り返す・2026-10-01 レビューで固着を指摘)
  enroll  本人登録の声域(別発話の生 YIN の中央値 = 発話外の固定値)を基準に ±1 オクターブ ± tol を折り返す(履歴を使わない)

指標(両者有声のフレーム): pYIN との ±3 半音一致率・±1 半音一致率。合格 = 発話ごとの最悪値が raw 以上・10 分連結で固着しない(後半の一致率 ≥ 前半 − 0.05)。

    uv run python f0_chain_check.py --n 8 --out ../results/a2vc_design_review/f0_chain.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D
import f0_fix as FX

SR = 48000
HOP = 240


def enroll_fold(f0: np.ndarray, ref_hz: float, tol: float = 0.3) -> np.ndarray:
    out = f0.astype(np.float32).copy()
    v = out > 0
    d = np.log2(np.where(v, out, 1.0) / ref_hz)
    up = v & (np.abs(d - 1) < tol)
    dn = v & (np.abs(d + 1) < tol)
    out[up] /= 2
    out[dn] *= 2
    return out


def pyin_ref(x: np.ndarray) -> np.ndarray:
    import librosa
    x16 = librosa.resample(x.astype(np.float32), orig_sr=SR, target_sr=16000)
    f, v, _ = librosa.pyin(x16, fmin=50, fmax=1000, sr=16000, frame_length=1024, hop_length=80, center=True)
    f = np.where(v, f, 0.0)
    f = np.nan_to_num(f)
    T = len(x) // HOP
    t_src = (np.arange(len(f)) * 80) / 16000
    t_dst = (np.arange(T) * HOP + HOP / 2) / SR
    idx = np.clip(np.searchsorted(t_src, t_dst), 0, len(f) - 1)
    return f[idx]


def agree(est: np.ndarray, ref: np.ndarray, lag: int = 2) -> dict:
    e = est[lag:]
    n = min(len(e), len(ref))
    e, r = e[:n], ref[:n]
    m = (e > 0) & (r > 0)
    if m.sum() < 20:
        return {"n": int(m.sum()), "a3": float("nan"), "a1": float("nan")}
    dd = np.abs(12 * np.log2(e[m] / r[m]))
    return {"n": int(m.sum()), "a3": float((dd < 3).mean()), "a1": float((dd < 1).mean())}


def utterances(n: int) -> list[tuple[str, Path, Path]]:
    from train_ddsp_vc import index
    from train_zsvc import male_index
    spk, _, ev = index()
    fem = []
    for k in ev[:n]:
        items = spk[k]
        fem.append(("F:" + k.split("/")[-1], items[0][1], items[-1][1]))
    _, mh = male_index(False)
    by = {}
    for z, w in mh:
        by.setdefault(w.parent.name, []).append(w)
    mal = [("M:" + s, ws[0], ws[-1]) for s, ws in sorted(by.items())[:n // 2]]
    vc = Path(__file__).resolve().parent.parent / "data/vctk/VCTK-Corpus/VCTK-Corpus/wav48"
    info = (vc.parent / "speaker-info.txt").read_text().splitlines()[1:]
    males = sorted("p" + r.split()[0] for r in info if len(r.split()) > 2 and r.split()[2] == "M")
    for m in males[:: max(1, len(males) // (n - n // 2))][: n - n // 2]:
        ws = sorted((vc / m).glob("*.wav"))
        if len(ws) >= 2:
            mal.append(("V:" + m, ws[1], ws[-1]))
    return fem + mal


def main() -> int:
    from train_ddsp_vc import load48
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    cat_x, cat_ref = [], []
    for name, w, w_enr in utterances(a.n):
        x = load48(w).astype(np.float64)
        xe = load48(w_enr).astype(np.float64)
        raw = D.causal_yin(x, voi_max=0.45)[0].astype(np.float32)
        fe = D.causal_yin(xe, voi_max=0.45)[0]
        ref_hz = float(np.median(fe[fe > 0])) if (fe > 0).sum() > 20 else float(np.median(raw[raw > 0]))
        ref = pyin_ref(x)
        r = {"utt": name, "ref_hz": round(ref_hz, 1),
             "raw": agree(raw, ref), "fix": agree(FX.fix_f0(raw)[0], ref), "enroll": agree(enroll_fold(raw, ref_hz), ref)}
        rows.append(r)
        cat_x.append(x)
        cat_ref.append(ref)
        print(name, {k: (round(v["a3"], 3), round(v["a1"], 3)) for k, v in r.items() if isinstance(v, dict)}, flush=True)
    summ = {}
    for k in ("raw", "fix", "enroll"):
        a3 = [r[k]["a3"] for r in rows if np.isfinite(r[k]["a3"])]
        a1 = [r[k]["a1"] for r in rows if np.isfinite(r[k]["a1"])]
        summ[k] = {"a3_mean": round(float(np.mean(a3)), 4), "a3_worst": round(float(np.min(a3)), 4),
                   "a1_mean": round(float(np.mean(a1)), 4), "a1_worst": round(float(np.min(a1)), 4)}
    long_x = np.concatenate([x for x, (nm, _, _) in zip(cat_x, utterances(a.n)) if nm.startswith(("M", "V"))] * 6)[: 600 * SR]
    long_raw = D.causal_yin(long_x, voi_max=0.45)[0].astype(np.float32)
    long_ref = pyin_ref(long_x)
    half = len(long_raw) // 2
    stick = {}
    for k, est in (("raw", long_raw), ("fix", FX.fix_f0(long_raw)[0])):
        stick[k] = {"first_half_a3": round(agree(est[:half], long_ref[:half - 2])["a3"], 4),
                    "second_half_a3": round(agree(est[half:], long_ref[half - 2:])["a3"], 4)}
    rep = {"rows": rows, "summary": summ, "long_stream_male_sec": round(len(long_x) / SR, 1), "sticking": stick}
    print(json.dumps({"summary": summ, "sticking": stick}, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""F1(f0est2)の女声の偽有声の中身(学習なし・held21 女声)。
harvest と pYIN が両方「無声」で、F1 が「有声」と言うフレームを集め、次の 3 軸で特徴づける:
  (1) 有声区間の端からの距離(端から ≤ 3 フレーム = 境界のずれ・それより遠い = 孤立の誤検出)
  (2) フレームの音量(発話の最大から −dB・無音に近いか)
  (3) 連続長(1〜2 フレームの瞬間か・長い塊か)
    uv run python f0est_false_probe.py --ckpt ../results/f0est2/last.pt --out ../results/f0est2/false_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import f0est as FE
import nvoc as N
import train_f0est as TF


def runs(b: np.ndarray) -> list:
    out, i = [], 0
    while i < len(b):
        if b[i]:
            j = i
            while j < len(b) and b[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = FE.F0Est(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in st["cfg"].items()}).to(dev) if "cfg" in st else FE.F0Est().to(dev)
    net.load_state_dict(st["net"] if "net" in st else st["ema"])
    net.eval()
    front = FE.Front().to(dev)
    near, far, lens, db = [], [], [], []
    tot_agreed_unv = 0
    ex = []
    for it in TF.held_female():
        x = it["x"]
        est = TF.infer(front, net, x, dev)
        ref = it["f0"]
        py = TF.pyin_ref(x, len(ref))
        m = min(len(est), len(ref), len(py))
        est, ref, py = est[:m], ref[:m], py[:m]
        unv = (ref == 0) & (py == 0)
        fv = unv & (est > 0)
        tot_agreed_unv += int(unv.sum())
        vox = ref > 0
        dist = np.full(m, 999)
        idx = np.where(vox)[0]
        if len(idx):
            for t in np.where(fv)[0]:
                dist[t] = np.abs(idx - t).min()
        e = 10 * np.log10(np.convolve(x[:m * N.HOP] ** 2, np.ones(N.HOP) / N.HOP, "valid")[::N.HOP][:m] + 1e-12)
        e = e - e.max()
        for t in np.where(fv)[0]:
            (near if dist[t] <= 3 else far).append(t)
            db.append(e[min(t, len(e) - 1)])
        lens += [j - i for i, j in runs(fv)]
        ex.append({"stem": it["stem"], "false": int(fv.sum()), "frames": int(unv.sum())})
    n = len(near) + len(far)
    rep = {"agreed_unvoiced_frames": tot_agreed_unv, "false_frames": n, "false_rate": round(n / max(1, tot_agreed_unv), 4),
           "within3_of_voiced": round(len(near) / max(1, n), 3), "isolated_far": round(len(far) / max(1, n), 3),
           "run_len_frames": {"median": float(np.median(lens)), "p90": float(np.percentile(lens, 90)), "frac_le2": round(float(np.mean(np.array(lens) <= 2)), 3)},
           "level_db_below_utt_max": {"median": round(float(np.median(db)), 1), "p10": round(float(np.percentile(db, 10)), 1), "p90": round(float(np.percentile(db, 90)), 1)},
           "per_utt": ex}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

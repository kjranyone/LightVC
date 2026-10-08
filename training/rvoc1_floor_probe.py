"""R3-0(学習なし): rvoc1 100k の PESQ の上限の帰属。held21・f0 = harvest。
(1) 励起の乱数の種だけ変えた 2 出力の間の PESQ・logmel(= 乱数で決まる上限・容量では動かない)
(2) 元音声との PESQ・logmel(種 0)
(3) 条件の包絡行を因果平滑(0.25·0.5·0.25)したときの元音声との PESQ(rev3(a) の費用)
(4) 区間の種類(有声・無声・無音)ごとの logmel 誤差
    uv run python rvoc1_floor_probe.py --out ../results/rvoc1/floor_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import eval_nvoc as E
import nvoc as N
import rvoc as R
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent


def render(gen, front, it, dev, seed, smooth):
    xa = torch.from_numpy(it["xa"])[None].to(dev)
    fa = torch.from_numpy(it["f0_h"])[None].to(dev)
    cond = front(xa, fa, torch.from_numpy(it["env"])[None].to(dev))
    if smooth:
        k = torch.tensor([0.25, 0.5, 0.25], device=dev)[None, None]
        z = F.conv1d(F.pad(cond[0, :25].reshape(-1, 1, cond.shape[-1]), (2, 0), mode="replicate"), k)
        cond = cond.clone()
        cond[0, :25] = z.reshape(25, -1)
    exc = TR.excitation(fa, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(seed))
    return gen(cond, exc)[0].cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(ROOT / "results/rvoc1/snap/ema_100k.pt"))
    ap.add_argument("--env_smooth", type=float, default=None, help="省略時は ckpt の env_smooth(無ければ 0)")
    a = ap.parse_args()
    dev = "cuda"
    items = TR.held(dev)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    front = TR.Front("pae", TR.ckpt_env_smooth(st, a.env_smooth)).to(dev)
    gen = R.RVoc(ch=st["cfg"]["ch"], d_cond=st["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(st["ema"])
    gen.eval()
    rows = {"vs_orig": [], "seed_vs_seed": [], "smooth_vs_orig": []}
    cls = {"voiced": [], "unvoiced": [], "silence": []}
    with torch.no_grad():
        for it in items:
            y0, y1 = render(gen, front, it, dev, 0, False), render(gen, front, it, dev, 1, False)
            ys = render(gen, front, it, dev, 0, True)
            rows["vs_orig"].append(E.metrics(y0, it["x"], N.DELAY, dev))
            rows["seed_vs_seed"].append(E.metrics(y1, y0[N.DELAY:], N.DELAY, dev))
            rows["smooth_vs_orig"].append(E.metrics(ys, it["x"], N.DELAY, dev))
            x, y = it["x"], y0[N.DELAY:]
            n = min(len(x), len(y)) // 480 * 480
            lx = np.log(np.abs(np.fft.rfft(x[:n].reshape(-1, 480) * np.hanning(480), axis=1)) + 1e-5)
            ly = np.log(np.abs(np.fft.rfft(y[:n].reshape(-1, 480) * np.hanning(480), axis=1)) + 1e-5)
            err = np.abs(lx - ly).mean(1)
            en = lx.mean(1)
            f0 = it["f0_h"]
            v = np.array([f0[min(len(f0) - 1, i * 2)] > 0 for i in range(len(err))])
            sil = en < np.percentile(en, 10)
            cls["voiced"] += err[v & ~sil].tolist()
            cls["unvoiced"] += err[~v & ~sil].tolist()
            cls["silence"] += err[sil].tolist()
    rep = {k: {m: round(float(np.nanmean([r[m] for r in v])), 4) for m in v[0]} for k, v in rows.items()}
    rep["frame_logspec_err"] = {k: [round(float(np.mean(v)), 3), len(v)] for k, v in cls.items()}
    print(json.dumps(rep, indent=1))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

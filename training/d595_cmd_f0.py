"""D595(推論だけ): 候補の出力を F1 が倍に読むフレームで、製品の高さの指令 c = FE.decode(F1(SRC), 0.5) × 2^(k/12) もすでに倍か(入力の側)を調べる。基準の高さ r は D593 rev1 と同じ。
    uv run python d595_cmd_f0.py --sets dev2 dev3 --out ../results/conv_p0/d595/cmd.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import H, SR, harvest

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import f0est as FE
    import nvoc as N
    from d6_unit_cov import load48
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())
    HOP = N.HOP

    def run(x: np.ndarray) -> tuple:
        n = len(x) // HOP
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * HOP])[None].to(dev)], 1), n))
            dec = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        k = pr[:, :m].argmax(0)
        ii = np.clip(k[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        f = np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12))
        return np.where(vo[:m] & (c >= 0.5), f, 0.0), dec

    def prep(x: np.ndarray) -> np.ndarray:
        x = x[: int(8 * SR)].astype(np.float32)
        return x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05

    def shift(r0: np.ndarray, lag: int) -> np.ndarray:
        r = np.roll(r0, lag)
        if lag > 0:
            r[:lag] = 0
        elif lag < 0:
            r[lag:] = 0
        return r

    def best_lag(r0: np.ndarray, h: np.ndarray) -> int:
        best = (0, -1.0)
        for lag in range(-10, 11):
            r = shift(r0, lag)
            n = min(len(r), len(h))
            b = (r[:n] > 0) & (h[:n] > 0)
            if b.sum() < 50:
                continue
            ag = float((np.abs(12 * np.log2(h[:n][b] / r[:n][b])) < 1).mean())
            if ag > best[1]:
                best = (lag, ag)
        return best[0]

    tot = {k: {"up": 0, "up_cmd_up": 0, "cmd_up": 0, "nv": 0, "edge_up": 0, "on": 0, "off": 0} for k in (5, 7, 12)}
    for s in a.sets:
        S = H / s
        src: dict = {}
        for k in (5, 7, 12):
            nm = f"OURS{k}RY58G"
            for f in sorted((S / "d570" / nm).glob("p*_t*.wav")):
                key = re.match(r"(p\d+_t\d+)_", f.name).group(1)
                if key not in src:
                    xs = prep(load48(S / "d56" / f"{key}_SRC.wav"))
                    src[key] = (harvest(xs), run(xs)[1])
                hs, dec = src[key]
                x = prep(load48(f))
                fo, _ = run(x)
                lag = best_lag(hs * 2 ** (k / 12), harvest(x))
                r = shift(hs * 2 ** (k / 12), lag)
                c = shift(dec * 2 ** (k / 12), lag)
                n = min(len(fo), len(r), len(c))
                fo, r, c = fo[:n], r[:n], c[:n]
                vb = (fo > 0) & (r > 0)
                up = vb & (12 * np.log2(np.where(vb, fo, 1.0) / np.where(r > 0, r, 1.0)) > 9)
                cb = (c > 0) & (r > 0)
                cup = cb & (12 * np.log2(np.where(cb, c, 1.0) / np.where(r > 0, r, 1.0)) > 9)
                vo = fo > 0
                on = np.zeros(n, bool)
                off = np.zeros(n, bool)
                for i in np.nonzero(vo[1:] & ~vo[:-1])[0] + 1:
                    on[i: i + 4] = True
                for i in np.nonzero(vo[:-1] & ~vo[1:])[0]:
                    off[max(0, i - 3): i + 1] = True
                if vo[0]:
                    on[:4] = True
                T = tot[k]
                T["up"] += int(up.sum())
                T["up_cmd_up"] += int((up & cup).sum())
                T["cmd_up"] += int(cup.sum())
                T["nv"] += int(cb.sum())
                T["edge_up"] += int((up & (on | off)).sum())
                T["on"] += int((up & on).sum())
                T["off"] += int((up & off).sum())
            print(s, nm, tot[k], flush=True)
    rep = {"per_k": {}}
    for k, T in tot.items():
        rep["per_k"][k] = {**T, "M1": round(T["up_cmd_up"] / max(T["up"], 1), 3), "cmd_up_frac_of_voiced": round(T["cmd_up"] / max(T["nv"], 1), 4),
                           "onset_share_of_edge_up": round(T["on"] / max(T["edge_up"], 1), 3), "offset_share_of_edge_up": round(T["off"] / max(T["edge_up"], 1), 3)}
    u = sum(tot[k]["up"] for k in (5, 7))
    uc = sum(tot[k]["up_cmd_up"] for k in (5, 7))
    rep["M1_k5k7"] = round(uc / max(u, 1), 3)
    print(json.dumps(rep, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

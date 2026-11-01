"""D364: 中間の帯(約 1〜3 Hz・0.15 s 平均 − 0.5 s 平均)の音色の揺らぎを乱数(固定の統計・因果な最小位相 FIR・共通の乱数)で足すと、候補(遅い揺らぎ σ 1.5)の床は下がるか(D363: 本物の中間の帯 × 1 で SR 0.29 → −0.25)。
学習なし。統計 = results/diag_ppgvoc24/mid_stats.npz(d339_slownoise.BAND = (100, 30))。中間の帯の乱数の種 = 遅い揺らぎの種 + 7777。
    uv run python d364_midnoise.py --ckpt ../results/diag_ppgvoc24/product_s15_007500.pt --out ../results/conv_p0/d364
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def mid_noise(T: int, seed: int, M: dict, sigma: float) -> np.ndarray:
    h = M["fir"]
    k, L = h.shape
    e = P.splitmix_normals(seed, (T + L - 1) * k).reshape(T + L - 1, k).T
    w = np.stack([np.convolve(e[j], h[j], "valid") for j in range(k)]) * M["gain"][:, None]
    return (sigma * (M["B"] @ w)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=str(P.ROOT / "results/diag_ppgvoc24/mid_stats.npz"))
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--sets", nargs="+", default=["sr", "devff", "devff2"])
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    z = np.load(a.stats)
    h = P.minphase_fir(torch.from_numpy(z["amp"]).float()).double().numpy()
    M = {"B": z["B"].astype(np.float64), "fir": h, "gain": z["tgt"] / np.sqrt((h ** 2).sum(1))}
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    held = P.e2e_held()
    its = {s_: (P.eval_items(held, "table", dev)[0] if s_ == "sr" else P.ff_items([s_])) for s_ in a.sets}
    groups: dict = {s_: [] for s_ in a.sets}
    with torch.no_grad():
        for sg in a.sigmas:
            for sd in a.seeds:
                P.SLOW["seed"] = sd
                tag = f"M{sg}_r{sd}"
                for s_ in a.sets:
                    od = O / s_ / tag
                    od.mkdir(parents=True, exist_ok=True)
                    for nm, c, lg, spk in its[s_]:
                        cond, exc, s2 = P.cond_exc(g, c, lg, spk, dev)
                        T = cond.shape[-1]
                        mz = torch.from_numpy(mid_noise(T, sd + T + 7777, M, sg)).to(dev)[None]
                        cond = torch.cat([cond[:, :1], cond[:, 1:25] + mz, cond[:, 25:]], 1)
                        y = g(cond, exc, s2)[0].cpu().numpy()[N.DELAY:]
                        sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
                    groups[s_].append(f"{tag}={od}/*.wav")
                print("rendered", tag, flush=True)
    rows = {}
    for s_ in a.sets:
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups[s_], "--out", str(O / f"{s_}_v1perp.json")], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-1500:], flush=True)
        j = json.loads((O / f"{s_}_v1perp.json").read_text())
        rows[s_] = {k: v["v1perp"]["logit"] for k, v in j.items()}
    (O / "summary.json").write_text(json.dumps(rows, indent=1))
    for sg in a.sigmas:
        m = [np.mean([v for k, v in rows[s_].items() if k.startswith(f"M{sg}_")]) for s_ in a.sets]
        print(f"mid σ {sg}:", " ".join(f"{s_} {x:.3f}" for s_, x in zip(a.sets, m)), f"| 3-set {np.mean(m):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

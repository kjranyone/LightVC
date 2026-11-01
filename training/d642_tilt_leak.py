"""D642: 内容の網(diag_envnet1)が入力の声の傾きを読むかの探針(学習の分布・実女声)。入力の側に録音の経路の 2 kHz の高域の棚 −9 / 0 / +9 dB をかけ、
事後・C2・c0n を作り直して網の出力の形(c0 なし)の高域(4 kHz 以上)− 低域の dB がどれだけ付いて動くかを、入力の包絡の同じ量の動きとの比で読む。表の引きの動きも記録。
規則は results/conv_p0/d642/prereg.yaml。
    uv run python d642_tilt_leak.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import nvoc as N
import pae as PA
import train_c1 as T1
import train_ppgvoc as P
import train_rvoc as TR
from d634_artic_render import DCT
from train_envnet import WARM, EnvNet

ROOT = P.ROOT
OUT = ROOT / "results/conv_p0/d642"
GAINS = (-9.0, 0.0, 9.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=str(ROOT / "results/diag_envnet1/last.pt"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    import c1_content as CC
    import librosa
    dev = "cuda"
    torch.manual_seed(642)
    TR.SEG = 33600 * 4
    TR.T = TR.SEG // N.HOP
    SEG, T = TR.SEG, TR.T
    held = P.e2e_held()
    rows, spks, _ = P.load_rows({h[0] for h in held}, True)
    spk_idx = {s: i for i, s in enumerate(spks)}
    tabT = torch.from_numpy(np.stack([np.load(ROOT / "data/rvoc_tables_real" / f"{s}.npz")["T"] for s in spks])).float().to(dev)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"]); c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    P.cvin_setup(str(ROOT / "results/c2_1/last.pt"), dev)
    sm = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    front = TR.Front("pae", TR.ckpt_env_smooth(sm, None)).to(dev)
    st = torch.load(a.net, map_location="cpu", weights_only=False)
    net = EnvNet(st["cfg"]["ch"], tuple(st["cfg"]["dils"]), st["cfg"]["d_c2"]).to(dev).eval()
    net.load_state_dict(st["net"])
    dl = iter(torch.utils.data.DataLoader(TR.DS(rows, rows, 642, spk_idx=spk_idx, ctx=T1.PRE), batch_size=8, num_workers=8))
    D = torch.from_numpy(DCT[1:25]).float().to(dev)
    fc = librosa.mel_frequencies(130, fmin=0.0, fmax=24000.0)[1:-1]
    hi = torch.from_numpy(fc >= 4000).to(dev)

    def hilo(r: torch.Tensor, m: torch.Tensor) -> float:
        lm = torch.einsum("bdt,dm->btm", r[..., WARM:] * 10, D)[m]
        return float((20 / np.log(10)) * (lm[:, hi].mean(1) - lm[:, ~hi].mean(1)).mean())

    acc = {g: {"in": [], "out": [], "tab": []} for g in GAINS}
    for _ in range(10):
        xa, f0, _, xl, si = [t for t in next(dl)][:5]
        si = si.to(dev)
        f0d = f0.to(dev)
        m = None
        for g in GAINS:
            xp = np.stack([TR.channel_apply(x.numpy(), (0.0, g)) for x in xa])
            xlp = np.stack([TR.channel_apply(x.numpy(), (0.0, g)) for x in xl])
            env = np.stack([PA.envelope(x, f.numpy()) for x, f in zip(xp, f0)])
            xpt, xlt = torch.from_numpy(xp).to(dev), torch.from_numpy(xlp).to(dev)
            with torch.no_grad():
                ct = front(xpt, f0d, torch.from_numpy(env).to(dev))
                xc = torch.cat([xlt, xpt[:, PA.A:PA.A + SEG]], 1)
                p = P.ppg_form(c1n(mfc(xc))[..., -T:], "soft")
                c2 = P.cvin_frames(xc, T)
                tab = torch.einsum("bdk,bkt->bdt", tabT[si], p) / 10
                y = net(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], tab, c2)
                if m is None:
                    m = ct[:, 26, WARM:] > 0.5
                acc[g]["in"].append(hilo(ct[:, 1:25], m))
                acc[g]["out"].append(hilo(y, m))
                acc[g]["tab"].append(hilo(tab, m))
    mean = {g: {x: float(np.mean(v)) for x, v in acc[g].items()} for g in GAINS}
    din = mean[9.0]["in"] - mean[-9.0]["in"]
    rep = {"means": {str(g): {x: round(v, 3) for x, v in mean[g].items()} for g in GAINS}, "d_in_db": round(din, 3),
           "d_out_db": round(mean[9.0]["out"] - mean[-9.0]["out"], 3), "d_tab_db": round(mean[9.0]["tab"] - mean[-9.0]["tab"], 3)}
    rep["leak_ratio"] = round(rep["d_out_db"] / din, 3) if abs(din) > 1e-6 else None
    rep["tab_ratio"] = round(rep["d_tab_db"] / din, 3) if abs(din) > 1e-6 else None
    rep["leak"] = bool(rep["leak_ratio"] is not None and rep["leak_ratio"] >= 0.25)
    rep["net"] = a.net
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "leak.json").write_text(json.dumps(rep, indent=1))
    print("leak", rep, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

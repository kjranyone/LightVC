"""D458: 操作者の遅い揺らぎ(D451・因果 τ 4 s・利得 1.5)に、操作者の中間の帯(約 1〜3 Hz)の包絡の動きを足すと、学習の外の入力(devff2)の 8 s の窓の自然さが写し合成へ近づくか。
中間の帯 = (操作者の包絡 − 操作者の表の引き)の 0.15 s 平均 − 0.5 s 平均(D363 と同じ帯)→ 遅い揺らぎの基底へ射影。中心の窓(oracle・先読みあり)と因果な窓(過去の平均)の両方。
devff2(30 組・操作者は学習の外)と devff の部分集合 0 を候補(既定 D3 = diag_ppgvoc43)で描画 → d286_window を写し合成・入力と並べる。
    uv run python d458_opmid.py --ckpt ../results/diag_ppgvoc43/product_002500_op15.pt --out ../results/conv_p0/d458
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P

VARS = {"OPS": (0.0, ""), "MIDC05": (0.5, "c"), "MIDC10": (1.0, "c"), "MIDO10": (1.0, "o")}


def band(r: torch.Tensor, causal: bool) -> torch.Tensor:
    if causal:
        a = F.avg_pool1d(F.pad(r, (29, 0), mode="replicate"), 30, 1)
        b = F.avg_pool1d(F.pad(r, (99, 0), mode="replicate"), 100, 1)
    else:
        a = F.avg_pool1d(F.pad(r, (15, 14), mode="replicate"), 30, 1)
        b = F.avg_pool1d(F.pad(r, (50, 49), mode="replicate"), 100, 1)
    B = P.SLOW["B"].to(r.device)
    return torch.einsum("dk,nkt->ndt", B, torch.einsum("dk,ndt->nkt", B, a - b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    sub = P.ROOT / "results/conv_p0/d444_p43"
    for s, refdir in (("devff2", sub / "devff2/sub"), ("devff", sub / "devff/sub")):
        keys = sorted(f.name[:10] for f in (refdir / "CP_0").glob("p*.wav"))
        for ref in ("CP_0", "SRC_0"):
            d = O / s / ref
            d.mkdir(parents=True, exist_ok=True)
            for f in (refdir / ref).glob("p*.wav"):
                if not (d / f.name).exists():
                    (d / f.name).symlink_to(f.resolve())
        with torch.no_grad():
            for k in keys:
                ti = int(k[7:10])
                c, lg = P.load_dump(P.H / "e2e" / f"eval_{s}", k)
                spk = P.load_enroll(P.H / s / "d85_work" / f"en_{ti:03d}")
                T_op = P.op_tab(P.H / s, k)
                p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                stt = P.spk_tensor(spk, dev)
                r = ct[:, 1:25] - torch.einsum("dk,bkt->bdt", torch.from_numpy(T_op).float().to(dev), p) / 10
                zs = P.OPSLOW["gain"] * P.opslow_causal(r, P.OPSLOW["tau"])
                f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
                exc = P.TR.excitation(f0, ct.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
                for v, (gm, kind) in VARS.items():
                    z = zs if gm == 0 else zs + gm * band(r, kind == "c")
                    y = g(g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt, z), exc, stt)[0].cpu().numpy()[N.DELAY:]
                    (O / s / v).mkdir(parents=True, exist_ok=True)
                    sf.write(O / s / v / f"{k}_{v}.wav", y / (np.abs(y).max() + 1e-9) * 0.9, N.SR)
        r_ = subprocess.run([sys.executable, str(Path(__file__).parent / "d286_window.py"), "--eq_dir", str(O / s), "--out", str(O / f"{s}_window.json")], capture_output=True, text=True)
        print(s, r_.stdout[-1500:], r_.stderr[-600:] if r_.returncode else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""D658: 一本の物差し(描画と同じ式: env_full = c0 を対数の平均・正規化なし・描画の較正の曲線・発声の区間の長時間のパワー)で、目標本人の参照の録音の
REAL(実の行)/ OWN(本人の事後で引いた表)/ OWNV(表 + 単位の中の対数の揺れの分散 σ²)を並べ、音で測る参照(REF)との hf / tilt の差を出す。推論の側は D655 の T / TPN。
規則は results/conv_p0/d658/prereg.yaml。
    uv run python d658_one_ruler.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

from d631_world_swap import H, ROOT, SETS, keys
from d634_artic_render import DCT
from d651_copy_hf import best8, hf_tilt
from d653_causal_render import env_full
from d655_deficit import env_stats

OUT = ROOT / "results/conv_p0/d658"
D24 = DCT[1:25]


def main() -> int:
    import nvoc as N
    import pae as PA
    import pyworld as pw
    import scipy.signal as ss
    import train_c1 as T1
    import c1_content as CC
    from d6_unit_cov import load48
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"]); c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    recs = []
    for s in SETS:
        for ti in sorted({int(k[7:10]) for k in keys(s)}):
            x = best8(load48(H / s / "d37" / "_work" / f"ref_{ti:03d}.wav").astype(np.float64))
            Tn = len(x) // N.HOP - 2
            seg = Tn * N.HOP
            xa = np.concatenate([np.zeros(PA.A), x[:seg], np.zeros(PA.A)])
            f0, _ = pw.harvest(ss.resample_poly(x[:seg + N.HOP], 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
            f0 = np.pad(f0, (0, max(0, Tn - len(f0))))[:Tn]
            env = PA.envelope(xa, f0).astype(np.float64)
            xc = torch.from_numpy(np.concatenate([np.zeros(T1.PRE), x[:seg]]).astype(np.float32))[None].to(dev)
            with torch.no_grad():
                p = torch.softmax(c1n(mfc(xc))[..., -Tn:], 1)[0].cpu().numpy().T.astype(np.float64)
            T = np.fromfile(H / s / "d85_work" / f"en_{ti:03d}" / "enroll.bin", "<f4").reshape(24, -1).astype(np.float64)
            c0 = env[0]
            real = env[1:25].T / 10.0
            own = (p @ T.T) / 10.0
            lm_r, lm_t = (real * 10.0) @ D24, (own * 10.0) @ D24
            sh_u = (T.T @ D24)
            res2 = (lm_r[:, None, :] - sh_u[None]) ** 2
            var_u = (p[:, :, None] * res2).sum(0) / (p.sum(0)[:, None] + 1e-9)
            add = p @ var_u
            ownv = own + (add @ D24.T) / 10.0
            rec = {"s": s, "ti": ti, "REF": hf_tilt(x)}
            for a, r in (("REAL", real), ("OWN", own), ("OWNV", ownv)):
                rec[a] = env_stats(env_full(c0, r))
            recs.append(rec)
    d5 = json.loads((ROOT / "results/conv_p0/d655/deficit.json").read_text())["by_k"]
    rep: dict = {}
    for i, st in enumerate(("hf", "tilt")):
        rep[st] = {a: round(float(np.median([r[a][i] - r["REF"][i] for r in recs])), 3) for a in ("REAL", "OWN", "OWNV")}
        rep[st]["infer_T_TPN_D655"] = {k: (v[st]["T"], v[st]["TPN"]) for k, v in d5.items()}
    rep["n"] = len(recs)
    (OUT / "result.json").write_text(json.dumps({"summary": rep, "items": recs}, indent=0))
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

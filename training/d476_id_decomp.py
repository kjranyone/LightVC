"""D476(本人らしさの誤差の分析): 本人らしさ(D394 / D443 と同じ識別の余白・中心 = dev/cen_reg)がどこで失われるかを devff の自己変換の組(p01xx = 目標本人の参照に使っていないクリップを入力)で分解する。
AA_SRC = 目標本人の別のクリップ(本物)・AA_CP = その写し合成(rvoc3hi・実の包絡)・AA_CV = 系(既定 P43OP)の自己変換(操作者 = 本人・表 = 本人)・
FF_CV = 別の女声からの変換(p00xx・学習のクリップを含む)・FF_SRC = 別の女声の入力そのもの。発話ごとの余白の平均と目標でまとめたブートストラップの区間。
    uv run python d476_id_decomp.py --sys P43OP --out ../results/conv_p0/d476.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", default="P43OP")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    from transformers import WavLMForXVector
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
    import idloss as ID
    ec = ID.Ecapa(dev)

    @torch.no_grad()
    def emb(p: Path) -> dict:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(1)[: 8 * sr]
        x16 = resample_poly(x, 16000, sr).astype(np.float32)
        w = torch.nn.functional.normalize(sv(input_values=torch.from_numpy(x16)[None].to(dev)).embeddings, dim=-1)[0].cpu().numpy()
        e = ec(torch.from_numpy(x)[None].to(dev))[0].cpu().numpy()
        return {"wsv": w, "ecapa": e / np.linalg.norm(e)}

    meta = json.loads((H / "dev/cen_reg/meta.json").read_text())
    ok = sorted(m["ti"] for m in meta if m["ok"])
    C = {ti: emb(H / "dev/cen_reg" / f"cen_{ti:03d}.wav") for ti in ok}

    def margin(e: dict, ti: int) -> dict:
        out = {}
        for m in ("wsv", "ecapa"):
            cs = {t: float(e[m] @ C[t][m]) for t in C}
            out[m] = cs[ti] - np.mean([v for t, v in cs.items() if t != ti])
        return out

    groups = {"AA_SRC": ("d56", "SRC", "p01"), "AA_CP": ("d444/CP", "CP", "p01"), "AA_CV": (f"d444/{a.sys}", a.sys, "p01"),
              "FF_CV": (f"d444/{a.sys}", a.sys, "p00"), "FF_SRC": ("d56", "SRC", "p00"), "FF_CP": ("d444/CP", "CP", "p00")}
    per: dict = {}
    for g, (d, suf, pre) in groups.items():
        for f in sorted((H / "devff" / d).glob(f"{pre}*_{suf}.wav")):
            ti = int(f.name[7:10])
            if ti in C:
                per.setdefault(g, []).append((ti, margin(emb(f), ti)))
    REF = [(ti, margin(emb(H / "dev/d37/_work" / f"ref_{ti:03d}.wav"), ti)) for ti in ok]
    per["REF"] = REF
    rng = np.random.default_rng(476)
    rep: dict = {}
    for g, rows in per.items():
        tg = np.array([t for t, _ in rows])
        out = {"n": len(rows)}
        for m in ("wsv", "ecapa"):
            v = np.array([r[m] for _, r in rows])
            us = sorted(set(tg.tolist()))
            idx = {u: np.flatnonzero(tg == u) for u in us}
            bs = [v[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
            out[m] = [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        rep[g] = out
        print(g, out, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

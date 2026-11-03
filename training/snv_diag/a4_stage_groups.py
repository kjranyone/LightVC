"""A4b の fork(H_fit_partial)の指示 = 段 2 の損失の形を調べる(記録だけ・学習なし)。A0F の音素の無い発声 336 本を c1・c2(登録で 400 回当てはめ = A4 の評価)と x1・x2 で描き、
TEX の群の内訳・変調の 4 帯(4〜8・8〜16・16〜32・32〜50 Hz)の符号つきの差を 段 1 と 段 2 で並べる。出力 = results/diag_snv_a4b/stage_groups.json。
    cd training && uv run --no-sync python snv_diag/a4_stage_groups.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT, load, to16
from snv_a2_eval import vocoder
from snv_a4_eval import N0R, SYS, cluster_ci
from snv_n2c_tex import GROUPS, tex_stats
from train_snv_a2 import FEAT, G, playing
from train_snv_a4 import C, fit_phi

OUT = ROOT / "results/diag_snv_a4b"
W = {g: (s.stop - s.start) / 227 for g, s in GROUPS.items()}


def main() -> int:
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    n0r = [it for it in json.loads((N0R / "items.json").read_text()) if it["sex"] == "F"]
    names = [f"{it['voice']}_{it['cat']}_{it['k']}" for it in n0r]
    tst = {n: tex_stats(to16(*load(FEAT / f"wav44/n0r__{n}.wav"))) for n in names}
    voc = vocoder()
    rows: dict = {}
    for s, tag in SYS.items():
        d = ROOT / "results" / tag
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
        net = (C() if ck["mode"] == "c" else G("x")).to(DEV).eval()
        net.load_state_dict(ck["g"])
        nz = np.load(d / "norm.npz")
        mu, sd = nz["mu"], nz["sd"]
        S = {}
        for v in sorted({it["voice"] for it in n0r}):
            rmel = torch.from_numpy(np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32))[None].to(DEV)
            rp = torch.from_numpy((playing(FEAT / f"sparc/a1src/tref_{v}.npz", rmel.shape[2]) - mu) / sd)[None].to(DEV)
            mask = torch.ones(1, rmel.shape[2], dtype=torch.bool, device=DEV)
            if ck["mode"] == "c":
                S[v] = fit_phi(net, rp, rmel, mask)
            else:
                with torch.no_grad():
                    S[v] = net.structure(rmel, rp, mask)
        for n in names:
            p = torch.from_numpy((playing(FEAT / f"sparc/n0r/{n}.npz", np.load(FEAT / f"mel/n0r/{n}.npy").shape[1]) - mu) / sd)[None].to(DEV)
            with torch.no_grad():
                w = voc(net(p, S[n.split("_")[0]])).squeeze().cpu().numpy()
            rows[(s, n)] = tex_stats(to16(w, 44100)) - tst[n]
        del net
        torch.cuda.empty_cache()
        print("done", s, flush=True)
    grp = np.array([n.split("_")[0] for n in names])
    res: dict = {"note": "記録だけ・A4b の fork の指示", "per_sys": {}, "diff": {}}
    for s in SYS:
        D = np.stack([rows[(s, n)] for n in names])
        A = np.abs(D) / scale
        res["per_sys"][s] = {"tex": round(float(A.mean()), 4)} | {g: round(float(A[:, sl].mean()), 4) for g, sl in GROUPS.items()} | {
            "mod_signed_by_bin": [round(float(x), 4) for x in D[:, GROUPS["mod"]].reshape(len(names), 32, 4).mean((0, 1))],
            "mod_abs_by_bin": [round(float(x), 4) for x in A[:, GROUPS["mod"]].reshape(len(names), 32, 4).mean((0, 1))]}
    for a_, b_ in (("c2", "x2"), ("c1", "x1"), ("c2", "c1"), ("x2", "x1")):
        r = {}
        for g, sl in GROUPS.items():
            d = (np.abs(np.stack([rows[(a_, n)] for n in names])) / scale)[:, sl].mean(1) - (np.abs(np.stack([rows[(b_, n)] for n in names])) / scale)[:, sl].mean(1)
            r[g] = {"mean": round(float(d.mean()), 4), "weighted": round(float(d.mean() * W[g]), 4), "ci95_voice_cluster": cluster_ci(d, grp, 2000)}
        res["diff"][f"{a_}_minus_{b_}"] = r
        print(a_, b_, {g: (r[g]["mean"], r[g]["weighted"]) for g in r}, flush=True)
    for s in SYS:
        print(s, res["per_sys"][s], flush=True)
    (OUT / "stage_groups.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

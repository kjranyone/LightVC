"""P3 の切り分け(synth_nam_vc §8.4・学習なし): 変換のときに C の注意が参照を調音の状態で読んでいるか = 注意の対応。
注意(第 4 塊の後・頭ごと)が参照のフレームに置いた重みで参照の SPARC の EMA(物差しとしてだけ)を平均し、源の EMA との相関を次元ごとに取って平均する。
組 = RM(本物の男声 12 区間 → A0F 24 声・参照 j 0〜2・登録 = 操作者の先頭 30 s)・FF(A0F の声 u の j 0 → 別の声 v の参照・u ごとに 6 声)。k ∈ {0, 7}。
併記 = 頭を平均した重みでの相関・頭ごとの相関の最大・打ち切りの床(重みの並びを時刻で入れ替えた相関)・実効の支え exp(エントロピー)・最大の重み。
    cd training && uv run --no-sync python snv_diag/p3_attn_match.py snv_p1_n snv_p3_x [out=results/snv_p3/attn_match.json]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p1_eval as EV
import snv_p2_nnc as NN
import train_snv_p1 as T

ROOT = EV.ROOT
SP = ROOT / "data/snv_e1_feat/sparc"
DEV = EV.DEV
SUB = 4


@torch.no_grad()
def attention(s: EV.System, src: dict, stat: list[dict], S: dict, k: int) -> torch.Tensor:
    y = torch.from_numpy(src["y"])[None].to(DEV)
    v = torch.from_numpy(src["v"])[None].to(DEV)
    lf = torch.from_numpy(src["lf"])[None].to(DEV) + k * T.LN2_12 * v
    C = s.C
    h = C.inp(torch.cat([s.play(y, v, s.op_stat(stat)), lf[:, None], v[:, None]], 1))
    for i, blk in enumerate(C.blocks):
        h = blk(h, S["z"])
        if i == 3:
            B, _, L = h.shape
            q = C.q(h.transpose(1, 2)).view(B, L, C.heads, C.dk).transpose(1, 2)
            logit = (q @ S["k"].transpose(-1, -2)) / math.sqrt(C.dk)
            return logit.masked_fill(~S["rmask"][:, None, None, :], -1e9).softmax(-1)[0]
    raise RuntimeError("no attention block")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])]))


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x]
    out = ROOT / opt.get("out", "results/snv_p3/attn_match.json")
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts, srcs = meta["targets"], meta["sources"]
    rng = np.random.default_rng(0)

    def ema_of(rel: str, npz: Path) -> np.ndarray:
        return NN.ema_at(npz, EV.utt(rel)["y"].shape[1])

    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    tema = {v: np.concatenate([ema_of(f"a0/{v}_{j}", SP / f"a0/{v}_{j}.npz") for j in (0, 1, 2)]) for v in tgts}
    mu_op = {sp: [EV.utt(f"real/{sp}_enroll")] for sp in ("uirou", "namikawa")}
    queries = [("RM", s_, EV.utt(f"real/{s_}"), ema_of(f"real/{s_}", SP / f"a1src/{s_}.npz"), mu_op[s_.rsplit("_", 1)[0]], tgts) for s_ in srcs]
    for i, u in enumerate(tgts):
        others = [tgts[(i + d) % len(tgts)] for d in (1, 5, 9, 13, 17, 21)]
        queries.append(("FF", u, trefs[u][0], tema[u][:trefs[u][0]["y"].shape[1]], trefs[u], others))
    res: dict = {"prereg": "results/snv_p3/prereg.yaml(H_P3_none / partial の切り分け: 注意の統計と NNC を並べる)", "ruler": "SPARC multi の EMA 12 次元(物差しとしてだけ)", "systems": {}}
    for tag in tags:
        s = EV.System(tag, None)
        row: dict = {}
        for k in (0, 7):
            acc: dict = {}
            for grp, name, src, qe, stat, tg in queries:
                vm = src["v"] > 0
                vm[np.arange(len(vm)) % SUB != 0] = False
                for v in tg:
                    S = s.structure(f"t|{v}", trefs[v])
                    p = attention(s, src, stat, S, k)[:, vm].cpu().numpy()
                    re = tema[v]
                    pa = p @ re
                    q = qe[vm]
                    per_head = [corr(pa[h], q) for h in range(p.shape[0])]
                    pm = pa.mean(0)
                    perm = rng.permutation(len(q))
                    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)
                    d = acc.setdefault(grp, {"corr_mean_heads": [], "corr_best_head": [], "shuffled": [], "support": [], "max_w": [], "ref_len": []})
                    d["corr_mean_heads"].append(corr(pm, q)), d["corr_best_head"].append(max(per_head)), d["shuffled"].append(corr(pm[perm], q))
                    d["support"].append(float(np.exp(ent).mean())), d["max_w"].append(float(p.max(-1).mean())), d["ref_len"].append(p.shape[-1])
            for grp, d in acc.items():
                row[f"{grp}_k{k}"] = {m: round(float(np.mean(x)), 4) for m, x in d.items()} | {"n_pairs": len(d["shuffled"])}
                print(tag, grp, k, json.dumps(row[f"{grp}_k{k}"]), flush=True)
        res["systems"][tag] = row
        del s
        torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

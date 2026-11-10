"""段 I8(snv_i8_m13)の声を跨ぐ内容の崩れの機構(学習なし): C が目標の参照を読む注意(4 段目・鍵 = 参照の演奏・値 = 参照の Y)の鋭さを、自己の reamp と声を跨ぐ変換で並べる。
条件は snv_diag/i8_text_probe.py と同じ選び方(unseen_self・unseen_mF_k0・real_self・real_F_k0)。測る量 = 頭ごとの最大の重みの平均・実効のフレーム数 1 / Σw² の中央値・
値の読み出しの寄与 = ‖o(ctx)‖ / ‖h‖(注意が C の隠れの状態に足す量の比)。
    cd training && uv run --no-sync python snv_diag/i8_attn_probe.py [snv_p3_x snv_i8_m13] [n=6] [out=results/snv_i8/attn_probe.json]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p1_eval as EV
import train_snv_p1 as T
from snv_diag.i8_text_probe import texts_by_voice

ROOT = EV.ROOT
DEV = EV.DEV


@torch.no_grad()
def attn_stats(C: T.Circ, a: torch.Tensor, lf: torch.Tensor, v: torch.Tensor, s: dict) -> dict:
    h = C.inp(torch.cat([a, lf[:, None], v[:, None]], 1))
    for i, blk in enumerate(C.blocks[:4]):
        h = blk(h, s["z"])
    B, Ch, L = h.shape
    q = C.q(h.transpose(1, 2)).view(B, L, C.heads, C.dk).transpose(1, 2)
    lg = (q @ s["k"].transpose(-1, -2)) / C.dk ** 0.5
    lg = lg.masked_fill(~s["rmask"][:, None, None, :], float("-inf"))
    w = lg.softmax(-1)
    ctx = (w @ s["v"]).transpose(1, 2).reshape(B, L, C.heads * C.dv)
    add = C.o(ctx).transpose(1, 2)
    m = v[0] > 0
    wm = w[0][:, m]
    return {"max_w": float(wm.max(-1).values.mean()), "eff_frames": float((1.0 / (wm ** 2).sum(-1)).median()),
            "add_ratio": float((add[0][:, m].norm(dim=0) / h[0][:, m].norm(dim=0).clamp(min=1e-6)).mean()), "ref_frames": int(s["rmask"].sum())}


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x] or ["snv_p3_x", "snv_i8_m13"]
    n = int(opt.get("n", 6))
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    tv = texts_by_voice()
    spk_of: dict = {}
    for vv in split["train"]:
        for _, t in tv.get(vv, []):
            spk_of.setdefault(t, set()).add(vv)
    rng = random.Random(0)
    val = split["val"][:]
    rng.shuffle(val)
    selfs = []
    for vv in val:
        un = [u for u in tv.get(vv, []) if u[1] not in spk_of]
        if un and len(selfs) < n:
            selfs.append((vv, rng.choice(un)))
    males = [vv for vv in split["val"] if vv.startswith("em")]
    mf = []
    for i in range(n):
        vv = males[i % len(males)]
        un = [u for u in tv.get(vv, []) if u[1] not in spk_of and all(u[0] != x[1][0] for x in mf)]
        if un:
            mf.append((vv, rng.choice(un)))
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts, srcs = meta["targets"][:2], meta["sources"][:n]
    trefs = {t: [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)] for t in tgts}
    enroll = {sp: [EV.utt(f"real/{sp}_enroll")] for sp in ("uirou", "namikawa")}
    stat_of = lambda vv, rel: [EV.utt(r) for r, _ in tv[vv] if r.startswith("e1/") and r != rel][:3]
    trials = []
    for vv, (rel, _) in selfs:
        st = stat_of(vv, rel)
        trials.append(("unseen_self", EV.utt(rel), st, None, vv))
    for vv, (rel, _) in mf:
        st = stat_of(vv, rel)
        for t in tgts:
            trials.append(("unseen_mF_k0", EV.utt(rel), st, t, vv))
    for s_ in srcs:
        u = EV.utt(f"real/{s_}")
        trials.append(("real_self", u, enroll[s_.rsplit("_", 1)[0]], None, s_))
        for t in tgts:
            trials.append(("real_F_k0", u, enroll[s_.rsplit("_", 1)[0]], t, s_))
    gen = EV.load_gen()
    res: dict = {"tags": tags, "n": n, "conditions": {}}
    for tag in tags:
        s = EV.System(tag, gen)
        rows = []
        for cond, u, st, t, who in trials:
            S = s.structure(f"self|{who}", st) if t is None else s.structure(f"t|{t}", trefs[t])
            y = torch.from_numpy(u["y"])[None].to(DEV)
            vv = torch.from_numpy(u["v"])[None].to(DEV)
            lf = torch.from_numpy(u["lf"])[None].to(DEV)
            a = s.play(y, vv, s.op_stat(st))
            r = attn_stats(s.C, a, lf, vv, S) | {"cond": cond}
            rows.append(r)
            print(tag, json.dumps(r), flush=True)
        res["conditions"][tag] = {c: {k: round(float(np.mean([r[k] for r in rows if r["cond"] == c])), 4) for k in ("max_w", "eff_frames", "add_ratio", "ref_frames")}
                                  for c in dict.fromkeys(r["cond"] for r in rows)}
    print(json.dumps(res["conditions"], indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i8/attn_probe.json")).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

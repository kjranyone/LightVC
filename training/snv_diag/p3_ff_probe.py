"""P3 の切り分け(synth_nam_vc §8.4・学習なし): 演奏が性別の中で対応するとき(女声 → 別の女声)、C は目標の声に着地するか。
A0F の声 u の j 3 の演奏 → 別の声 v の参照(j 0〜2)・k = 0・ECAPA + LDA の 24 声中の 1 位(登録 = j 0〜2 の元の音)。
M_FF = 1 位が v の率・src_FF = 1 位が u の率(演奏の残りの声の写し)・他 = それ以外(どちらでもない声に着地)。v ごとに u を 3 声(24 × 3 = 72 本)。
mode = 演奏の入力(u の源と登録)だけを登録の定数で v の側へずらす: per = 周期性 4 帯の有声の平均・env = 包絡 c1..24 の有声の平均・both = 両方(高さ・有声・C の出力の経路は元のまま)。
warpα = 演奏の入力(源と登録)の包絡を一律の伸縮 α で(乱しの族が性別の中の写しに届くか)。
    cd training && uv run --no-sync python snv_diag/p3_ff_probe.py snv_p1_n snv_p3_x [mode=none,per,env,both] [out=results/...json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p1_eval as EV

ROOT = EV.ROOT
DEV = EV.DEV


def vmean(us: list[dict], lo: int, hi: int) -> np.ndarray:
    return np.concatenate([u["y"][lo:hi, u["v"] > 0] for u in us], 1).mean(1)


def shifted(u: dict, d: dict) -> dict:
    y = u["y"].copy()
    vm = u["v"] > 0
    for (lo, hi), dv in d.items():
        y[lo:hi, vm] += dv[:, None]
    y[25:29] = np.clip(y[25:29], 0.0, 1.5) * vm[None]
    return u | {"y": y}


def main() -> int:
    tags = [x for x in sys.argv[1:] if "=" not in x]
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    modes = opt.get("mode", "none").split(",")
    from snv_a0_measure import load, to16
    from snv_a0b_measure import proj
    from snv_a4_eval import cluster_ci
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    gen = EV.load_gen()
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (Tz["mu"], Tz["scalings"])

    def emb(w16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"]
    Et = {v: emb(np.concatenate([lw(EV.A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)])) for v in tgts}
    top = lambda e: max(tgts, key=lambda u: float(e @ Et[u]))
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    play = {v: EV.utt(f"a0/{v}_3") for v in tgts}
    pairs = [(tgts[(i + d) % len(tgts)], v) for i, v in enumerate(tgts) for d in (1, 8, 15)]
    res: dict = {"probe": "P3 の切り分け: 女声 → 別の女声(results/snv_p3/verdict.yaml の次)", "pairs": len(pairs), "systems": {}}
    vec = {}
    span = {"per": [(25, 29)], "env": [(1, 25)], "both": [(1, 25), (25, 29)], "none": []}
    for tag in tags:
        s = EV.System(tag, gen)
        for mode in modes:
            hit, src, grp = [], [], []
            for u, v in pairs:
                if mode.startswith("warp"):
                    al = float(mode[4:])
                    pu, ru = EV.warp_utt(play[u], al), [EV.warp_utt(r, al) for r in trefs[u]]
                else:
                    d = {sp: vmean(trefs[v], *sp) - vmean(trefs[u], *sp) for sp in span[mode]}
                    pu, ru = (shifted(play[u], d), [shifted(r, d) for r in trefs[u]]) if d else (play[u], trefs[u])
                w = s.convert(pu, ru, s.structure(f"t|{v}", trefs[v]), 0)
                t1 = top(emb(trim(w16(w))))
                hit.append(t1 == v), src.append(t1 == u), grp.append(v)
            key = tag if mode == "none" else f"{tag}|{mode}"
            vec[key] = (np.array(hit, float), np.array(grp))
            row = {"M_FF": round(float(np.mean(hit)), 4), "src_FF": round(float(np.mean(src)), 4), "other": round(1 - float(np.mean(hit)) - float(np.mean(src)), 4)}
            res["systems"][key] = row
            print(key, json.dumps(row), flush=True)
        del s
        torch.cuda.empty_cache()
    if len(tags) > 1 and modes == ["none"]:
        a_, g_ = vec[tags[1]]
        b_, _ = vec[tags[0]]
        res["diff"] = {"mean": round(float((a_ - b_).mean()), 4), "ci95_target_cluster": cluster_ci(a_ - b_, g_)}
        print("diff", json.dumps(res["diff"]), flush=True)
    out = ROOT / opt.get("out", "results/snv_p3/ff_probe.json" if modes == ["none"] else f"results/snv_p3/ff_probe_{'_'.join(tags)}_{'_'.join(modes)}.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

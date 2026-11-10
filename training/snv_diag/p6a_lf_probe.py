"""P6a の探り(synth_nam_vc §8・学習なし・results/snv_p6a/prereg.yaml): C の入力の絶対の高さ lf は写しの道か。
C に渡す lf だけを登録の定数で別の声域へ置く(描画器の lf と励振の f0 は運用 = 操作者 × 2^(k/12))。
none = 今の評価・rel = 操作者の lf − μ_操作者 + μ_目標・decoy(女声 → 女声だけ)= 第 3 の声 w の域へ。μ = 登録の有声のフレームの lf の平均。
dev = snv_p1_eval.py の dev と同じ組(k = 7)・ff = p3_ff_probe.py と同じ 72 組(k = 0)。
    cd training && uv run --no-sync python snv_diag/p6a_lf_probe.py snv_p3_x snv_p3e_x [out=results/snv_p7/lf_probe.json]
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
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV
ST = 12 / math.log(2.0)


def mu_lf(us: list[dict]) -> float:
    return float(np.concatenate([u["lf"][u["v"] > 0] for u in us]).mean())


def main() -> int:
    tags = [x for x in sys.argv[1:] if "=" not in x]
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    from snv_a0_measure import load, to16
    from snv_a0b_measure import proj
    from snv_a4_eval import cluster_ci
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    gen = EV.load_gen()
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (Tz["mu"], Tz["scalings"])

    def emb(w16: np.ndarray, lda: bool = True) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        e = torch.nn.functional.normalize(e, dim=0).cpu().numpy()
        return proj(e, Tt) if lda else e

    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)
    res: dict = {"prereg": "results/snv_p6a/prereg.yaml", "dev": {}, "ff": {}}
    split = json.loads((ROOT / "results" / tags[0].partition("@")[0] / "split.json").read_text())
    tg = [v for v in split["val"] if v.startswith("ef")]
    spk = sorted({p.stem.split("_")[0] for p in (EV.FEAT / "vctk").glob("*.npz")})
    srcs, enr = [], {}
    for s_ in spk:
        fs = sorted((EV.FEAT / "vctk").glob(f"{s_}_*.npz"))
        enr[s_] = [T.load_utt(f) for f in fs[:2]]
        srcs += [(s_, T.load_utt(f)) for f in fs[2:4]]
    Et = {v: emb(np.concatenate([lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 4, 5)]), lda=False) for v in tg}
    drefs = {v: [EV.utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)] for v in tg}
    mu_d = {v: mu_lf(drefs[v]) for v in tg}
    mu_e = {s_: mu_lf(enr[s_]) for s_ in spk}
    res["dev_shift_semitones"] = {"mean": round(float(np.mean([(mu_d[v] - mu_e[s_]) * ST for s_, _ in srcs for v in tg])), 2),
                                  "k7_minus_rel_mean": round(float(np.mean([7 - (mu_d[v] - mu_e[s_]) * ST for s_, _ in srcs for v in tg])), 2)}
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"]
    Ef = {v: emb(np.concatenate([lw(EV.A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)])) for v in tgts}
    top = lambda e: max(tgts, key=lambda u: float(e @ Ef[u]))
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    mu_f = {v: mu_lf(trefs[v]) for v in tgts}
    play = {v: EV.utt(f"a0/{v}_3") for v in tgts}
    pairs = [(tgts[(i + d) % len(tgts)], v, tgts[(i + 12) % len(tgts)]) for i, v in enumerate(tgts) for d in (1, 8, 15)]
    res["ff_shift_semitones"] = {"rel_abs_mean": round(float(np.mean([abs(mu_f[v] - mu_f[u]) * ST for u, v, _ in pairs])), 2),
                                 "decoy_abs_mean": round(float(np.mean([abs(mu_f[w] - mu_f[u]) * ST for u, _, w in pairs])), 2)}
    print("shifts", json.dumps({k: res[k] for k in ("dev_shift_semitones", "ff_shift_semitones")}), flush=True)
    for tag in tags:
        s = EV.System(tag, gen)
        s_, src = srcs[0]
        S = s.structure(f"dev|{tg[0]}", drefs[tg[0]])
        y0, y1 = s.convert(src, enr[s_], S, 7, with_y=True)[1], s.convert(src, enr[s_], S, 7, with_y=True, dlf_c=7 * T.LN2_12)[1]
        assert torch.allclose(y0, y1, atol=1e-5), "C の lf の差し替えが恒等で一致しない"
        hits, grp = {"none": [], "rel": []}, []
        for s_, src in srcs:
            for v in tg:
                grp.append(v)
                S = s.structure(f"dev|{v}", drefs[v])
                for mode in hits:
                    w = s.convert(src, enr[s_], S, 7, dlf_c=None if mode == "none" else mu_d[v] - mu_e[s_])
                    e = emb(trim(w16(w)), lda=False)
                    hits[mode].append(max(tg, key=lambda u: float(e @ Et[u])) == v)
        a_, b_ = np.array(hits["rel"], float), np.array(hits["none"], float)
        res["dev"][tag] = {"M_k7_none": round(float(b_.mean()), 4), "M_k7_rel": round(float(a_.mean()), 4),
                           "dM": {"mean": round(float((a_ - b_).mean()), 4), "ci95_target_cluster": cluster_ci(a_ - b_, np.array(grp))}, "n": len(grp)}
        print("dev", tag, json.dumps(res["dev"][tag]), flush=True)
        rows = {m: {"hit": [], "src": [], "w": []} for m in ("none", "rel", "decoy")}
        fgrp = []
        for u, v, w_ in pairs:
            fgrp.append(v)
            S = s.structure(f"t|{v}", trefs[v])
            for mode, d in (("none", None), ("rel", mu_f[v] - mu_f[u]), ("decoy", mu_f[w_] - mu_f[u])):
                t1 = top(emb(trim(w16(s.convert(play[u], trefs[u], S, 0, dlf_c=d)))))
                rows[mode]["hit"].append(t1 == v), rows[mode]["src"].append(t1 == u), rows[mode]["w"].append(t1 == w_)
        ff = {m: {"M_FF": round(float(np.mean(r["hit"])), 4), "src_FF": round(float(np.mean(r["src"])), 4), "w_FF": round(float(np.mean(r["w"])), 4)} for m, r in rows.items()}
        for m in ("rel", "decoy"):
            for key in ("hit", "src", "w"):
                a_, b_ = np.array(rows[m][key], float), np.array(rows["none"][key], float)
                ff[f"d_{key}_{m}"] = {"mean": round(float((a_ - b_).mean()), 4), "ci95_target_cluster": cluster_ci(a_ - b_, np.array(fgrp))}
        res["ff"][tag] = ff
        print("ff", tag, json.dumps(ff), flush=True)
        del s
        torch.cuda.empty_cache()
    out = ROOT / opt.get("out", "results/snv_p6a/lf_probe.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

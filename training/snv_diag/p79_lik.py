"""段 P79 step b(results/snv_p79/prereg.yaml・学習なし・ASR なし): 族の符号帳の下の尤度の量(最も近い級までの距離・事後確率のエントロピー・最大の事後確率・級の占有の KL)は、崩れる源を分けるか。
声のフレーム = 有声かつ声の水準 − 20 dB 以上。CER と写しの R は既に在る行(results/snv_p78/reread.json・results/snv_p79/post_frames.json)。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=4 uv run --no-sync python snv_diag/p79_lik.py [sys=snv_p30_c] [out=results/snv_p79/lik.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p18_align as AL
import p78_reread as R78
import snv_cmn as CM
import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV
KEYS = ("dmin", "ent", "pmax", "occ_kl")


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag = opt.get("sys", "snv_p30_c")
    out = ROOT / opt.get("out", "results/snv_p79/lik.json")
    old = {(r["set"], r["src"]): r for r in json.loads((ROOT / "results/snv_p78/reread.json").read_text())["rows"]}
    post = json.loads((ROOT / "results/snv_p79/post_frames.json").read_text())
    sy = EV.System(tag, EV.load_gen())
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    sets = {nm: [x for x in its if x["valid"] and (nm, x["src"]) in old] for nm, its in AL.build_sets(["cv", "x:d_clean"], 64, split).items()}
    rows, occ = [], {}
    for nm, its in sets.items():
        for it in its:
            u, enr = it["u"], it["enr"][:3]
            x = librosa.load(str(T.FEAT / f"cv/{it['src']}.wav") if nm == "cv" else it["path"], sr=P.SR, mono=True)[0].astype(np.float64)
            Tn = len(u["v"])
            xp = np.concatenate([x, np.zeros(960)])
            e = np.array([10 * np.log10((xp[t * P.H: t * P.H + 960] ** 2).mean() + 1e-12) for t in range(Tn)])
            m = torch.from_numpy((u["v"] > 0.5) & (e >= np.percentile(e, 95) - 20)).to(DEV)
            y, v = torch.from_numpy(u["y"])[None].to(DEV), torch.from_numpy(u["v"])[None].to(DEV)
            st = sy.op_stat(enr)
            with torch.no_grad():
                xe = torch.einsum("bij,bjt->bit", st[1], sy.Pl.stack(sy.Pl.ctx(y, v), st[0])).transpose(1, 2)[0].float()[m]
                c = sy.Pl.cb.float()
                d = (xe * xe).sum(-1, keepdim=True) - 2 * xe @ c.T + (c * c).sum(-1)
                pr = CM.posterior(xe, c, sy.Pl.cb_s2)
            occ[(nm, it["src"])] = pr.mean(0).cpu().numpy()
            rows.append({"set": nm, "spk": it["spk"], "src": it["src"], "n": int(m.sum()), "dmin": float((d.min(-1).values / sy.Pl.cb_s2).mean()),
                         "ent": float(-(pr * pr.clamp(min=1e-12).log()).sum(-1).mean()), "pmax": float(pr.max(-1).values.mean())})
        print(nm, "done", flush=True)
    prior = np.mean([occ[(r["set"], r["src"])] for r in rows if r["set"] != "cv"], 0) + 1e-6
    prior /= prior.sum()
    for r in rows:
        q = occ[(r["set"], r["src"])] + 1e-6
        q /= q.sum()
        r["occ_kl"] = float((q * np.log(q / prior)).sum())
    cv = [r for r in rows if r["set"] == "cv"]
    pc = [r for r in post if r["set"] == "cv"]
    assert len(cv) == len(pc) and all(a["spk"] == b["spk"] for a, b in zip(cv, pc))
    C, Rh, spk = np.array([r["cer_x"] for r in pc]), np.array([r["R_hi"] for r in pc]), np.array([r["spk"] for r in cv])
    us = sorted(set(spk))
    g = np.random.default_rng(0)
    idx = [np.concatenate([np.flatnonzero(spk == us[j]) for j in g.integers(0, len(us), len(us))]) for _ in range(2000)]
    sp = lambda a, b: float(np.corrcoef(R78.rank(a), R78.rank(b))[0, 1])
    o = np.argsort(C)
    k3 = len(o) // 3
    res = {"prereg": "results/snv_p79/prereg.yaml(step_b)", "sys": tag, "s2": float(sy.Pl.cb_s2), "量": {}}
    for k in KEYS:
        z = np.array([r[k] for r in cv])
        bs = [sp(z[ix], C[ix]) for ix in idx]
        res["量"][k] = {"CER との順位相関": round(sp(z, C), 3), "区間": [round(float(np.percentile(bs, q)), 3) for q in (2.5, 97.5)], "R_hi との順位相関": round(sp(z, Rh), 3),
                       "CER の下 1/3・中・上 1/3 の平均": [round(float(z[ix].mean()), 4) for ix in (o[:k3], o[k3:2 * k3], o[2 * k3:])], "族の中央値": round(float(np.median([r[k] for r in rows if r["set"] != "cv"])), 4)}
    out.write_text(json.dumps(res | {"rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

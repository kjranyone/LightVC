"""段 P85 step a(results/snv_p85/prereg.yaml・学習なし・ASR なし): 声のフレームの符号の読みの揺れは、追跡器の成分(有声・周期性)と包絡の成分のどちらが運ぶか。
源の特徴と、写し(出力部を通して前段で読み直した特徴)の間で、成分を 1 つずつ差し替えて符号の読み(PlayW.feat)の余弦を出す。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=4 uv run --no-sync python snv_diag/p85_swap.py [sys=snv_p30_c] [out=results/snv_p85/swap.json]
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
import i13_bgm_dev_probe as I13
import p18_align as AL
import p78_reread as R78
import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV


def shifted(z: np.ndarray, lag: int, n: int) -> np.ndarray:
    idx = np.clip(np.arange(n) + lag, 0, z.shape[-1] - 1)
    return z[..., idx]


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag = opt.get("sys", "snv_p30_c")
    out = ROOT / opt.get("out", "results/snv_p85/swap.json")
    post = {(r["set"], i): r for nm in ("cv", "x:d_clean") for i, r in enumerate([q for q in json.loads((ROOT / "results/snv_p79/post_frames.json").read_text()) if q["set"] == nm])}
    old = {(r["set"], r["src"]) for r in json.loads((ROOT / "results/snv_p78/reread.json").read_text())["rows"]}
    sy = EV.System(tag, EV.load_gen())
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    sets = {nm: [x for x in its if x["valid"] and (nm, x["src"]) in old] for nm, its in AL.build_sets(["cv", "x:d_clean"], 64, split).items()}
    rows = []
    for nm, its in sets.items():
        for i, it in enumerate(its):
            u, enr, pr = it["u"], it["enr"][:3], post[(nm, i)]
            assert pr["spk"] == it["spk"]
            x = librosa.load(str(T.FEAT / f"cv/{it['src']}.wav") if nm == "cv" else it["path"], sr=P.SR, mono=True)[0].astype(np.float64)
            Tn = len(u["v"])
            xp = np.concatenate([x, np.zeros(960)])
            e = np.array([10 * np.log10((xp[t * P.H: t * P.H + 960] ** 2).mean() + 1e-12) for t in range(Tn)])
            uc = I13.utt_of(np.asarray(sy.render(u, torch.from_numpy(u["y"])[None].to(DEV), 0), np.float64))
            st = sy.op_stat(enr)
            a, va = R78.feat(sy, u, st)
            b, vb = R78.feat(sy, uc, st)
            n = min(len(va), len(vb))
            m0 = (va[:n] & vb[:n]).float()
            lag = int(np.argmax([R78.cosm(a, b, m0, g) for g in range(-4, 5)])) - 4
            yc, vc = shifted(uc["y"], lag, Tn), shifted(uc["v"], lag, Tn)
            m = (torch.from_numpy((u["v"] > 0.5) & (vc > 0.5) & (e >= np.percentile(e, 95) - 20)).to(DEV)).float()
            mix = lambda env_from, pv_from, v: R78.feat(sy, {"y": np.concatenate([env_from[:25], pv_from[25:]], 0).astype(np.float32), "v": v.astype(np.float32)}, st)[0]
            f_c = mix(yc, yc, vc)
            f_pv = mix(u["y"], yc, vc)
            f_env = mix(yc, u["y"], u["v"])
            ylo = np.concatenate([yc[:9], u["y"][9:]], 0)
            yhi = np.concatenate([u["y"][:9], yc[9:25], u["y"][25:]], 0)
            f_elo, f_ehi = mix(ylo, u["y"], u["v"]), mix(yhi, u["y"], u["v"])
            mk = m.cpu().numpy() > 0.5
            dd = np.abs(u["y"][:25, :Tn] - yc[:25])[:, mk]
            rows.append({"set": nm, "spk": it["spk"], "cer_x": pr["cer_x"], "R_hi_p79": pr["R_hi"], "n": int(m.sum()),
                         "R_full": round(R78.cosm(a, f_c, m), 4), "R_pv": round(R78.cosm(a, f_pv, m), 4), "R_env": round(R78.cosm(a, f_env, m), 4), "R_elo": round(R78.cosm(a, f_elo, m), 4), "R_ehi": round(R78.cosm(a, f_ehi, m), 4),
                         "d_lo": round(float(dd[1:9].mean()), 4), "d_mid": round(float(dd[9:17].mean()), 4), "d_hi": round(float(dd[17:25].mean()), 4)})
        print(nm, "done", flush=True)
    res: dict = {"prereg": "results/snv_p85/prereg.yaml(step_a)", "sys": tag}
    sp = lambda p, q: float(np.corrcoef(R78.rank(np.asarray(p)), R78.rank(np.asarray(q)))[0, 1])
    for nm in sets:
        rs = [r for r in rows if r["set"] == nm]
        res[nm] = {"源": len(rs), "中央値": {k: round(float(np.median([r[k] for r in rs])), 4) for k in ("R_full", "R_pv", "R_env", "R_elo", "R_ehi", "d_lo", "d_mid", "d_hi")}}
    cv = [r for r in rows if r["set"] == "cv"]
    Rf, C = np.array([r["R_full"] for r in cv]), np.array([r["cer_x"] for r in cv])
    o = np.argsort(Rf)
    k3 = len(o) // 3
    res["cv の R_full の下 1/3・中・上 1/3"] = {k: [round(float(np.mean([cv[i][k] for i in ix])), 4) for ix in (o[:k3], o[k3:2 * k3], o[2 * k3:])] for k in ("R_full", "R_pv", "R_env", "R_elo", "R_ehi", "d_lo", "d_mid", "d_hi", "cer_x")}
    res["CER との順位相関"] = {k: round(sp([r[k] for r in cv], C), 3) for k in ("R_full", "R_pv", "R_env", "R_elo", "R_ehi", "d_lo", "d_mid", "d_hi", "R_hi_p79")}
    out.write_text(json.dumps(res | {"rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

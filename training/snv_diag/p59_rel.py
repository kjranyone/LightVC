"""段 P59 step a(results/snv_p59/prereg.yaml・学習なし・ASR なし): 番地の信頼を、操作者の側の量だけで測れるか。
devA の 1〜64 人目の有効な源(足さない / 伴奏 10 dB / 0 dB・p37_cv_music と同じ混ぜ)で、研究の形(snv_p30_c+mask:snv_p45_gate_t80)の前段を通した特徴から、源ごとの量を出す(有声のフレームの平均):
  e_self = ‖y − C(a, 操作者)‖(c1..c24・包絡の単位)・d_cb = 白くした符号(級の正規化の前)と符号帳の最も近い中心の距離の 2 乗 ÷ 次元・h_cb = 級の事後確率のエントロピー・p_max = 事後確率の最大・go = 門の開く率。
出力 = results/snv_p59/rel.json(源 × 条件の行)。変換の CER との関係は scratchpad の p59_a.py で読む。
    cd training && OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p59_rel.py [n=64] [conds=mus10,mus0] [stat=enr,self,both(段 P62: 白くする式と級の正規化の統計を取る元)] [out=results/snv_p59/rel.json]
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i13_bgm_dev_probe as B
import snv_cmn as CM
import snv_p1_eval as EV
import train_snv_p1 as T
import train_snv_sep as SEP

ROOT = EV.ROOT
DEV = EV.DEV
TAG = "snv_p30_c+mask:snv_p45_gate_t80"


@torch.no_grad()
def signals(sy: EV.System, u: dict, enr: list[dict], So: dict, stat: str = "enr") -> dict:
    src = sy.prep(u)
    y = torch.from_numpy(src["y"])[None].to(DEV)
    v = torch.from_numpy(src["v"])[None].to(DEV)
    lf0 = torch.from_numpy(src["lf"])[None].to(DEV)
    st = sy.op_stat(enr if stat == "enr" else [u] if stat == "self" else enr + [u])
    Pl = sy.Pl
    xe = torch.einsum("bij,bjt->bit", st[1], Pl.stack(Pl.ctx(y, v), st[0])).transpose(1, 2)[0]
    c = Pl.cb
    d = (xe * xe).sum(-1, keepdim=True) - 2 * xe @ c.T + (c * c).sum(-1)
    p = torch.softmax(-d / (2 * Pl.cb_s2), -1)
    h = -(p * p.clamp(min=1e-12).log()).sum(-1)
    e = (10 * (y[:, 1:25] - sy.C(Pl(y, v, st), lf0, v, So)[:, 1:25])).pow(2).mean(1).sqrt()[0]
    m = v[0] > 0.5
    f = lambda t: round(float(t[m].mean()), 4) if bool(m.any()) else None
    return {"e_self": f(e), "d_cb": f(d.min(-1).values / xe.shape[-1]), "h_cb": f(h), "p_max": f(p.max(-1).values), "go": round(float(src["gate"].mean()), 3), "voiced": round(float(m.float().mean()), 3)}


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    n = int(opt.get("n", 64))
    conds = [c for c in opt.get("conds", "mus10,mus0").split(",") if c]
    stats = opt.get("stat", "enr").split(",")
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())["ref"]
    pool = SEP.bg_split()["dev"]["music"]
    load = lambda name: librosa.load(str(T.FEAT / f"cv/{name}.wav"), sr=B.SR, mono=True)[0].astype(np.float64)
    sy = EV.System(TAG, EV.load_gen())
    rows = []
    for i, s in enumerate([x for x in man["speakers"] if x["split"] == "devA"][:n]):
        sp = s["speaker"]
        srcs = [f"{sp}_{c['name']}" for c in s["src"][:2] if f"{sp}_{c['name']}" in dv and dv[f"{sp}_{c['name']}"]["text_vs_orig"] <= 0.1]
        if not srcs:
            continue
        xe = [load(f"{sp}_{c['name']}") for c in s["enroll"]]
        mus = pool[zlib.crc32(sp.encode()) % len(pool)]
        for cond in ["clean"] + conds:
            mixp = (lambda x, seed: x) if cond == "clean" else (lambda x, seed: B.with_music(x, mus, seed, False, float(cond[3:]))[1])
            enr = [B.utt_of(mixp(x, 7 + j)) for j, x in enumerate(xe)]
            So = sy.structure(f"self|{sp}|{cond}", enr)
            for name in srcs:
                u = B.utt_of(mixp(load(name), 1 + i))
                for stn in stats:
                    rows.append({"spk": sp, "idx": i, "src": name, "cond": cond, "stat": stn} | signals(sy, u, enr, So, stn))
        sy.cache = {q: c for q, c in sy.cache.items() if not q.startswith("self|")}
        sy.pcache.clear()
        sy.scache.clear()
        print("speaker", i, {f"{c}|{r['stat']}": {k: r[k] for k in ("e_self", "d_cb", "h_cb", "p_max", "go")} for c in ["clean"] + conds for r in rows if r["idx"] == i and r["cond"] == c}, flush=True)
    outp = ROOT / opt.get("out", "results/snv_p59/rel.json")
    outp.write_text(json.dumps({"prereg": "results/snv_p59/prereg.yaml(step_a)", "sys": TAG, "rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

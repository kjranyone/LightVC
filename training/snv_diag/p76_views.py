"""段 P76 step a(results/snv_p76/prereg.yaml・学習なし): 入力の視点(源と操作者の登録の包絡を同じ α で伸縮)で回路の出力を平均して描いた音の CER。
腕: s<i> = 系 i の単独(α = 1)・v<i> = 系 i の全ての視点の平均・c = 全ての系の平均(α = 1)・cv = 全ての系 × 全ての視点の平均。女声 2 声へ k 7。
参照 = 源の元の音の書き起こし(1 で切る)。書き起こしの文字は保存も表示もしない(数値だけ)。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p76_views.py [sys=snv_p30_c,snv_p24_c] [alphas=0.93,1,1.07] [sets=cv] [nmax=64] [out=results/snv_p76/views.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvoc as N
import p18_align as AL
import snv_p1_eval as EV
import train_rvoc as TR
from p74_concord import boot

ROOT = EV.ROOT
DEV = EV.DEV


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags, names = opt.get("sys", "snv_p30_c,snv_p24_c").split(","), opt.get("sets", "cv").split(",")
    alphas = [float(a) for a in opt.get("alphas", "0.93,1,1.07").split(",")]
    assert 1.0 in alphas
    out = ROOT / opt.get("out", "results/snv_p76/views.json")
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    tr = lambda w: kana("".join(s.text for s in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    gen = EV.load_gen()
    syss = [EV.System(t, gen) for t in tags]
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    sets = {nm: [x for x in its if x["valid"]][:int(opt.get("nmax", 64))] for nm, its in AL.build_sets(names, 64, split).items()}
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:2]
    St = [{t: sy.structure(f"t|{t}", [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)]) for t in tgts} for sy in syss]

    @torch.no_grad()
    def render(u: dict, yo: torch.Tensor, k: int) -> np.ndarray:
        v = torch.from_numpy(u["v"])[None].to(DEV)
        lf = torch.from_numpy(u["lf"])[None].to(DEV) + k * EV.T.LN2_12 * v
        cond = torch.cat([yo[:, :25], lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"] * 2 ** (k / 12))[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    arms = [f"s{i}" for i in range(len(tags))] + [f"v{i}" for i in range(len(tags))] + ["c", "cv"]
    i1 = alphas.index(1.0)
    rows = []
    for nm, its in sets.items():
        for it in its:
            u = it["u"]
            views = [(u, it["enr"][:3]) if a == 1.0 else (EV.warp_utt(u, a), [EV.warp_utt(e, a) for e in it["enr"][:3]]) for a in alphas]
            r = {"set": nm, "spk": it["spk"], "src": it["src"]} | {k: [] for k in arms}
            for t in tgts:
                ys = [[sy.convert(uw, ew, S[t], 7, with_y=True)[1][:, :29] for uw, ew in views] for sy, S in zip(syss, St)]
                c = lambda yo: round(min(cer(it["ref"], tr(render(u, yo, 7))), 1.0), 4)
                for i in range(len(tags)):
                    r[f"s{i}"].append(c(ys[i][i1]))
                    r[f"v{i}"].append(c(sum(ys[i]) / len(alphas)))
                r["c"].append(c(sum(y[i1] for y in ys) / len(tags)))
                r["cv"].append(c(sum(sum(y) for y in ys) / (len(tags) * len(alphas))))
            for sy in syss:
                sy.scache.clear()
            rows.append(r)
    val = {k: np.array([float(np.mean(r[k])) for r in rows]) for k in arms}
    g = np.array([r["spk"] for r in rows])
    diff = {f"v{i}-s{i}": boot(val[f"v{i}"] - val[f"s{i}"], g) for i in range(len(tags))} | {"cv-c": boot(val["cv"] - val["c"], g), "c-mean(s)": boot(val["c"] - np.mean([val[f"s{i}"] for i in range(len(tags))], 0), g)}
    res = {"prereg": "results/snv_p76/prereg.yaml(step_a)", "sys": tags, "alphas": alphas, "n": len(rows), "mean": {k: round(float(v.mean()), 4) for k, v in val.items()}, "diff": diff, "rows": rows}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

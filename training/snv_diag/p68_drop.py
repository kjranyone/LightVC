"""段 P68 step a(results/snv_p68/prereg.yaml・学習なし・ASR なし): 回路が強さを描き足りない所は、崩れた試行の「消えた区間」か。
snv_dev_eval と同じ試行(devA の 1〜n 人目と JVNV・源 × 目標 2 声 × k 0 / 7)で、系ごとに z(t) = (直接の c0 − 回路の c0) / s_t を出す。
直接の c0 = m_t + s_t (源の c0 − m_o) / s_o(paper_snv §5 M48 と同じ・m・s は有声のフレームの c0 の平均と標準偏差)。有声で門が閉じたフレームの上で、
drop = z > 1 の割合・dmean = relu(z) の平均・run = z > 1 が 10 フレーム以上続く区間に入るフレームの割合・over = z < −1 の割合。
出力 = results/snv_p68/drop.json(試行の行・数値だけ)。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p68_drop.py [<tag> ...] [n=64] [out=results/snv_p68/drop.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_dev_eval as DE
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV
TAGS = ["snv_p30_c+mask:snv_p45_gate_t80+fb:anch", "snv_p34_c+mask:snv_p45_gate_t80+fb:anch"]


def runs(b: np.ndarray, n: int) -> float:
    out, i = 0, 0
    while i < len(b):
        if b[i]:
            j = i
            while j < len(b) and b[j]:
                j += 1
            if j - i >= n:
                out += j - i
            i = j
        else:
            i += 1
    return out / max(len(b), 1)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x] or TAGS
    gen = EV.load_gen()
    tg4 = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:4]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tg4}
    items = DE.cv_items("devA", int(opt.get("n", 64))) + DE.jvnv_items()
    rows = []
    for tag in tags:
        sy = EV.System(tag, gen)
        for it in items:
            u, enr = sy.prep(it["u"]), it["enr"]
            y, v = torch.from_numpy(u["y"])[None].to(DEV), torch.from_numpy(u["v"])[None].to(DEV)
            g = torch.from_numpy(u["gate"])[None].to(DEV) if "gate" in u else torch.zeros_like(v)
            lf = torch.from_numpy(u["lf"])[None].to(DEV)
            c = np.concatenate([e["y"][0][e["v"] > 0.5] for e in (sy.prep(e) for e in enr)])
            m0, s0 = float(c.mean()), max(float(c.std()), 1e-3)
            mk = ((v > 0.5) & (g <= 0.5))[0].cpu().numpy()
            with torch.no_grad():
                pl = sy.play(y, v, sy.op_stat(enr))
            for k in DE.KS:
                for tg in (tg4[it["idx"] % 4], tg4[(it["idx"] + 2) % 4]):
                    St = sy.structure(f"t|{tg}", trefs[tg])
                    with torch.no_grad():
                        yo = sy.C(pl, lf + k * T.LN2_12 * v, v, St)
                    mt, st = sy.hc0[id(St)]
                    z = ((float(mt) + float(st) * (y[0, 0] - m0) / s0 - yo[0, 0]) / float(st)).cpu().numpy()[mk]
                    rows.append({"sys": tag, "group": it["group"], "spk": it["spk"], "src": it["src"], "tgt": tg, "k": k, "n": int(mk.sum()), "drop": round(float((z > 1).mean()), 4), "dmean": round(float(np.maximum(z, 0).mean()), 4),
                                 "run": round(runs(z > 1, 10), 4), "over": round(float((z < -1).mean()), 4), "zmean": round(float(z.mean()), 4)})
            sy.pcache.clear()
        print(tag, "devA の drop / dmean / run / over / zmean の中央値", [round(float(np.median([r[q] for r in rows if r["sys"] == tag and r["group"] == "devA"])), 3) for q in ("drop", "dmean", "run", "over", "zmean")], flush=True)
        del sy
        torch.cuda.empty_cache()
    outp = ROOT / opt.get("out", "results/snv_p68/drop.json")
    outp.write_text(json.dumps({"prereg": "results/snv_p68/prereg.yaml(step_a)", "tags": tags, "rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

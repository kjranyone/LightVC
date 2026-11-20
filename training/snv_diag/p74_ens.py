"""段 P74 step d(results/snv_p74/prereg.yaml・学習なし): 複数の系の回路の出力(描画の前の 29 次元)を平均して描いた音の CER。
個々の系と、先頭 2 つの平均(ens2)・全ての平均(ensN)を、女声 2 声へ k 7 で比べる。参照 = 源の元の音の書き起こし(1 で切る)。書き起こしの文字は保存も表示もしない(数値だけ)。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p74_ens.py [sys=snv_p30_c,snv_p31_c1,snv_p24_c,snv_p26_c1] [sets=cv] [nmax=64] [out=results/snv_p74/ens.json]
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
    tags, names = opt.get("sys", "snv_p30_c,snv_p31_c1,snv_p24_c,snv_p26_c1").split(","), opt.get("sets", "cv").split(",")
    out = ROOT / opt.get("out", "results/snv_p74/ens.json")
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

    rows = []
    for nm, its in sets.items():
        for it in its:
            enr, u = it["enr"][:3], it["u"]
            r = {"set": nm, "spk": it["spk"], "src": it["src"]} | {k: [] for k in tags + ["ens2", f"ens{len(tags)}"]}
            for t in tgts:
                ws, ys = zip(*[sy.convert(u, enr, S[t], 7, with_y=True) for sy, S in zip(syss, St)])
                for tag, w in zip(tags, ws):
                    r[tag].append(round(min(cer(it["ref"], tr(w)), 1.0), 4))
                r["ens2"].append(round(min(cer(it["ref"], tr(render(u, (ys[0] + ys[1]) / 2, 7))), 1.0), 4))
                r[f"ens{len(tags)}"].append(round(min(cer(it["ref"], tr(render(u, sum(ys) / len(ys), 7))), 1.0), 4))
            rows.append(r)
    val = {k: np.array([float(np.mean(r[k])) for r in rows]) for k in tags + ["ens2", f"ens{len(tags)}"]}
    g = np.array([r["spk"] for r in rows])
    ind = np.mean([val[t] for t in tags], 0)
    res = {"prereg": "results/snv_p74/prereg.yaml(step_d)", "sys": tags, "n": len(rows), "mean": {k: round(float(v.mean()), 4) for k, v in val.items()},
           "diff": {"ens2-mean(first2)": boot(val["ens2"] - (val[tags[0]] + val[tags[1]]) / 2, g), f"ens{len(tags)}-mean(all)": boot(val[f"ens{len(tags)}"] - ind, g), f"ens{len(tags)}-{tags[0]}": boot(val[f"ens{len(tags)}"] - val[tags[0]], g)}, "rows": rows}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

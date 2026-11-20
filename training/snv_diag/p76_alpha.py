"""段 P76 step a2(results/snv_p76/prereg.yaml・学習なし): 入力の伸縮 α(源と操作者の登録の包絡を同じ α で伸縮)を 1 つずつ当てた CER。
global = α ごとの平均と α = 1 との対の差。crossfit = 源が 2 本ある話者で、片方の源で CER が最小の α(同じ値なら 1 に近い方)を選び、もう片方の源に当てた CER − α = 1 の CER。
参照 = 源の元の音の書き起こし(1 で切る)。書き起こしの文字は保存も表示もしない(数値だけ)。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p76_alpha.py [sys=snv_p30_c] [alphas=0.86,0.93,1,1.07,1.14] [sets=cv] [nmax=200] [out=results/snv_p76/alpha.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p18_align as AL
import snv_p1_eval as EV
from p74_concord import boot

ROOT = EV.ROOT
DEV = EV.DEV


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tag, names = opt.get("sys", "snv_p30_c"), opt.get("sets", "cv").split(",")
    alphas = [float(a) for a in opt.get("alphas", "0.86,0.93,1,1.07,1.14").split(",")]
    i1 = alphas.index(1.0)
    out = ROOT / opt.get("out", "results/snv_p76/alpha.json")
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    tr = lambda w: kana("".join(s.text for s in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    sy = EV.System(tag, EV.load_gen())
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    sets = {nm: [x for x in its if x["valid"]][:int(opt.get("nmax", 200))] for nm, its in AL.build_sets(names, 64, split).items()}
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:2]
    St = {t: sy.structure(f"t|{t}", [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)]) for t in tgts}
    rows = []
    for nm, its in sets.items():
        for it in its:
            u = it["u"]
            views = [(u, it["enr"][:3]) if a == 1.0 else (EV.warp_utt(u, a), [EV.warp_utt(e, a) for e in it["enr"][:3]]) for a in alphas]
            rows.append({"set": nm, "spk": it["spk"], "src": it["src"], "cer": [round(float(np.mean([min(cer(it["ref"], tr(sy.convert(uw, ew, St[t], 7))), 1.0) for t in tgts])), 4) for uw, ew in views]})
            sy.scache.clear()
    C = np.array([r["cer"] for r in rows])
    g = np.array([r["spk"] for r in rows])
    glob = {str(a): {"mean": round(float(C[:, i].mean()), 4), "diff_vs_1": boot(C[:, i] - C[:, i1], g)} for i, a in enumerate(alphas)}
    by: dict = {}
    for i, r in enumerate(rows):
        by.setdefault(r["spk"], []).append(i)
    two = {s: ix[:2] for s, ix in by.items() if len(ix) >= 2}
    best = lambda a: min(range(len(alphas)), key=lambda i: (C[a, i], abs(alphas[i] - 1.0)))
    cf = np.array([np.mean([C[b, best(a)] - C[b, i1] for a, b in (ix, ix[::-1])]) for ix in two.values()])
    ins = np.array([np.mean([C[a].min() - C[a, i1] for a in ix]) for ix in two.values()])
    pick = np.array([[best(a) for a in ix] for ix in two.values()])
    gs = np.array(list(two.keys()))
    res = {"prereg": "results/snv_p76/prereg.yaml(step_a2)", "sys": tag, "alphas": alphas, "n": len(rows), "n_spk": len(by), "global": glob,
           "crossfit": {"n_spk": len(two), "diff": boot(cf, gs), "in_sample(上限の目安)": boot(ins, gs), "同じ α を選んだ話者の率": round(float((pick[:, 0] == pick[:, 1]).mean()), 3),
                        "選んだ α の分布": {str(a): int((pick == i).sum()) for i, a in enumerate(alphas)}}, "rows": rows}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

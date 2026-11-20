"""段 P64 step a0(results/snv_p64/prereg.yaml・重みの学習なし・ASR なし): 回路の出力の渡りと錨の渡りの一致 A(paper_snv §5 M47・snv_p1_eval.agree)が、変換の CER を当てるか。
snv_dev_eval と同じ試行(devA の 1〜n 人目と JVNV・源 × 目標 2 声 × k 0 / 7)で、系(既定は F2)の A・錨を 20 フレームずらした床・自己の経路(C(a, 操作者) と源の包絡)の一致を出す。
fit=enr / fit=src を付けると、(T, b) を当てはめた後の、変換する発話の A も出す(enr は登録だけで当てはめた、当てはめに使っていない発話の値)。
q=1(段 P65 step a・results/snv_p65/prereg.yaml)= 回路の出力 o と錨 w の食い違いの量(quantities)も出す。
出力 = results/snv_p64/agree.json(試行の行・数値だけ)。CER との関係は scratchpad の p64_a0.py・p65_a.py で読む。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p64_agree.py [n=64] [fit=enr] [q=1] [out=results/snv_p64/agree.json]
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
TAG = "snv_p30_c+mask:snv_p45_gate_t80"


def quantities(yo: torch.Tensor, yw: torch.Tensor, y: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> dict:
    """段 P65 step a の量。yo = 回路の出力・yw = 錨・y = 源の特徴 [1, 29+, T]。fm・sd・sm・ud・um・ed・em・ap・jm は有声で門が閉じたフレーム、loc10・locf = 粗い形の速い成分の、窓 40 フレーム・歩み 20 の局所の余弦の 10 % 点と 0.5 未満の窓の割合(step b の補助)、xd・xm は無声で門が閉じたフレーム。vr・rate・dyn は源だけの対照。"""
    mv, mx = ((v > 0.5) & (g <= 0.5)).float(), ((v <= 0.5) & (g <= 0.5)).float()
    nrm = lambda z, m: float((z * m[:, None]).pow(2).sum().sqrt())
    cos = lambda a, b, m: float((a * b * m[:, None]).sum()) / max(nrm(a, m) * nrm(b, m), 1e-8)
    lr = lambda a, b, m: float(np.log(max(nrm(a, m), 1e-8) / max(nrm(b, m), 1e-8)))
    dem = lambda z, m: z - (z * m[:, None]).sum(-1, keepdim=True) / m.sum().clamp(min=1.0)
    fo, fw = EV.fast_part(yo[:, 1:13]), EV.fast_part(yw[:, 1:13])
    so, sw = dem(yo[:, 1:13] - fo, mv), dem(yw[:, 1:13] - fw, mv)
    uo, uw = EV.fast_part(yo[:, 13:25]), EV.fast_part(yw[:, 13:25])
    eo, ew = dem(yo[:, :1], mv), dem(yw[:, :1], mv)
    xo, xw = dem(yo[:, 1:25], mx), dem(yw[:, 1:25], mx)
    do, dw, md = yo[:, 1:13, 1:] - yo[:, 1:13, :-1], yw[:, 1:13, 1:] - yw[:, 1:13, :-1], mv[:, 1:] * mv[:, :-1]
    c0 = y[0, 0]
    loc = []
    for t in range(0, fo.shape[-1] - 39, 20):
        m = mv[:, t:t + 40]
        if float(m.sum()) >= 10:
            loc.append(cos(fo[..., t:t + 40], fw[..., t:t + 40], m))
    loc = np.array(loc if loc else [0.0])
    return {k: round(q, 4) for k, q in {
        "loc10": float(np.percentile(loc, 10)), "locf": float((loc < 0.5).mean()),
        "fm": lr(fo, fw, mv), "sd": cos(so, sw, mv), "sm": lr(so, sw, mv), "ud": cos(uo, uw, mv), "um": lr(uo, uw, mv), "ed": cos(eo, ew, mv), "em": lr(eo, ew, mv),
        "ap": float(((yo[:, 25:29].clamp(0, 1.5) - yw[:, 25:29].clamp(0, 1.5)).abs() * mv[:, None]).sum() / (4 * mv.sum().clamp(min=1.0))),
        "xd": cos(xo, xw, mx), "xm": lr(xo, xw, mx), "jm": lr(do, dw, md), "vr": float((v > 0.5).float().mean()), "rate": nrm(dw, md) / max(float(md.sum()), 1.0) ** 0.5,
        "dyn": float(c0[v[0] > 0.5].median() - torch.quantile(c0, 0.05)) if bool((v > 0.5).any()) else 0.0}.items()}


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    fit = opt.get("fit", "")
    sy = EV.System(TAG + (f"+tf:{fit}" if fit else "") + "+fb:anch", EV.load_gen())
    tg4 = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:4]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tg4}
    rows = []
    for it in DE.cv_items("devA", int(opt.get("n", 64))) + DE.jvnv_items():
        u, enr = sy.prep(it["u"]), it["enr"]
        key = "self|" + ",".join(str(id(e)) for e in enr)
        sy.pins.setdefault(key, enr)
        So = sy.structure(key, enr)
        for k in DE.KS:
            for tg in (tg4[it["idx"] % 4], tg4[(it["idx"] + 2) % 4]):
                St = sy.structure(f"t|{tg}", trefs[tg])
                x, lf, v, yw, m = sy.tf_inputs(u, enr, St, k * T.LN2_12)
                with torch.no_grad():
                    yo = sy.C(sy.tf_play(x, None), lf, v, St)
                    r = {"group": it["group"], "spk": it["spk"], "src": it["src"], "tgt": tg, "k": k, "n": int(m.sum()), "A": round(float(EV.agree(yo, yw, m)), 4),
                         "A_floor": round(float(EV.agree(yo, torch.cat([yw[..., 20:], yw[..., :20]], -1), m)), 4)}
                    if opt.get("q"):
                        r |= quantities(yo, yw, torch.from_numpy(u["y"])[None].to(DEV), v, torch.from_numpy(u["gate"])[None].to(DEV) if "gate" in u else torch.zeros_like(v))
                    if k == 0:
                        r["A_self"] = round(float(EV.agree(sy.C(sy.tf_play(x, None), lf, v, So), torch.from_numpy(u["y"])[None].to(DEV), m)), 4)
                if fit:
                    Tm, b, log = sy.tf_fit(u, enr, St, k * T.LN2_12)
                    with torch.no_grad():
                        r |= {"A_fit": round(float(EV.agree(sy.C(sy.tf_play(x, (Tm, b)), lf, v, St), yw, m)), 4), "A_fit_train": [round(log[0], 4), round(log[-1], 4)], "dT": round(float((Tm - torch.eye(Tm.shape[0], device=DEV)).norm()), 3)}
                rows.append(r)
        print(it["group"], it["idx"], it["src"][-12:], {q: r[q] for q in ("A", "A_floor", "A_fit") if q in r}, flush=True)
    outp = ROOT / opt.get("out", "results/snv_p64/agree.json")
    outp.write_text(json.dumps({"prereg": "results/snv_p64/prereg.yaml(step_a0)", "sys": sy.tag, "tf": [EV.TF_STEPS, EV.TF_LR, EV.TF_LAM] if fit else None, "rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""段 I9(paper_snv rev20 §5 M14)の方法の実現の確認(学習なし): 推論の E が乱しの族(包絡の一律の伸縮・傾き = train_snv_p2_cpc.augment)に不変になったかと、出力に残る性別の分離を、段 I8 と段 I9 の両方の腕で並べる。
検証の Irodori の声の e1 の発話(源 = 各声 2 本・E の参照 = 同じ声の別の 3 本・学習と同じく源と参照に同じ乱し)。pool=train = 学習の声から検証と同じ数(男声 6・女声 30)を seed= で選ぶ(学習の声と検証の声の差 = 当てはめ)。
inv = Σ ‖E(乱した y) − E(y)‖² / Σ ‖E(y) − 発話の平均‖²(全フレーム・内容 59 次元と演技 5 次元を別に)。
sex_d2 = 発話の平均(内容)を男声と女声の平均の差の向きに射影した 1 次元の分離 (m_M − m_F)² / 平均の分散・shift_a1.25 = 男声を α 1.25 で伸縮したときに女声の側へ動く割合。
    cd training && SNV_DEV=cpu uv run --no-sync python snv_diag/i9_inv_probe.py [snv_i8_m13 snv_i9_ctl snv_i9_aug] [pool=val|train] [seed=0] [out=results/snv_i9/inv_probe.json]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p1_eval as EV
import train_snv_p2_cpc as P2
from snv_diag.i8_text_probe import texts_by_voice

ROOT = EV.ROOT
DEV = EV.DEV
D_CT = 59
WARPS = [("a0.80", 0, 0.0), ("a0.89", 8, 0.0), ("a1.12", 24, 0.0), ("a1.25", 32, 0.0), ("t-1", 16, -1.0), ("t+1", 16, 1.0)]


def aug(y: torch.Tensor, Ms: torch.Tensor, tvec: torch.Tensor, ia: int, tau: float) -> torch.Tensor:
    fx = (torch.tensor([ia], device=DEV), torch.tensor([tau], device=DEV, dtype=y.dtype))
    return P2.augment(y, Ms, tvec, None, fixed=fx)


@torch.no_grad()
def play(s: EV.System, y: torch.Tensor, v: torch.Tensor, ry: torch.Tensor, rv: torch.Tensor, rm: torch.Tensor) -> torch.Tensor:
    return s.Es(s.P11.xin(y, v), s.Es.structure(s.P11.xin(ry, rv), rm))[0][0]


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x] or ["snv_i8_m13", "snv_i9_ctl", "snv_i9_aug"]
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    tv = texts_by_voice()
    Ms, tvec = torch.from_numpy(P2.warp_mats()).to(DEV), torch.from_numpy(P2.tilt_vec()).to(DEV)
    voices = split["val"]
    if opt.get("pool", "val") == "train":
        rng = random.Random(int(opt.get("seed", 0)))
        tm, tf = [v for v in split["train"] if v.startswith("em")], [v for v in split["train"] if v.startswith("ef")]
        nm = sum(v.startswith("em") for v in voices)
        voices = rng.sample(tm, nm) + rng.sample(tf, len(voices) - nm)
    trials = []
    for vv in voices:
        e1 = [r for r, _ in tv.get(vv, []) if r.startswith("e1/")]
        for src in e1[:2]:
            st = [r for r in e1 if r != src][:3]
            if len(st) == 3:
                trials.append((vv, EV.utt(src), [EV.utt(r) for r in st]))
    y0 = torch.from_numpy(trials[0][1]["y"])[None].to(DEV)
    assert torch.allclose(aug(y0, Ms, tvec, 16, 0.0), y0, atol=1e-4), "恒等の乱しが恒等でない"
    assert not torch.allclose(aug(y0, Ms, tvec, 32, 0.0), y0, atol=1e-2), "伸縮が入力を変えない"
    res: dict = {"tags": tags, "pool": opt.get("pool", "val"), "seed": opt.get("seed"), "voices": voices, "n_utts": len(trials), "n_male": sum(t[0].startswith("em") for t in trials), "rows": {}}
    for tag in tags:
        s = EV.System(tag, None)
        assert s.p11, tag
        num = {w[0]: np.zeros(2) for w in WARPS}
        den = np.zeros(2)
        um, uw, male = [], [], []
        for vv, u, refs in trials:
            y = torch.from_numpy(u["y"])[None].to(DEV)
            v = torch.from_numpy(u["v"])[None].to(DEV)
            ry, rv, rm, _ = EV.struct_inputs(refs)
            a0 = play(s, y, v, ry, rv, rm)
            c = a0 - a0.mean(-1, keepdim=True)
            den += [float((c[:D_CT] ** 2).sum()), float((c[D_CT:] ** 2).sum())]
            for name, ia, tau in WARPS:
                aw = play(s, aug(y, Ms, tvec, ia, tau), v, aug(ry, Ms, tvec, ia, tau), rv, rm)
                d = aw - a0
                num[name] += [float((d[:D_CT] ** 2).sum()), float((d[D_CT:] ** 2).sum())]
                if name == "a1.25":
                    uw.append(aw[:D_CT].mean(-1).cpu().numpy())
            um.append(a0[:D_CT].mean(-1).cpu().numpy())
            male.append(vv.startswith("em"))
        um, uw, male = np.stack(um), np.stack(uw), np.array(male)
        dv = um[male].mean(0) - um[~male].mean(0)
        dv /= np.linalg.norm(dv)
        p, pw = um @ dv, uw @ dv
        sep = float((p[male].mean() - p[~male].mean()) ** 2 / (0.5 * (p[male].var() + p[~male].var())))
        shift = float((pw[male].mean() - p[male].mean()) / (p[~male].mean() - p[male].mean()))
        row = {"inv_ct": {k: round(float(x[0] / den[0]), 4) for k, x in num.items()}, "inv_act": {k: round(float(x[1] / den[1]), 4) for k, x in num.items()},
               "sex_d2": round(sep, 3), "shift_a1.25_male": round(shift, 3), "sex_dist_over_within_sd": round(float(np.sqrt(sep)), 3)}
        res["rows"][tag] = row
        print(tag, json.dumps(row), flush=True)
    out = ROOT / opt.get("out", "results/snv_i9/inv_probe.json")
    out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

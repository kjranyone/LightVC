"""P2b の c の切り分け(synth_nam_vc §8.4・学習なし): c は一律の伸縮に不変か = 源(と操作者の登録)の包絡を伸縮 α で先に伸ばしてから c にしたときの NNC。
組と物差しは snv_p2_nnc.py と同じ(RM・IM の源 → 検証の女声 10 声の t00〜t02・SPARC の EMA 12 次元の相関)。女声の側は伸ばさない。
α で NNC が上がる = c に伸縮の分の声が残る(不変性の型の不足)・平ら = c は伸縮に不変で、残りの跨ぎは伸縮でない違い。
自己の伸縮の対応 = 同じ発話の c(x) と c(伸縮(x))の各フレームの最近傍が ±2 フレーム以内に戻る率(女声の t09)。
c-w = 話者の登録(源 = 操作者の登録・目標 = 参照)の平均と共分散で白色化(縮み W × 平均の分散・登録の定数)。FF = 検証の女声の t09 → 自分以外の女声(α 1.0 だけ)。
    cd training && uv run --no-sync python snv_diag/p2_warp_nnc.py P=snv_p2b_cpc[@step15k] [A=1.0,1.08,1.16,1.25] [W=0.1]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p0c_anchor as A
import snv_p1_eval as EV
import snv_p2_nnc as NN
import train_snv_p1 as T
import train_snv_p2_cpc as CP

ROOT = EV.ROOT
SP = ROOT / "data/snv_e1_feat/sparc"
DEV = EV.DEV
SUB = 4


def warped(u: dict, alpha: float) -> dict:
    if alpha == 1.0:
        return u
    y = u["y"].copy()
    env = np.concatenate([y[:1] * 10 + T.PA.C0_SIL, y[1:25] * 10], 0)
    ew = A.warp_env(env, alpha)
    y[0], y[1:25] = (ew[0] - T.PA.C0_SIL) / 10, ew[1:25] / 10
    return u | {"y": y}


def whitener(x: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(0)
    cov = np.cov((x - mu).T)
    cov = cov + shrink * np.trace(cov) / len(cov) * np.eye(len(cov))
    ev, U = np.linalg.eigh(cov)
    return mu, (U / np.sqrt(np.maximum(ev, 1e-8))) @ U.T


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = opt.get("P", "snv_p2b_cpc").split(",")
    alphas = [float(a) for a in opt.get("A", "1.0,1.08,1.16,1.25").split(",")]
    shrink = float(opt.get("W", "0.1"))
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    fem = [v for v in split["val"] if v.startswith("ef")][::3]
    mal = [v for v in split["val"] if v.startswith("em")]
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    res: dict = {"ruler": "SPARC multi の EMA 12 次元(物差しとしてだけ)", "alphas": alphas, "by": {}}
    for tag in tags:
        name, _, ck = tag.partition("@")
        st = torch.load(ROOT / "results" / name / f"{ck or 'last'}.pt", map_location="cpu", weights_only=False)
        cm = CP.CPC().to(DEV).eval()
        cm.load_state_dict(st["cpc"])
        sig = st["sig"]

        @torch.no_grad()
        def ctx(u: dict) -> np.ndarray:
            return cm.encode(torch.from_numpy(CP.x_of(u, sig))[None].to(DEV))[1][0].cpu().numpy().T

        pools = {}
        for v in fem:
            us = [EV.utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)]
            c = np.concatenate([ctx(u)[u["v"] > 0] for u in us])
            e = np.concatenate([NN.ema_at(SP / f"e1/{v}/t0{j}.npz", u["y"].shape[1])[u["v"] > 0] for j, u in zip((0, 1, 2), us)])
            mw, Ww = whitener(c, shrink)
            pools[v] = (c, c - c.mean(0), e, (c - mw) @ Ww)
        qs = [("RM", EV.utt(f"real/{s_}"), NN.ema_at(SP / f"a1src/{s_}.npz", EV.utt(f"real/{s_}")["y"].shape[1]), [EV.utt(f"real/{s_.rsplit('_', 1)[0]}_enroll")]) for s_ in meta["sources"]]
        qs += [("IM", EV.utt(f"e1/{v}/t09"), NN.ema_at(SP / f"e1/{v}/t09.npz", EV.utt(f"e1/{v}/t09")["y"].shape[1]), [EV.utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)]) for v in mal]
        qs += [("FF:" + v, EV.utt(f"e1/{v}/t09"), NN.ema_at(SP / f"e1/{v}/t09.npz", EV.utt(f"e1/{v}/t09")["y"].shape[1]), [EV.utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)]) for v in fem]
        row: dict = {}
        for a in alphas:
            acc: dict = {}
            for grp, u, qe, enr in qs:
                if grp.startswith("FF") and a != 1.0:
                    continue
                c = ctx(warped(u, a))
                ce = np.concatenate([ctx(warped(x, a))[x["v"] > 0] for x in enr])
                mu = ce.mean(0)
                mw, Ww = whitener(ce, shrink)
                vm = u["v"] > 0
                vm[np.arange(len(vm)) % SUB != 0] = False
                for v in fem:
                    if grp == "FF:" + v:
                        continue
                    P, Pc, Pe, Pw = pools[v]
                    for name, q, pool in (("c", c[vm], P), ("c-c", c[vm] - mu, Pc), ("c-w", (c[vm] - mw) @ Ww, Pw)):
                        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
                        pn = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-8)
                        nn = (torch.from_numpy(qn).to(DEV) @ torch.from_numpy(pn).to(DEV).T).argmax(1).cpu().numpy()
                        acc.setdefault(f"{grp.split(':')[0]}|{name}", []).append(NN.corr(qe[vm], Pe[nn]))
            row[f"a{a:.2f}"] = {k: round(float(np.mean(x)), 4) for k, x in acc.items()}
            print(tag, a, json.dumps(row[f"a{a:.2f}"]), flush=True)
        selfc = {}
        for a in alphas[1:]:
            hits = []
            for v in fem:
                u = EV.utt(f"e1/{v}/t09")
                c0, c1 = ctx(u), ctx(warped(u, a))
                vm = np.where(u["v"] > 0)[0][::SUB]
                q = c1[vm] / (np.linalg.norm(c1[vm], axis=1, keepdims=True) + 1e-8)
                p = c0 / (np.linalg.norm(c0, axis=1, keepdims=True) + 1e-8)
                nn = (q @ p.T).argmax(1)
                hits.append(float(np.mean(np.abs(nn - vm) <= 2)))
            selfc[f"a{a:.2f}"] = round(float(np.mean(hits)), 4)
        row["self_warp_return_pm2"] = selfc
        print(tag, "self", json.dumps(selfc), flush=True)
        res["by"][tag] = row
    out = ROOT / (f"results/snv_p2/warp_nnc_{'_'.join(tags).replace('@', '-')}" + (f"_W{shrink}" if "W" in opt else "") + ".json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

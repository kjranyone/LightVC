"""段 I10 の後の切り分け(学習なし): 本物の男声の崩れは推論の符号器 E の入力の分布の外か・E の出力が内容の型の多様体の外か。
入力 = xin(y, v) の 30 次元と過去の文脈を積んだ 90 次元(t−8・t−4・t)を学習の声の発話の参照のフレームで標準化し、1 近傍の距離を発話ごとに中央値と p90 で。
出力 = E の内容の 59 次元の 1 近傍の距離を型(npos > 0)のフレームまで。組 = Irodori の検証の声(元の音・録音の版 k0 / k1)・本物の男声(区間と登録)。
CER の列 = results/snv_i10/text_final.json の real_self(組み合わせ)。
    cd training && uv run --no-sync python snv_diag/i10_real_ood.py [ckpts=snv_i10_combo,snv_i8_m13] [n_ref=400] [out=results/snv_i10/real_ood.json]
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "data/snv_p_feat_rec"
DEV = os.environ.get("SNV_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
LAGS = (8, 4, 0)


def xin_np(u: dict) -> np.ndarray:
    return np.concatenate([u["y"], u["v"][None]], 0).T.astype(np.float32)


def ctx(x: np.ndarray) -> np.ndarray:
    pad = np.concatenate([np.repeat(x[:1], max(LAGS), 0), x], 0)
    n = x.shape[0]
    return np.concatenate([pad[max(LAGS) - l: max(LAGS) - l + n] for l in LAGS], 1)


def nn_dist(q: torch.Tensor, ref: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    out = []
    for i in range(0, q.shape[0], chunk):
        out.append(torch.cdist(q[i:i + chunk], ref).min(1).values)
    return torch.cat(out)


def stats(d: torch.Tensor) -> dict:
    d = d.float().cpu().numpy()
    return {"median": round(float(np.median(d)), 4), "p90": round(float(np.quantile(d, 0.9)), 4)}


@torch.no_grad()
def e_out(E: P11.EncS, u: dict, refs: list[dict]) -> torch.Tensor:
    t = lambda a: torch.from_numpy(a)[None].to(DEV)
    ry, rv = t(np.concatenate([r["y"] for r in refs], 1)), t(np.concatenate([r["v"] for r in refs]))
    s = E.structure(P11.xin(ry, rv), torch.ones(rv.shape, dtype=torch.bool, device=DEV))
    return E(P11.xin(t(u["y"]), t(u["v"])), s)[0][0].T


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = opt.get("ckpts", "snv_i10_combo,snv_i8_m13").split(",")
    rng = random.Random(0)
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    tr_files = [p for v in split["train"] for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))]
    ref_u = [T.load_utt(p) for p in rng.sample(tr_files, int(opt.get("n_ref", 400)))]
    X = np.concatenate([xin_np(u) for u in ref_u], 0)
    XC = np.concatenate([ctx(xin_np(u)) for u in ref_u], 0)
    mu, sd = X.mean(0), X.std(0) + 1e-3
    muc, sdc = XC.mean(0), XC.std(0) + 1e-3
    Rx = torch.from_numpy((X - mu) / sd).to(DEV)
    Rc = torch.from_numpy((XC - muc) / sdc).to(DEV)
    groups: dict[str, list[tuple[str, dict, list[dict]]]] = {"irodori_val": [], "irodori_rec": [], "real": []}
    for v in split["val"]:
        fs = sorted((T.FEAT / "e1" / v).glob("t*.npz"))
        if len(fs) < 5:
            continue
        refs = [T.load_utt(p) for p in fs[:3]]
        for p in fs[3:5]:
            groups["irodori_val"].append((f"{v}/{p.stem}", T.load_utt(p), refs))
            for k in (0, 1):
                q = REC / "e1" / v / f"{p.stem}_k{k}.npz"
                if q.exists():
                    rr = [T.load_utt(REC / "e1" / v / f"{r.stem}_k{k}.npz") for r in fs[:3]]
                    groups["irodori_rec"].append((f"{v}/{p.stem}_k{k}", T.load_utt(q), rr))
    for sp in ("namikawa", "uirou"):
        enr = [T.load_utt(T.FEAT / f"real/{sp}_enroll.npz")]
        for i in range(6):
            groups["real"].append((f"{sp}_{i}", T.load_utt(T.FEAT / f"real/{sp}_{i}.npz"), enr))
        groups["real"].append((f"{sp}_enroll", enr[0], enr))
    text = json.loads((ROOT / "results/snv_i10/text_final.json").read_text())
    cer = {x["rel"].split("/")[-1]: x["cer"] for x in text["rows"] if x["cond"] == "real_self" and x["sys"] == "snv_i10_combo"}
    Es = {}
    tplf = None
    for tg in tags:
        st = torch.load(ROOT / "results" / tg / "last.pt", map_location="cpu", weights_only=False)
        E = P11.EncS().to(DEV).eval()
        E.load_state_dict(st["Es"])
        Es[tg] = E
        if tplf is None:
            tplf = st["tpl"][st["npos"] > 0].float().to(DEV)
    res: dict = {"tags": tags, "n_ref_frames": int(X.shape[0]), "n_tpl_frames": int(tplf.shape[0]), "groups": {}, "rows": []}
    for g, items in groups.items():
        agg: dict = {}
        for name, u, refs in items:
            x = xin_np(u)
            row = {"group": g, "name": name, "frames": int(x.shape[0]), "voiced": round(float(u["v"].mean()), 3),
                   "c0_unv_med": round(float(np.median(u["y"][0][u["v"] < 0.5])) if (u["v"] < 0.5).any() else float("nan"), 3),
                   "in_x": stats(nn_dist(torch.from_numpy((x - mu) / sd).to(DEV), Rx)),
                   "in_ctx": stats(nn_dist(torch.from_numpy((ctx(x) - muc) / sdc).to(DEV), Rc))}
            for tg, E in Es.items():
                a = e_out(E, u, refs)
                row[f"out_{tg}"] = stats(nn_dist(a[:, :tplf.shape[1]].contiguous(), tplf))
                row[f"tstd_{tg}"] = round(float(a[:, :tplf.shape[1]].std(0).mean()), 4)
            if name in cer:
                row["cer_combo"] = round(cer[name], 3)
            res["rows"].append(row)
            for k in ["in_x", "in_ctx"] + [f"out_{tg}" for tg in tags]:
                agg.setdefault(k, []).append(row[k]["median"])
            for k in [f"tstd_{tg}" for tg in tags]:
                agg.setdefault(k, []).append(row[k])
        res["groups"][g] = {k: round(float(np.median(v)), 4) for k, v in agg.items()} | {"n": len(items)}
        print(g, res["groups"][g], flush=True)
    for r in res["rows"]:
        if r["group"] == "real":
            print(r["name"], "cer", r.get("cer_combo"), "v", r["voiced"], "c0u", r["c0_unv_med"], "in", r["in_x"], r["in_ctx"], {k: r[k] for k in r if k.startswith("out_") or k.startswith("tstd_")}, flush=True)
    (ROOT / opt.get("out", "results/snv_i10/real_ood.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

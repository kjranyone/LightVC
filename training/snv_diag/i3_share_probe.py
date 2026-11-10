"""同じ文を読んだ声どうしで、表現の軌跡がどれだけ共有されるか(paper_snv §6 段 I の 3 走行の文の型 R² ≈ 0 の切り分け・学習なし・CPU)。
段 D の文(学習の声が 6 声以上読んだもの)ごとに、1 声を抜いた残りの声の軌跡から型を作り(時刻の比で平均 → 単調な対応で揃えて平均を 3 回)、抜いた声を型へ単調に対応づけて
R² = 1 − Σ(x − 型)² / Σ(x − 文の平均)²。表現 = Y(前段の 29 次元・次元ごとに全体の標準偏差で割る)・声ごとに標準化した Y(その声の普通の発話の平均と標準偏差 = 1 次と 2 次の声の音色を外す)と、重みを与えたときの E の演奏の平均 u(構造 = 同じ声の別の普通の発話 3 本)。
4 フレームの平均・冒頭 WARM_SELF を外す。Y の R² が u より高い = E が声ごとの座標で内容を書いている(命題 1)。
残差なしの固い共有(M8 / M9)の重みでは、E の出力を内容の軌跡(先頭 D_CT)と明示の演技(残り 5)に分けても測る(段 I5: まとめた R² は演技から来ていた)。
    cd training && uv run --no-sync python snv_diag/i3_share_probe.py [ckpt=results/snv_i3_m5/step10k.pt] [n=30] [out=results/snv_i3/share_probe.json]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_snv_i as I
import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = I.ROOT


def pool(x: np.ndarray) -> np.ndarray:
    x = x[:, T.WARM_SELF:]
    n = x.shape[1] // I.POOL
    return x[:, :n * I.POOL].reshape(x.shape[0], n, I.POOL).mean(-1).T


def align(x: np.ndarray, tp: np.ndarray) -> np.ndarray:
    lp = -0.5 * ((x[:, None, :] - tp[None]) ** 2).sum(-1)
    return I.mas_free(torch.from_numpy(lp)[None].float(), torch.tensor([len(x)]), torch.tensor([len(tp)]))[0].numpy()


def template(xs: list[np.ndarray], L: int) -> np.ndarray:
    tp = np.mean([x[np.minimum((np.arange(L) * len(x) / L).astype(int), len(x) - 1)] for x in xs], 0)
    for _ in range(3):
        s, c = np.zeros_like(tp), np.zeros(L)
        for x in xs:
            p = align(x, tp)
            np.add.at(s, p, x)
            np.add.at(c, p, 1)
        tp = np.where(c[:, None] > 0, s / np.maximum(c, 1)[:, None], tp)
    return tp


def r2_loo(xs: list[np.ndarray]) -> float:
    L = int(np.median([len(x) for x in xs]))
    mean = np.concatenate(xs).mean(0)
    num = den = 0.0
    for k in range(len(xs)):
        tp = template([x for j, x in enumerate(xs) if j != k], L)
        p = align(xs[k], tp)
        num += float(((xs[k] - tp[p]) ** 2).sum())
        den += float(((xs[k] - mean) ** 2).sum())
    return 1.0 - num / den


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    ck, n = opt.get("ckpt", "results/snv_i3_m5/step10k.pt"), int(opt.get("n", 30))
    torch.set_num_threads(8)
    st = torch.load(ROOT / ck, map_location="cpu", weights_only=False)
    E = P11.EncS().eval()
    E.load_state_dict(st["Es"])
    nores = st.get("residual") == "none"
    names = sorted(json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"], key=lambda v: v["voice"])
    tr, _ = T.split([v["voice"] for v in names])
    c = I.Corpus(tr)
    by: dict = {}
    for v in tr:
        for u in c.u[v]["norm"]:
            if u["cid"] >= 0 and u["key"][0] == "txt":
                by.setdefault(u["key"], []).append((v, u))
    keys = sorted(k for k, it in by.items() if len({v for v, _ in it}) >= 6)
    random.Random(0).shuffle(keys)
    ysd = np.concatenate([u["y"] for v in tr[:40] for u in c.u[v]["norm"]], 1).std(1) + 1e-6

    @torch.no_grad()
    def u_of(v: str, u: dict) -> np.ndarray:
        rs = [x for x in c.u[v]["norm"] if x is not u][:I.REF_N]
        ry = torch.from_numpy(np.concatenate([x["y"][:, :I.REF_LEN] for x in rs], 1))[None]
        rv = torch.from_numpy(np.concatenate([x["v"][:I.REF_LEN] for x in rs]))[None]
        s = E.structure(P11.xin(ry, rv), torch.ones(rv.shape, dtype=torch.bool))
        return E(P11.xin(torch.from_numpy(u["y"])[None], torch.from_numpy(u["v"])[None]), s)[0][0].numpy()

    def vstat(v: str) -> tuple:
        z = np.concatenate([x["y"] for x in c.u[v]["norm"]], 1)
        return z.mean(1, keepdims=True), z.std(1, keepdims=True) + 1e-6

    rows = []
    for k in keys[:n]:
        it = by[k][:8]
        ry = [pool(u["y"] / ysd[:, None]) for _, u in it]
        rn = [pool((u["y"] - vstat(v)[0]) / vstat(v)[1]) for v, u in it]
        uu = [u_of(v, u) for v, u in it]
        ru = [pool(x) for x in uu]
        rows.append((r2_loo(ry), r2_loo(rn), r2_loo(ru)) + ((r2_loo([pool(x[:I.D_CT]) for x in uu]), r2_loo([pool(x[I.D_CT:]) for x in uu])) if nores else ()))
        print(len(rows), [round(x, 3) for x in rows[-1]], flush=True)
    a = np.array(rows)
    res = {"ckpt": ck, "contents": len(rows), "voices_per_content": "6〜8", "R2_Y_mean": round(float(a[:, 0].mean()), 4), "R2_Y_voice_standardized_mean": round(float(a[:, 1].mean()), 4),
           "R2_u_mean": round(float(a[:, 2].mean()), 4), "R2_Y_minus_u": round(float((a[:, 0] - a[:, 2]).mean()), 4), "frac_contents_Y_gt_u": round(float((a[:, 0] > a[:, 2]).mean()), 4)}
    if nores:
        res |= {"R2_u_content_mean": round(float(a[:, 3].mean()), 4), "R2_u_acting_mean": round(float(a[:, 4].mean()), 4)}
    print(json.dumps(res), flush=True)
    (ROOT / opt.get("out", "results/snv_i3/share_probe.json")).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

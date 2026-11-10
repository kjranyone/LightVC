"""P5a の探り(synth_nam_vc §8・学習なし・results/snv_p5/probe_prereg.yaml): 共鳴の位置を操作者の座標に正規化した低次元の軌跡は、性別を跨いで対応し声を運ばないか。
NNC = snv_p2_nnc.py と同じ組(RM・IM・FF)と物差し(SPARC の EMA 12 次元の相関)・声 = p2_voice_nn.py と同じ 16 声の同じ声に戻る率。
表現は snv_p5_formant.py の共鳴(F・B・LSF)と Y(包絡 c1..c24 + 周期性)・z = 話者の登録の有声のフレームの平均と標準偏差で標準化(源 = 操作者の登録・目標 = 参照)。
最近傍はユークリッド(Y_cos だけ余弦 = 既存の nnc.json の Y の再現の確認)。
    cd training && uv run --no-sync python snv_diag/p5_coord_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p1_eval as EV
import snv_p2_nnc as NN
import snv_p5_formant as FM
import train_snv_p1 as T

ROOT = EV.ROOT
SP = ROOT / "data/snv_e1_feat/sparc"
DEV = EV.DEV
SUB = 4
PER = 1500
ZREPS = ["Y", "F", "F3", "FB", "L", "EMA"]
RAW = ["F", "L", "EMA"]


def feats(rel: str, ema: Path | None, sig: np.ndarray) -> dict:
    u = EV.utt(rel)
    n = u["y"].shape[1]
    env = T.y_env(torch.from_numpy(u["y"])[None])[0].numpy()
    F, B = FM.formants(env)
    lF = np.log(F.T)
    return {"v": u["v"] > 0, "Y": u["y"][1:29].T, "Y_cos": u["y"][1:29].T / sig[1:29], "F": lF, "F3": lF[:, :3], "FB": np.concatenate([lF, np.log(np.maximum(B.T, 1.0))], 1),
            "L": np.log(np.maximum(FM.lsf(env).T, 1.0)), "EMA": NN.ema_at(ema, n) if ema is not None else np.zeros((n, 12), np.float32)}


def zstats(fs: list[dict]) -> dict:
    out = {}
    for r in ZREPS:
        x = np.concatenate([f[r][f["v"]] for f in fs])
        out[r] = (x.mean(0), x.std(0) + 1e-6)
    return out


def with_z(f: dict, st: dict) -> dict:
    return f | {r + "z": (f[r] - st[r][0]) / st[r][1] for r in ZREPS}


def nn_idx(a: np.ndarray, b: np.ndarray, cos: bool) -> np.ndarray:
    A = torch.from_numpy(np.ascontiguousarray(a)).float().to(DEV)
    Bm = torch.from_numpy(np.ascontiguousarray(b)).float().to(DEV)
    if cos:
        A = torch.nn.functional.normalize(A, dim=1)
        Bm = torch.nn.functional.normalize(Bm, dim=1)
        return (A @ Bm.T).argmax(1).cpu().numpy()
    return torch.cdist(A, Bm).argmin(1).cpu().numpy()


def main() -> int:
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    tr_us = [T.load_utt(p) for v in split["train"][::12] for p in sorted((T.FEAT / "e1" / v).glob("t0[0-4].npz"))]
    sig = np.mean([u["y"].std(1) for u in tr_us], 0).astype(np.float32)
    fem = [v for v in split["val"] if v.startswith("ef")][::3]
    mal = [v for v in split["val"] if v.startswith("em")]
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    reps = ["Y_cos", "Yz", "F", "Fz", "F3z", "FBz", "L", "Lz", "EMA", "EMAz"]
    cos = {"Y_cos"}
    own = {}
    for v in fem + mal:
        fs = [feats(f"e1/{v}/t0{j}", SP / f"e1/{v}/t0{j}.npz", sig) for j in (0, 1, 2)]
        st = zstats(fs)
        own[v] = ([with_z(f, st) for f in fs], with_z(feats(f"e1/{v}/t09", SP / f"e1/{v}/t09.npz", sig), st))
    pools = {v: {r: np.concatenate([f[r][f["v"]] for f in own[v][0]]) for r in reps} for v in fem}
    queries = []
    for s_ in meta["sources"]:
        sp = s_.rsplit("_", 1)[0]
        st = zstats([feats(f"real/{sp}_enroll", SP / f"a1src/{sp}_enroll.npz", sig)])
        queries.append(("RM", s_, with_z(feats(f"real/{s_}", SP / f"a1src/{s_}.npz", sig), st)))
    queries += [("IM", v, own[v][1]) for v in mal] + [("FF", v, own[v][1]) for v in fem]
    res: dict = {"prereg": "results/snv_p5/probe_prereg.yaml", "ruler": "SPARC multi の EMA 12 次元(物差しとしてだけ)", "nnc": {}, "voice": {}}
    for grp in ("RM", "IM", "FF"):
        row: dict = {r: [] for r in reps}
        for g, name, q in queries:
            if g != grp:
                continue
            qm = q["v"].copy()
            qm[np.arange(len(qm)) % SUB != 0] = False
            for v in fem:
                if grp == "FF" and v == name:
                    continue
                for r in reps:
                    nn = nn_idx(q[r][qm], pools[v][r], r in cos)
                    row[r].append(NN.corr(q["EMA"][qm], pools[v]["EMA"][nn]))
        res["nnc"][grp] = {r: round(float(np.mean(x)), 4) for r, x in row.items()} | {"n_pairs": len(row["Yz"])}
        print("NNC", grp, json.dumps(res["nnc"][grp]), flush=True)
    voices = fem + mal
    rng = np.random.default_rng(0)
    for r in reps:
        P, lab = [], []
        for i, v in enumerate(voices):
            x = np.concatenate([f[r][f["v"]] for f in own[v][0]])
            sel = rng.choice(len(x), min(PER, len(x)), replace=False)
            P.append(x[sel]), lab.append(np.full(len(sel), i))
        P, lab = np.concatenate(P), np.concatenate(lab)
        same, sex = [], []
        for i, v in enumerate(voices):
            q = own[v][1]
            nn = lab[nn_idx(q[r][q["v"]][::SUB], P, r in cos)]
            same.append(float(np.mean(nn == i)))
            sex.append(float(np.mean([(voices[j] in fem) == (v in fem) for j in nn])))
        res["voice"][r] = {"same_voice": round(float(np.mean(same)), 4), "same_sex": round(float(np.mean(sex)), 4)}
        print("voice", r, json.dumps(res["voice"][r]), flush=True)
    res["chance_same_voice"] = round(1 / len(voices), 4)
    out = ROOT / "results/snv_p5/coord_probe.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

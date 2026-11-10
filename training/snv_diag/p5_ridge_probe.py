"""P5b の探り(synth_nam_vc §8・学習なし・results/snv_p5/ridge_prereg.yaml): 演奏の候補の情報と座標の共有を、固定した線形の読み出しで分けて測る。
女声 10 声(t00〜t02 の有声)で表現 → 話者ごとに標準化した EMA 12 次元の ridge を当て、FX(見ていない女声 10 声の t09)・IM(Irodori の男声 6 声の t09)・RM(本物の男声 12 区間)で次元ごとの相関の平均。
表現は p5_coord_probe.py の z(登録の定数で標準化)と、その因果の文脈(0〜100 ms の過去を 4 フレームおきに 6 枚積む)・較正 = 凍結した P2b の c − 登録の平均。声 = 16 声の同じ声に戻る率(ユークリッド)。
    cd training && uv run --no-sync python snv_diag/p5_ridge_probe.py [P=snv_p2b_cpc,snv_p2e_cpc@step15k] [B=Yz,Lz,Fz,F3z] [W=1(c を登録の共分散で白くした w_c_* も)] [S=results/snv_p8b/pca_c_P2b.npy(PCA 32 を 0〜120 ms に積んで白くした wst_c_* も)]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p1_eval as EV
import train_snv_p1 as T
import train_snv_p2_cpc as CP
from snv_diag import p5_coord_probe as PC
from snv_diag import p8a_draw_voice as DV

ROOT = EV.ROOT
SP = PC.SP
DEV = EV.DEV
LAGS = (0, 4, 8, 12, 16, 20)
LAM = 10.0
BASE = ["Yz", "Lz", "Fz", "F3z"]


def ctx(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.concatenate([np.repeat(x[:1], l, 0), x[:len(x) - l]]) if l else x for l in LAGS], 1)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    ctags = opt.get("P", "snv_p2b_cpc").split(",")
    base = [b for b in opt.get("B", ",".join(BASE)).split(",") if b]
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    tr_us = [T.load_utt(p) for v in split["train"][::12] for p in sorted((T.FEAT / "e1" / v).glob("t0[0-4].npz"))]
    sig = np.mean([u["y"].std(1) for u in tr_us], 0).astype(np.float32)
    efs = [v for v in split["val"] if v.startswith("ef")]
    fem, fx = efs[::3], [v for v in efs if v not in efs[::3]][:10]
    mal = [v for v in split["val"] if v.startswith("em")]
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    cpcs = {}
    for t in ctags:
        name, _, ck = t.partition("@")
        st = torch.load(ROOT / "results" / name / f"{ck or 'last'}.pt", map_location="cpu", weights_only=False)
        cm = CP.CPC().to(DEV).eval()
        cm.load_state_dict(st["cpc"])
        cpcs[f"c_{t}"] = (cm, st["sig"])
    wh = opt.get("W", "0") == "1"
    Ps = np.load(ROOT / opt["S"])[:, :32] if "S" in opt else None
    reps = base + [r + "_ctx" for r in base] + list(cpcs) + ([f"w_{k}" for k in cpcs] if wh else []) + ([f"wst_{k}" for k in cpcs] if Ps is not None else [])

    def one(rel: str, ema: Path) -> dict:
        f = PC.feats(rel, ema, sig)
        for k, (cm, csig) in cpcs.items():
            with torch.no_grad():
                f[k] = cm.encode(torch.from_numpy(CP.x_of(EV.utt(rel), csig))[None].to(DEV))[1][0].cpu().numpy().T
        return f

    def finish(fs: list[dict], q: dict) -> tuple[list[dict], dict]:
        zs = PC.zstats(fs)
        cmu = {k: np.concatenate([f[k][f["v"]] for f in fs]).mean(0) for k in cpcs}
        cw = {k: DV.whiten(np.concatenate([f[k][f["v"]] for f in fs]) - cmu[k]) for k in cpcs} if wh else {}
        stk = lambda f, k: DV.lagged((f[k] - cmu[k]) @ Ps, DV.ST_LAGS)
        sw = {k: DV.whiten(np.concatenate([stk(f, k)[f["v"]] for f in fs])) for k in cpcs} if Ps is not None else {}
        out = []
        for f in fs + [q]:
            g = PC.with_z(f, zs) | {k: f[k] - cmu[k] for k in cpcs} | {f"w_{k}": (f[k] - cmu[k]) @ w for k, w in cw.items()} | {f"wst_{k}": stk(f, k) @ w for k, w in sw.items()}
            for r in base:
                g[r + "_ctx"] = ctx(g[r])
            out.append(g)
        return out[:-1], out[-1]

    own = {}
    for v in fem + fx + mal:
        fs = [one(f"e1/{v}/t0{j}", SP / f"e1/{v}/t0{j}.npz") for j in (0, 1, 2)]
        own[v] = finish(fs, one(f"e1/{v}/t09", SP / f"e1/{v}/t09.npz"))
    rm = []
    for s_ in meta["sources"]:
        sp = s_.rsplit("_", 1)[0]
        _, q = finish([one(f"real/{sp}_enroll", SP / f"a1src/{sp}_enroll.npz")], one(f"real/{s_}", SP / f"a1src/{s_}.npz"))
        rm.append(q)
    groups = {"FF": [own[v][1] for v in fem], "FX": [own[v][1] for v in fx], "IM": [own[v][1] for v in mal], "RM": rm}
    res: dict = {"prereg": "results/snv_p5/ridge_prereg.yaml", "ruler": "SPARC multi の EMA 12 次元(話者ごとに標準化・物差しとしてだけ)", "ridge": {}, "voice": {}}
    for r in reps:
        X = np.concatenate([f[r][f["v"]] for v in fem for f in own[v][0]])
        Y = np.concatenate([f["EMAz"][f["v"]] for v in fem for f in own[v][0]])
        mx, sx = X.mean(0), X.std(0) + 1e-6
        Xn = np.concatenate([(X - mx) / sx, np.ones((len(X), 1))], 1)
        W = np.linalg.solve(Xn.T @ Xn + LAM * np.eye(Xn.shape[1]), Xn.T @ Y)
        row = {}
        for g, qs in groups.items():
            cs = []
            for q in qs:
                vm = q["v"]
                pr = np.concatenate([(q[r][vm] - mx) / sx, np.ones((vm.sum(), 1))], 1) @ W
                cs.append(float(np.nanmean([np.corrcoef(pr[:, j], q["EMAz"][vm, j])[0, 1] for j in range(12)])))
            row[g] = round(float(np.mean(cs)), 4)
        row["RM/FX"] = round(row["RM"] / max(row["FX"], 1e-6), 3)
        res["ridge"][r] = row
        print("ridge", r, json.dumps(row), flush=True)
    voices = fem + mal
    rng = np.random.default_rng(0)
    for r in reps:
        P, lab = [], []
        for i, v in enumerate(voices):
            x = np.concatenate([f[r][f["v"]] for f in own[v][0]])
            sel = rng.choice(len(x), min(PC.PER, len(x)), replace=False)
            P.append(x[sel]), lab.append(np.full(len(sel), i))
        P, lab = np.concatenate(P), np.concatenate(lab)
        same = []
        for i, v in enumerate(voices):
            q = own[v][1]
            nn = lab[PC.nn_idx(q[r][q["v"]][::PC.SUB], P, False)]
            same.append(float(np.mean(nn == i)))
        res["voice"][r] = round(float(np.mean(same)), 4)
        print("voice", r, res["voice"][r], flush=True)
    out = ROOT / ("results/snv_p5/ridge_probe.json" if "P" not in opt else f"results/snv_p5/ridge_probe_{'_'.join(ctags).replace('@', '-')}{'_w' if wh else ''}{'_wst' if Ps is not None else ''}.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""D570 (P) 構音の配置: 出力の母音・子音の平均の包絡が目標本人の位置にあるか。規則は OVERNIGHT の D570。
各信号の有声のフレーム(harvest)の包絡(pae.envelope = CheapTrick の mel 包絡の DCT・c1〜c12 = 高さに鈍い次数)を、入力の C1 の硬い単位(時刻がそろう)ごとに平均し、
目標の独立の録音(<組>/d37/cen_TTT.wav・25 s・自分の C1 の単位・有声 5 フレーム以上の単位)の単位の平均との距離(dB 換算 = 8.686·‖Δc‖/√N_MEL)を、出力のフレーム数で重みづけて平均する。
版どうしの差(OURS_k − GU_k,α)を目標でまとめたブートストラップ。
    uv run python d570_place.py --sets dev2 dev3 --out ../results/conv_p0/d570_place.json
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.signal as ss
import torch

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
SYS = ["IN"] + [f"OURS{k}" for k in (5, 7, 12)] + [f"GU{k}_a{a}" for k in (5, 7, 12) for a in (100, 112, 120, 128, 136, 144)]
G: dict = {}


def init() -> None:
    torch.set_num_threads(1)
    import c1_content as CC
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"]))
    c1.load_state_dict(st1["net"])
    c1.eval()
    G["c1"], G["mf"] = c1, CC.MelFront()


def analyze(x: np.ndarray, units: bool) -> tuple:
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    n = len(x) // N.HOP
    x = x[: n * N.HOP]
    f0, _ = pw.harvest(ss.resample_poly(x.astype(np.float64), 1, 3), 16000, f0_floor=50.0, f0_ceil=1100.0, frame_period=1000.0 * N.HOP / N.SR)
    f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
    xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
    e = PA.envelope(xa, f0)[1:13]
    u = None
    if units:
        with torch.no_grad():
            u = G["c1"](G["mf"](torch.from_numpy(T1.prime(x))[None]))[..., -n:].argmax(1)[0].numpy()
        u = np.pad(u, (0, max(0, n - len(u))), mode="edge")[:n]
    return e, f0 > 0, u


def unit_means(e: np.ndarray, v: np.ndarray, u: np.ndarray, mn: int) -> dict:
    out = {}
    for j in np.unique(u[v]):
        m = v & (u == j)
        if m.sum() >= mn:
            out[int(j)] = (e[:, m].mean(1), int(m.sum()))
    return out


def one(job: tuple) -> dict:
    import nvoc as N
    from d6_unit_cov import load48
    s, k, ti = job
    S = H / s
    ct = G.setdefault("cen", {})
    if (s, ti) not in ct:
        e, v, u = analyze(load48(S / "d37" / f"cen_{ti:03d}.wav").astype(np.float32), True)
        ct[(s, ti)] = unit_means(e, v, u, 5)
    T = ct[(s, ti)]
    x = load48(S / "d56" / f"{k}_SRC.wav").astype(np.float32)[: 8 * N.SR]
    ein, vin, uin = analyze(x, True)
    row = {}
    for nm in SYS:
        if nm == "IN":
            e, v = ein, vin
        else:
            f = S / "d570" / nm / f"{k}_{nm}.wav"
            if not f.exists():
                continue
            e, v, _ = analyze(load48(f).astype(np.float32)[: len(x)], False)
        n = min(e.shape[1], len(uin))
        X = unit_means(e[:, :n], v[:n], uin[:n], 3)
        ks = [j for j in X if j in T]
        if not ks:
            continue
        w = np.array([X[j][1] for j in ks], float)
        d = np.array([np.linalg.norm(X[j][0] - T[j][0]) for j in ks]) * 8.686 / np.sqrt(N.N_MEL)
        row[nm] = float((w * d).sum() / w.sum())
    return {"key": f"{s}_{k}", "tg": f"{s}_{ti}", "d": row}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = []
    for s in a.sets:
        for r in json.loads((H / s / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if (H / s / "d570" / "OURS7" / f"{k}_OURS7.wav").exists():
                jobs.append((s, k, r["ti"]))
    jobs.sort(key=lambda j: (j[0], j[2]))
    with Pool(8, initializer=init) as pl:
        rows = list(pl.imap(one, jobs, chunksize=4))
    rng = np.random.default_rng(0)
    rep = {"n": len(rows), "mean_db": {}, "paired": {}}
    for nm in SYS:
        vals = [r["d"][nm] for r in rows if nm in r["d"]]
        if vals:
            rep["mean_db"][nm] = round(float(np.mean(vals)), 3)

    def boot(a_: str, b_: str) -> dict:
        rr = [r for r in rows if a_ in r["d"] and b_ in r["d"]]
        d = np.array([r["d"][a_] - r["d"][b_] for r in rr])
        tg = np.array([r["tg"] for r in rr])
        us = sorted(set(tg))
        idx = {u: np.flatnonzero(tg == u) for u in us}
        bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
        return {"n": len(rr), "diff": round(float(d.mean()), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}

    for k in (5, 7, 12):
        gus = {nm: rep["mean_db"][nm] for nm in rep["mean_db"] if nm.startswith(f"GU{k}_")}
        best = min(gus, key=gus.get)
        rep["paired"][f"OURS{k}-{best}(best)"] = boot(f"OURS{k}", best)
        for nm in gus:
            rep["paired"][f"OURS{k}-{nm}"] = boot(f"OURS{k}", nm)
        rep["paired"][f"OURS{k}-IN"] = boot(f"OURS{k}", "IN")
    rep["rows"] = rows
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["mean_db"]), flush=True)
    for kk, v in rep["paired"].items():
        print(kk, v, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

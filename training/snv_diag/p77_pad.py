"""段 P77 step a′(results/snv_p77/prereg.yaml・学習なし・ASR なし): 強さを源から渡す形に、回路の c0 との一致の判定(+fb:c0j<τ>)を置いたときの、伴奏だけの時間帯の出力と、本物の男声で残る強さの戻し。
A = 検証の Irodori の声(i13_bgm_dev_probe と同じ声・音源・種)の lead0・claplead15 で、形ごとに pad_db(声の時間の出力に対する、声の無い時間の出力の dB)と pad_vfrac。
    声の無い時間のフレームの、門が開く率と z = (直接の c0 − 回路の c0) / s_t の分位も出す。
B = devA の 1〜n 人目(snv_dev_eval と同じ試行・k 7)で、門が閉じたフレームの z の分布と、τ ごとの「残る戻し」= Σ relu(z)·[z ≤ τ] / Σ relu(z)(有声 / 無声)。
出力は数値だけ。
gs= を渡すと(step a″)、門の確率(持続の条件と閾の前の値)< θ のフレームだけに強さを渡す形 +fb:c0g<θ> も同じ測りに掛ける。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p77_pad.py [sys=snv_p30_c+mask:snv_p45_gate_t80] [taus=0.5,1,1.5,2,3] [gs=0.1,0.2,0.3,0.5] [n=6] [ndev=64] [out=results/snv_p77/pad.json]
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i13_bgm_dev_probe as I13
import snv_dev_eval as DE
import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV
SR = P.SR


def zof(sy: EV.System, u: dict, st: list[dict], S: dict, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, v = torch.from_numpy(u["y"])[None].to(DEV), torch.from_numpy(u["v"])[None].to(DEV)
    g = torch.from_numpy(u["gate"])[None].to(DEV) if "gate" in u else torch.zeros_like(v)
    c = np.concatenate([e["y"][0][e["v"] > 0.5] for e in (sy.prep(e) for e in st)])
    m0, s0 = float(c.mean()), max(float(c.std()), 1e-3)
    with torch.no_grad():
        yo = sy.C(sy.play(y, v, sy.op_stat(st)), torch.from_numpy(u["lf"])[None].to(DEV) + k * T.LN2_12 * v, v, S)
    mt, sd = sy.hc0[id(S)]
    z = ((float(mt) + float(sd) * (y[0, 0] - m0) / s0 - yo[0, 0]) / float(sd)).cpu().numpy()
    return z, (v[0] > 0.5).cpu().numpy(), (g[0] > 0.5).cpu().numpy(), u["gate_p"] if "gate_p" in u else np.zeros(len(z), np.float32)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    base = opt.get("sys", "snv_p30_c+mask:snv_p45_gate_t80")
    taus = [float(t) for t in opt.get("taus", "0.5,1,1.5,2,3").split(",")]
    gs = [float(t) for t in opt.get("gs", "").split(",") if t]
    forms = {"F2": ["anch"], "c0s": ["c0s", "anch"]} | {f"c0j{t:g}": [f"c0j{t:g}", "anch"] for t in taus} | {f"c0g{t:g}": [f"c0g{t:g}", "anch"] for t in gs}
    n = int(opt.get("n", 6))
    conds = [I13.parse(c) for c in opt.get("conds", "lead0,claplead15").split(",")]
    evp = json.loads((ROOT / "data/snv_sep/fsd50k_eval_sources.json").read_text())["pools"]
    import train_snv_sep as SEP
    pool = SEP.bg_split()["dev"]["music"]
    val = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())["val"]
    fem = [v for v in val if v.startswith("ef")]
    voices = [v for v in val if v.startswith("em")][:n] + fem[:: max(1, len(fem) // n)][:n]
    load = lambda p: librosa.load(str(p), sr=SR, mono=True)[0].astype(np.float64)
    gen = EV.load_gen()
    sy = EV.System(base + "+fb:anch", gen)
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:2]
    trefs = {t: [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)] for t in tgts}

    def degrade(x: np.ndarray, kind: str, lead: bool, lvl: float, src: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if kind in ("mus", "lead"):
            return I13.with_music(x, src, seed, lead, lvl)
        return I13.with_events(x, evp["clap" if kind.startswith("clap") else "op"], seed, lead, lvl, 3 if lead else 4)

    rows, zpad = [], {c: {"z": [], "open": [], "v": [], "p": []} for c, _, _, _ in conds}
    for vi, vv in enumerate(voices):
        wavs = sorted((ROOT / "data/snv_e1" / vv).glob("t*.wav"))
        xs, xr = load(wavs[0]), [load(p) for p in wavs[1:4]]
        src = pool[zlib.crc32(vv.encode()) % len(pool)]
        for name, kind, lead, lvl in conds:
            bk = kind if kind in ("mus", "lead") else kind.replace("lead", "")
            st = [I13.utt_of(degrade(x, bk, False, lvl, src, 7 + j)[1]) for j, x in enumerate(xr)]
            u = I13.utt_of(degrade(xs, kind, lead, lvl, src, 1 + vi)[1])
            a, Tn = int(I13.LEAD * SR) // P.H, len(u["v"])
            fm = np.zeros(Tn, bool)
            fm[:a] = True
            fm[(int(I13.LEAD * SR) + len(xs)) // P.H:] = True
            for tgt in [None] + tgts:
                S = sy.structure(f"self|{name}|{vv}", st) if tgt is None else sy.structure(f"t|{tgt}", trefs[tgt])
                z, v, g, gp = zof(sy, sy.prep(u), st, S, 0)
                zpad[name]["z"].append(z[fm[: len(z)] & ~g])
                zpad[name]["p"].append(gp[fm[: len(z)] & ~g])
                zpad[name]["open"].append(float(g[fm[: len(g)]].mean()))
                zpad[name]["v"].append(float(v[fm[: len(v)]].mean()))
                for fn, fb in forms.items():
                    sy.fb = fb
                    w = sy.convert(u, st, S, 0)
                    rows.append({"voice": vv, "cond": name, "form": fn, "tgt": tgt} | I13.pad_stats(np.asarray(w, np.float64), len(xs)))
            sy.scache.clear()
        print(vv, "done", flush=True)
    A = {c: {fn: {"pad_db_median": round(float(np.median([r["pad_db"] for r in rows if r["cond"] == c and r["form"] == fn])), 2),
                  "pad_db_mean": round(float(np.mean([r["pad_db"] for r in rows if r["cond"] == c and r["form"] == fn])), 2),
                  "pad_vfrac_median": round(float(np.median([r["pad_vfrac"] for r in rows if r["cond"] == c and r["form"] == fn])), 3)} for fn in forms} for c, _, _, _ in conds}
    Z = {}
    for c, d in zpad.items():
        z = np.concatenate(d["z"]) if d["z"] else np.zeros(0)
        Z[c] = {"門が開く率": round(float(np.mean(d["open"])), 3), "入力の有声の率": round(float(np.mean(d["v"])), 3), "門が閉じたフレームの数": int(len(z)),
                "z の分位(5・25・50・75・95 %)": [round(float(q), 2) for q in np.percentile(z, [5, 25, 50, 75, 95])] if len(z) else None,
                "z ≤ τ の率": {f"{t:g}": round(float((z <= t).mean()), 3) for t in taus} if len(z) else None,
                "門の確率 < θ の率(門が閉じたフレーム)": {f"{t:g}": round(float((np.concatenate(d["p"]) < t).mean()), 3) for t in gs} if len(z) else None}
    print(json.dumps({"A": A, "Z": Z}, ensure_ascii=False, indent=1), flush=True)

    items = DE.cv_items("devA", int(opt.get("ndev", 64)))
    tg4 = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:4]
    tr4 = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tg4}
    brow = []
    for it in items:
        u = sy.prep(it["u"])
        for tg in (tg4[it["idx"] % 4], tg4[(it["idx"] + 2) % 4]):
            z, v, g, gp = zof(sy, u, it["enr"], sy.structure(f"t|{tg}", tr4[tg]), 7)
            for nm, mk in (("有声", v & ~g), ("無声", ~v & ~g)):
                zz, pp = z[mk], gp[mk]
                r = np.maximum(zz, 0)
                brow.append({"spk": it["spk"], "src": it["src"], "part": nm, "n": int(mk.sum()), "relu": float(r.sum()), "zmean": float(zz.mean()) if len(zz) else 0.0,
                             "keep": {f"{t:g}": float(r[zz <= t].sum()) for t in taus}, "rej": {f"{t:g}": float((zz > t).mean()) if len(zz) else 0.0 for t in taus},
                             "keepg": {f"{t:g}": float(r[pp < t].sum()) for t in gs}, "rejg": {f"{t:g}": float((pp >= t).mean()) if len(zz) else 0.0 for t in gs}})
        sy.scache.clear()
        sy.pcache.clear()
    B = {}
    for nm in ("有声", "無声"):
        R = [r for r in brow if r["part"] == nm]
        tot = sum(r["relu"] for r in R)
        B[nm] = {"フレームの数": int(sum(r["n"] for r in R)), "relu(z) の平均": round(tot / max(sum(r["n"] for r in R), 1), 3),
                 "残る戻し": {f"{t:g}": round(sum(r["keep"][f"{t:g}"] for r in R) / max(tot, 1e-9), 3) for t in taus},
                 "z > τ の率": {f"{t:g}": round(float(np.average([r["rej"][f"{t:g}"] for r in R], weights=[max(r["n"], 1) for r in R])), 3) for t in taus},
                 "残る戻し(門の確率 < θ)": {f"{t:g}": round(sum(r["keepg"][f"{t:g}"] for r in R) / max(tot, 1e-9), 3) for t in gs},
                 "門の確率 ≥ θ の率": {f"{t:g}": round(float(np.average([r["rejg"][f"{t:g}"] for r in R], weights=[max(r["n"], 1) for r in R])), 3) for t in gs}}
    res = {"prereg": "results/snv_p77/prereg.yaml(step_a_prime)", "sys": base, "taus": taus, "voices": len(voices), "A": A, "Z": Z, "B": B, "rows_A": rows, "rows_B": brow}
    (ROOT / opt.get("out", "results/snv_p77/pad.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({"B": B}, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

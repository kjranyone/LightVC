"""段 P40 step a(results/snv_p40/prereg.yaml・学習なし): 学んだ前段の重みが声を削る量を、声の高さで分けて測る。
fam = 族の中。train_snv_mask の検証の声(results/snv_i8_m13/split.json の val)の普通の発話のうち、擬似の操作者の版(data/snv_p_wav_x16w)が在るものを n 本選び、
版の声に未使用の音楽を 10 dB・0 dB で足した入力(x_s10・x_s0)と足さない入力(x_clean)で、項目ごとに 声の残り(Σ m·P_s / Σ P_s・全帯域 / 1.5 kHz より下 / 上)・足された音の残り・帯の対数の誤差 を出す。
項目の高さ = 版の発話の前段の高さの中央値(data/snv_p_feat_x16 の f0)。115 Hz 未満 / 130 Hz 以上で分け、声で束ねた bootstrap で差の区間を出す。
devA = 開発の組 devA の有効な源(足さない録音)で、重みが有声のフレームで落とすパワー(10·log10(Σ m·P / Σ P)・全帯域 / 1.5 kHz より下 / 上)・帯の利得のフレーム間の揺れ・前段の高さの中央値・静かな 1 割のフレームの大きさ を話者ごとに出す。
出力は並びの位置(有効な話者の manifest の順)と数値だけ。Common Voice の録音は評価の入力だけに使う(学習・目標にしない・話者の特定を試みない)。
sig(step b の b2)= 本物の低い声と版の声の信号の差。devA の話者と、族の中の版の声・元の Irodori の声(足さない)で、有声のフレームの 高さの揺れ(隣り合うフレームの |Δ ln f0| の中央値と 0.2 を超える率)・有声の続く長さ・
前段の周期性(最も低い帯 / 最も高い帯)・1.5 kHz より上と下のパワーの比・倍音の櫛の見え方(網の入力の 2048 点の窓で、倍音の位置と倍音の間の位置のパワーの比)・重みが落とす量 と、落とす量と櫛・周期性のフレームごとの相関 を出す。
gate(step c)= 門の確率の分布。devA の足さない録音(源と登録)と、同じ録音に未使用の音楽を 10 dB・0 dB で足した入力(p37_cv_music と同じ混ぜ方)で、門の確率の分位と、閾を 0.5〜0.9 に替えたときの開く率(持続つき)を出す。
gfam(段 P41 の起動前の測り)= 族の中の門。検証の声の普通の発話 n 本で、足さない入力(Irodori の声・版の声・符号化)・雑音の背景 10 dB / 0 dB(当てる先は閉)・音楽 10 dB / 20 dB(当てる先は開)の、項目ごとの開く率(持続つき)を出す。
realmask(段 P44 step a の a2)= 本物の男声 + 伴奏の上の学んだ重みと理想の重み。devA の録音を S・足した音楽を N として、学んだ重みの 声の残り・足された音の残り・帯の対数の誤差(理想の比の重みとの差)を話者ごとに出す(評価の中の計算で、学習の目標にしない)。
gapsets(段 P51)= 族と本物の録音の隔たりの切り分け。声の組(本物の消費者の録音 = devA の Common Voice・本物のスタジオの録音 = VCTK の男声 6 人・族の中 = 検証の Irodori の声とその擬似の操作者の版)と、それぞれを MP3(32 kHz・48 kb/s・Common Voice と同じ形式)に通した組に、正しい音楽を 10 dB・0 dB で足し、重みの 声の残り・足された音の残り・帯の対数の誤差(理想の比の重みとの差)を、声ごとに平均して出す。
    cd training && uv run --no-sync python snv_diag/p40_pitch.py fam|devA|sig|gate|gfam|realmask|gapsets [n=240] [masks=snv_p38_mask,snv_p37_mask] [out=results/snv_p40/pitch_fam.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p0_front as P
import pae as PA
import train_snv_mask as MK
import train_snv_p1 as T
import train_snv_sep as SEP

ROOT = SEP.ROOT
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LO_BINS = SEP.N_LIN + 1
F_LO, F_HI = 115.0, 130.0


def boot_diff(a: dict, b: dict, n: int = 20000) -> list:
    g = np.random.default_rng(0)
    ka, kb = sorted(a), sorted(b)
    ma, mb = np.array([np.mean(a[k]) for k in ka]), np.array([np.mean(b[k]) for k in kb])
    d = ma[g.integers(0, len(ma), (n, len(ma)))].mean(1) - mb[g.integers(0, len(mb), (n, len(mb)))].mean(1)
    return [round(float(ma.mean() - mb.mean()), 4), [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]]


def fam(opt: dict, masks: list[str]) -> dict:
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    va = SEP.utterances(split["val"])
    fm = SEP.FAMS["p38"]
    _, pdv = SEP.make_pools(SEP.bg_split(), fm)
    ok = [i for i, (q, w) in enumerate(va) if w.parent.parent.name == "snv_e1" and SEP.xop_wav(w).exists()]
    g = np.random.default_rng(40)
    pick = [ok[i] for i in g.choice(len(ok), size=min(int(opt.get("n", 240)), len(ok)), replace=False)]
    mus = pdv["music_fma"] + pdv["music_musan"]
    specs, meta = [], []
    for ui in pick:
        q, w = va[ui]
        f0 = np.load(ROOT / "data/snv_p_feat_x16/e1" / w.parent.name / f"{w.stem}.npz")["f0"]
        src = mus[int(g.integers(len(mus)))]
        for cond, snr in (("x_s10", 10.0), ("x_s0", 0.0), ("x_clean", 0.0)):
            specs.append({"utt": int(ui), "cond": cond, "bg": "none" if cond == "x_clean" else "music", "src": src, "snr": snr, "pre": 0, "post": 0, "st": 0, "xop": True, "floor_db": -60.0})
            meta.append({"voice": w.parent.name, "f0": float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 0.0, "cond": cond})
    dl = torch.utils.data.DataLoader(SEP.Mix(va, pdv, 999, len(specs), specs, fm, parts="ev"), batch_size=6, num_workers=4)
    Ms = {t: MK.load(ROOT / "results" / t / "last.pt", DEV) for t in masks}
    rows, k = [], 0
    with torch.no_grad():
        for vb in dl:
            b = {kk: x.to(DEV) for kk, x in vb.items()}
            b = b | {"s": b["s"] + b["f"], "n": b["n"] - b["f"]}
            ps, pn, px = MK.power(b["s"]), MK.power(b["n"]), MK.power(b["x"])
            irm = ps / (ps + pn + 1e-10)
            ms = {t: torch.sigmoid(M(b["x"])).transpose(1, 2) for t, M in Ms.items()}
            for i in range(b["x"].shape[0]):
                r = dict(meta[k])
                k += 1
                for t, m in ms.items():
                    kept = lambda sl: float((m[i][:, sl] * ps[i][:, sl]).sum() / (ps[i][:, sl].sum() + 1e-10))
                    r[t] = {"voice_kept": kept(slice(None)), "voice_kept_lo": kept(slice(0, LO_BINS)), "voice_kept_hi": kept(slice(LO_BINS, None)),
                            "noise_kept": float((m[i] * pn[i]).sum() / (pn[i].sum() + 1e-10)), "lb": float(MK.band_err(m[i], irm[i], px[i]).mean())}
                r["ideal_voice_kept"] = float((irm[i] * ps[i]).sum() / (ps[i].sum() + 1e-10))
                rows.append(r)
    f0s = np.array([r["f0"] for r in rows if r["cond"] == "x_clean"])
    res: dict = {"prereg": "results/snv_p40/prereg.yaml(step_a の a1)", "n_items": len(pick), "voices": len({r["voice"] for r in rows}), "f0_pct(5,25,50,75,95)": [round(float(np.percentile(f0s, p)), 1) for p in (5, 25, 50, 75, 95)],
                 "n_low": int((f0s < F_LO).sum()), "n_high": int((f0s >= F_HI).sum()), "summary": {}, "rows": rows}
    for t in masks:
        for cond in ("x_s10", "x_s0", "x_clean"):
            for key in ("voice_kept", "voice_kept_lo", "voice_kept_hi", "noise_kept", "lb"):
                lo, hi = {}, {}
                for r in rows:
                    if r["cond"] == cond and r["f0"] > 0:
                        (lo if r["f0"] < F_LO else hi if r["f0"] >= F_HI else {}).setdefault(r["voice"], []).append(r[t][key])
                res["summary"][f"{t}|{cond}|{key}"] = {"low": round(float(np.mean([np.mean(v) for v in lo.values()])), 4), "high": round(float(np.mean([np.mean(v) for v in hi.values()])), 4), "low_minus_high": boot_diff(lo, hi), "voices": [len(lo), len(hi)]}
    for cond in ("x_s10", "x_s0"):
        lo, hi = {}, {}
        for r in rows:
            if r["cond"] == cond and r["f0"] > 0:
                (lo if r["f0"] < F_LO else hi if r["f0"] >= F_HI else {}).setdefault(r["voice"], []).append(r["ideal_voice_kept"])
        res["summary"][f"ideal|{cond}|voice_kept"] = {"low": round(float(np.mean([np.mean(v) for v in lo.values()])), 4), "high": round(float(np.mean([np.mean(v) for v in hi.values()])), 4)}
    return res


def devA(opt: dict, masks: list[str]) -> dict:
    import librosa
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())["ref"]
    Ms = {t: MK.load(ROOT / "results" / t / "last.pt", DEV) for t in masks}
    fb = torch.from_numpy(PA._np_consts()[0]).double()
    rows, pos = [], 0
    for s in [x for x in man["speakers"] if x["split"] == "devA"][: int(opt.get("n", 64))]:
        sp = s["speaker"]
        srcs = [f"{sp}_{c['name']}" for c in s["src"][:2] if dv[f"{sp}_{c['name']}"]["text_vs_orig"] <= 0.1]
        if not srcs:
            continue
        acc: dict = {}
        for name in srcs:
            x = librosa.load(str(T.FEAT / f"cv/{name}.wav"), sr=P.SR, mono=True)[0].astype(np.float32)
            Tn = P.n_frames(len(x))
            xt = torch.from_numpy(x[: Tn * P.H])
            f0 = P.f0_track(xt)[0].cpu().numpy()[:Tn]
            v = f0 > 0
            pw = MK.power(xt[None])[0].double()[:Tn]
            lev = 10 * np.log10(pw.sum(1).numpy() + 1e-12)
            acc.setdefault("f0_med", []).append(float(np.median(f0[v])) if v.any() else 0.0)
            acc.setdefault("voiced", []).append(float(v.mean()))
            acc.setdefault("pause_db", []).append(float(np.percentile(lev, 10) - np.median(lev[v])) if v.any() else 0.0)
            vt = torch.from_numpy(v)
            for t, M in Ms.items():
                with torch.no_grad():
                    p = torch.sigmoid(M(xt[None].to(DEV))[0].double()).transpose(0, 1).cpu()[:Tn]
                kd = lambda sl: float(10 * torch.log10((p[vt][:, sl] * pw[vt][:, sl]).sum() / (pw[vt][:, sl].sum() + 1e-12) + 1e-12))
                gb = 10 * torch.log10(((p * pw) @ fb.T + 1e-12) / (pw @ fb.T + 1e-12))
                dg = (gb[1:] - gb[:-1])[vt[1:] & vt[:-1]]
                for key, val in (("kept_db", kd(slice(None))), ("kept_lo_db", kd(slice(0, LO_BINS))), ("kept_hi_db", kd(slice(LO_BINS, None))), ("flick_db", float(dg.std(0).median()) if len(dg) > 1 else 0.0),
                                 ("kept_u_db", float(10 * torch.log10((p[~vt] * pw[~vt]).sum() / (pw[~vt].sum() + 1e-12) + 1e-12)) if (~vt).any() else 0.0)):
                    acc.setdefault(f"{t}|{key}", []).append(val)
        rows.append({"i": pos} | {k: round(float(np.mean(x)), 3) for k, x in acc.items()})
        pos += 1
        if pos % 10 == 0:
            print(pos, flush=True)
    return {"prereg": "results/snv_p40/prereg.yaml(step_a の a2)", "masks": masks, "rows": rows}


def sig_stats(x: np.ndarray, M: torch.nn.Module) -> dict:
    Tn = P.n_frames(len(x))
    x = np.asarray(x[: Tn * P.H], np.float32)
    xt = torch.from_numpy(x)
    f = P.analyze(x)
    f0, per = f["f0"][:Tn], f["per"][:, :Tn]
    v = f0 > 0
    if v.sum() < 20:
        return {}
    pw = MK.power(xt[None])[0].double()[:Tn].numpy()
    with torch.no_grad():
        p = torch.sigmoid(M(xt[None].to(DEV))[0].double()).transpose(0, 1).cpu()[:Tn].numpy()
    df = P.SR / SEP.NFFT
    pair = v[1:] & v[:-1]
    dl = np.abs(np.log(f0[1:][pair] / f0[:-1][pair]))
    runs, n = [], 0
    for b in v:
        if b:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    comb, rem = np.full(Tn, np.nan), np.full(Tn, np.nan)
    for t in np.where(v)[0]:
        h = np.arange(1, int(SEP.LIN_HZ / f0[t]))
        if len(h) < 3:
            continue
        on, off = np.clip(np.round(h * f0[t] / df).astype(int), 1, LO_BINS - 1), np.clip(np.round((h + 0.5) * f0[t] / df).astype(int), 1, LO_BINS - 1)
        comb[t] = 10 * np.log10((pw[t, on].mean() + 1e-14) / (pw[t, off].mean() + 1e-14))
        rem[t] = 10 * np.log10(((p[t] * pw[t]).sum() + 1e-14) / (pw[t].sum() + 1e-14))
    ok = ~np.isnan(comb)
    lo, hi = pw[v][:, :LO_BINS].sum(), pw[v][:, LO_BINS:].sum()
    cc = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 5 and np.std(a) > 0 and np.std(b) > 0 else 0.0
    return {"f0_med": float(np.median(f0[v])), "jitter": float(np.median(dl)) if len(dl) else 0.0, "jump": float((dl > 0.2).mean()) if len(dl) else 0.0, "run_med": float(np.median(runs)), "voiced": float(v.mean()),
            "per_lo": float(per[0][v].mean()), "per_hi": float(per[-1][v].mean()), "hi_lo_db": float(10 * np.log10((hi + 1e-14) / (lo + 1e-14))), "comb_db": float(np.median(comb[ok])),
            "sub_db": float(10 * np.log10((pw[v][:, 1:3].sum() + 1e-14) / (lo + 1e-14))), "kept_db": float(10 * np.log10(((p[v] * pw[v]).sum() + 1e-14) / (pw[v].sum() + 1e-14))),
            "rem_p10_db": float(np.percentile(rem[ok], 10)), "r_rem_comb": cc(rem[ok], comb[ok]), "r_rem_per": cc(rem[ok], per[0][ok])}


def sig(opt: dict, masks: list[str]) -> dict:
    import librosa
    import soundfile as sf
    M = MK.load(ROOT / "results" / masks[0] / "last.pt", DEV)
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())["ref"]
    mean = lambda rs: {k: round(float(np.mean([r[k] for r in rs])), 4) for k in rs[0]}
    real, pos = [], 0
    for s in [x for x in man["speakers"] if x["split"] == "devA"][: int(opt.get("n", 64))]:
        sp = s["speaker"]
        srcs = [f"{sp}_{c['name']}" for c in s["src"][:2] if dv[f"{sp}_{c['name']}"]["text_vs_orig"] <= 0.1]
        if not srcs:
            continue
        rs = [r for r in (sig_stats(librosa.load(str(T.FEAT / f"cv/{name}.wav"), sr=P.SR, mono=True)[0], M) for name in srcs) if r]
        real.append({"i": pos} | (mean(rs) if rs else {}))
        pos += 1
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    va = SEP.utterances(split["val"])
    ok = [i for i, (q, w) in enumerate(va) if w.parent.parent.name == "snv_e1" and SEP.xop_wav(w).exists()]
    g = np.random.default_rng(40)
    pick = [ok[i] for i in g.choice(len(ok), size=min(int(opt.get("nf", 240)), len(ok)), replace=False)]
    vers, orig = {}, {}
    for ui in pick:
        w = va[ui][1]
        for dst, path in ((vers, SEP.xop_wav(w)), (orig, w)):
            r = sig_stats(sf.read(str(path), dtype="float32")[0], M)
            if r:
                dst.setdefault(w.parent.name, []).append(r)
    return {"prereg": "results/snv_p40/prereg.yaml(step_b の b2)", "mask": masks[0], "real": real, "versions": [{"voice_i": k} | mean(rs) for k, rs in enumerate(vers.values())], "irodori": [{"voice_i": k} | mean(rs) for k, rs in enumerate(orig.values())]}


THRS = (0.5, 0.6, 0.7, 0.8, 0.9)


def open_rate(p: np.ndarray, thr: float, on: int, off: int) -> np.ndarray:
    out = np.zeros(len(p), bool)
    state, c_on, c_off = False, 0, 0
    for t, v in enumerate(p >= thr):
        c_on, c_off = (c_on + 1, 0) if v else (0, c_off + 1)
        if not state and c_on >= on:
            state = True
        elif state and c_off >= off:
            state = False
        out[t] = state
    return out


def gate(opt: dict, masks: list[str]) -> dict:
    import zlib
    import librosa
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import i13_bgm_dev_probe as B
    G = MK.load(ROOT / "results" / opt.get("gate", "snv_p38_gate") / "last.pt", DEV)
    on, off = G.hyst
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())["ref"]
    pool = SEP.bg_split()["dev"]["music"]
    load = lambda name: librosa.load(str(T.FEAT / f"cv/{name}.wav"), sr=P.SR, mono=True)[0].astype(np.float64)

    def one(x: np.ndarray) -> dict:
        Tn = P.n_frames(len(x))
        xt = torch.from_numpy(np.asarray(x[: Tn * P.H], np.float32))
        with torch.no_grad():
            p = torch.sigmoid(G.gate(xt[None].to(DEV))[0, 0].double()).cpu().numpy()[:Tn]
        return {"p10": float(np.percentile(p, 10)), "p50": float(np.median(p)), "p90": float(np.percentile(p, 90))} | {f"open{int(th * 100)}": float(open_rate(p, th, on, off).mean()) for th in THRS}

    mean = lambda rs: {k: round(float(np.mean([r[k] for r in rs])), 4) for k in rs[0]}
    rows, pos = [], 0
    for i, s in enumerate([x for x in man["speakers"] if x["split"] == "devA"][: int(opt.get("n", 64))]):
        sp = s["speaker"]
        srcs = [f"{sp}_{c['name']}" for c in s["src"][:2] if dv[f"{sp}_{c['name']}"]["text_vs_orig"] <= 0.1]
        if not srcs:
            continue
        xs = [load(n) for n in srcs]
        r = {"i": pos, "idx": i, "clean": mean([one(x) for x in xs]), "enroll": mean([one(load(f"{sp}_{c['name']}")) for c in s["enroll"][:3]])}
        if i < int(opt.get("nm", 24)):
            mus = pool[zlib.crc32(sp.encode()) % len(pool)]
            for snr in (10.0, 0.0):
                r[f"mus{int(snr)}"] = mean([one(B.with_music(x, mus, 1 + i, False, snr)[1]) for x in xs])
        rows.append(r)
        pos += 1
    summ = {}
    for th in THRS:
        k = f"open{int(th * 100)}"
        summ[k] = {"足さない: 源で開く率 > 0.05 の話者": int(sum(r["clean"][k] > 0.05 for r in rows)), "登録で > 0.05": int(sum(r["enroll"][k] > 0.05 for r in rows)), "足さない: 開く率の平均": round(float(np.mean([r["clean"][k] for r in rows])), 4)} | {
            f"伴奏 {c}: 開く率の平均 / 最小 / 0.9 未満の話者": [round(float(np.mean([r[c][k] for r in rows if c in r])), 4), round(float(np.min([r[c][k] for r in rows if c in r])), 4), int(sum(r[c][k] < 0.9 for r in rows if c in r))] for c in ("mus10", "mus0")}
    return {"prereg": "results/snv_p40/prereg.yaml(step_c)", "gate": opt.get("gate", "snv_p38_gate"), "hyst": [on, off], "n": len(rows), "n_music": int(sum("mus0" in r for r in rows)), "summary": summ, "rows": rows}


def gfam(opt: dict, masks: list[str]) -> dict:
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    va = SEP.utterances(split["val"])
    fm = SEP.FAMS[opt.get("fam", "p41")]
    _, pdv = SEP.make_pools(SEP.bg_split(), fm)
    ok = [i for i, (q, w) in enumerate(va) if w.parent.parent.name == "snv_e1" and SEP.xop_wav(w).exists()]
    g = np.random.default_rng(41)
    pick = [ok[i] for i in g.choice(len(ok), size=min(int(opt.get("n", 150)), len(ok)), replace=False)]
    mus = pdv["music_fma"] + pdv["music_musan"]
    conds = [("clean", "none", 0.0, False, False), ("x_clean", "none", 0.0, True, False), ("c_clean", "none", 0.0, False, True), ("xc_clean", "none", 0.0, True, True),
             ("n10", "noise", 10.0, False, False), ("n0", "noise", 0.0, False, False), ("x_n10", "noise", 10.0, True, False), ("x_n0", "noise", 0.0, True, False), ("xc_n10", "noise", 10.0, True, True),
             ("s10", "music", 10.0, False, False), ("x_s10", "music", 10.0, True, False), ("xc_s10", "music", 10.0, True, True), ("x_s20", "music", 20.0, True, False)]
    specs = []
    for ui in pick:
        ms, ns = mus[int(g.integers(len(mus)))], pdv["noise"][int(g.integers(len(pdv["noise"])))]
        for cond, bg, snr, xop, codec in conds:
            specs.append({"utt": int(ui), "cond": cond, "bg": bg, "src": ns if bg == "noise" else ms, "snr": snr, "pre": 0, "post": 0, "st": 0, "xop": xop, "codec": codec, "floor_db": -60.0})
    dl = torch.utils.data.DataLoader(SEP.Mix(va, pdv, 999, len(specs), specs, fm, parts="ev"), batch_size=8, num_workers=8)
    tags = opt.get("gates", "snv_i23_gate").split(",")
    Gs = {}
    for t in tags:
        st = torch.load(ROOT / "results" / t / "last.pt", map_location="cpu", weights_only=False)
        G = MK.MaskNet(st["cfg"]["ch"], 0.0, 1).to(DEV).eval()
        G.load_state_dict(st["M"])
        Gs[t] = G
    rows, k = [], 0
    with torch.no_grad():
        for vb in dl:
            b = {kk: x.to(DEV) for kk, x in vb.items()}
            lab = MK.gate_label(b).cpu().numpy()
            ps = {t: torch.sigmoid(G(b["x"])[:, 0]).cpu().numpy() for t, G in Gs.items()}
            for i in range(b["x"].shape[0]):
                r = {"cond": specs[k]["cond"], "label": float(lab[i].mean())}
                k += 1
                for t in tags:
                    r[t] = {"open": float((ps[t][i] >= 0.5).mean()), "open_h": float(open_rate(ps[t][i], 0.5, 40, 100).mean()), "p50": float(np.median(ps[t][i]))}
                rows.append(r)
    summ = {}
    for t in tags:
        for cond, bg, *_ in conds:
            rs = [r for r in rows if r["cond"] == cond]
            o = np.array([r[t]["open_h"] for r in rs])
            summ[f"{t}|{cond}"] = {"label": round(float(np.mean([r["label"] for r in rs])), 3), "open_h_mean": round(float(o.mean()), 4), "items_open>0.05": int((o > 0.05).sum()), "items_open<0.9": int((o < 0.9).sum()), "n": len(rs), "p50_median": round(float(np.median([r[t]["p50"] for r in rs])), 4)}
    return {"prereg": "results/snv_p41/prereg.yaml(起動前の測り)", "fam": opt.get("fam", "p41"), "gates": tags, "summary": summ, "rows": rows}


def realmask(opt: dict, masks: list[str]) -> dict:
    import zlib
    import librosa
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import i13_bgm_dev_probe as B
    Ms = {t: MK.load(ROOT / "results" / t / "last.pt", DEV) for t in masks}
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())["ref"]
    pool = SEP.bg_split()["dev"]["music"]
    rows, pos = [], 0
    for i, s in enumerate([x for x in man["speakers"] if x["split"] == "devA"][: int(opt.get("n", 64))]):
        sp = s["speaker"]
        srcs = [f"{sp}_{c['name']}" for c in s["src"][:2] if dv[f"{sp}_{c['name']}"]["text_vs_orig"] <= 0.1]
        if not srcs:
            continue
        mus = pool[zlib.crc32(sp.encode()) % len(pool)]
        acc: dict = {}
        for name in srcs:
            x = librosa.load(str(T.FEAT / f"cv/{name}.wav"), sr=P.SR, mono=True)[0].astype(np.float64)
            Tn = P.n_frames(len(x))
            f0 = P.f0_track(torch.from_numpy(x[: Tn * P.H].astype(np.float32)))[0].cpu().numpy()[:Tn]
            v = torch.from_numpy(f0 > 0)
            acc.setdefault("f0_med", []).append(float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 0.0)
            for snr in (10.0, 0.0):
                c, z = B.with_music(x, mus, 1 + i, False, snr)
                ct, zt = torch.from_numpy(c[: Tn * P.H].astype(np.float32)), torch.from_numpy(z[: Tn * P.H].astype(np.float32))
                ps, pn, px = MK.power(ct[None])[0].double()[:Tn], MK.power((zt - ct)[None])[0].double()[:Tn], MK.power(zt[None])[0].double()[:Tn]
                irm = ps / (ps + pn + 1e-10)
                key = f"mus{int(snr)}"
                acc.setdefault(f"{key}|ideal|voice_kept", []).append(float((irm * ps).sum() / (ps.sum() + 1e-10)))
                acc.setdefault(f"{key}|ideal|noise_kept", []).append(float((irm * pn).sum() / (pn.sum() + 1e-10)))
                for t, M in Ms.items():
                    with torch.no_grad():
                        m = torch.sigmoid(M(zt[None].to(DEV))[0].double()).transpose(0, 1).cpu()[:Tn]
                    kv = lambda fr, sl: float((m[fr][:, sl] * ps[fr][:, sl]).sum() / (ps[fr][:, sl].sum() + 1e-10))
                    for nm, val in (("voice_kept", kv(slice(None), slice(None))), ("voice_kept_v", kv(v, slice(None))), ("voice_kept_lo", kv(v, slice(0, LO_BINS))), ("voice_kept_hi", kv(v, slice(LO_BINS, None))),
                                    ("noise_kept", float((m * pn).sum() / (pn.sum() + 1e-10))), ("noise_kept_v", float((m[v] * pn[v]).sum() / (pn[v].sum() + 1e-10))), ("noise_kept_u", float((m[~v] * pn[~v]).sum() / (pn[~v].sum() + 1e-10)) if (~v).any() else 0.0),
                                    ("lb", float(MK.band_err(m.float(), irm.float(), px.float()).mean())), ("lb_v", float(MK.band_err(m.float(), irm.float(), px.float())[v].mean()))):
                        acc.setdefault(f"{key}|{t}|{nm}", []).append(val)
        rows.append({"i": pos, "idx": i} | {k: round(float(np.mean(x)), 4) for k, x in acc.items()})
        pos += 1
    f0 = np.array([r["f0_med"] for r in rows])
    summ = {}
    for k in rows[0]:
        if "|" in k:
            a_ = np.array([r[k] for r in rows])
            summ[k] = {"all": round(float(a_.mean()), 4), "low(<115)": round(float(a_[f0 < F_LO].mean()), 4), "other": round(float(a_[f0 >= F_LO].mean()), 4)}
    return {"prereg": "results/snv_p44/prereg.yaml(step_a の a2)", "masks": masks, "n": len(rows), "n_low": int((f0 < F_LO).sum()), "summary": summ, "rows": rows}


def mask_stats(M: torch.nn.Module, x: np.ndarray, mus: str, seed: int) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import i13_bgm_dev_probe as B
    Tn = P.n_frames(len(x))
    f0 = P.f0_track(torch.from_numpy(x[: Tn * P.H].astype(np.float32)))[0].cpu().numpy()[:Tn]
    v = torch.from_numpy(f0 > 0)
    out = {"f0_med": float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 0.0}
    for snr in (10.0, 0.0):
        c, z = B.with_music(x, mus, seed, False, snr)
        ct, zt = torch.from_numpy(c[: Tn * P.H].astype(np.float32)), torch.from_numpy(z[: Tn * P.H].astype(np.float32))
        ps, pn, px = MK.power(ct[None])[0].double()[:Tn], MK.power((zt - ct)[None])[0].double()[:Tn], MK.power(zt[None])[0].double()[:Tn]
        irm = ps / (ps + pn + 1e-10)
        with torch.no_grad():
            m = torch.sigmoid(M(zt[None].to(DEV))[0].double()).transpose(0, 1).cpu()[:Tn]
        key = f"mus{int(snr)}"
        out[f"{key}|voice_kept"] = float((m * ps).sum() / (ps.sum() + 1e-10))
        out[f"{key}|noise_kept"] = float((m * pn).sum() / (pn.sum() + 1e-10))
        out[f"{key}|ideal_voice_kept"] = float((irm * ps).sum() / (ps.sum() + 1e-10))
        out[f"{key}|ideal_noise_kept"] = float((irm * pn).sum() / (pn.sum() + 1e-10))
        out[f"{key}|lb"] = float(MK.band_err(m.float(), irm.float(), px.float()).mean())
        out[f"{key}|lb_v"] = float(MK.band_err(m.float(), irm.float(), px.float())[v].mean()) if v.any() else 0.0
        hz = np.arange(MK.NB) * P.SR / SEP.NFFT
        for bn, (lo, hi) in (("b0", (0, 1500)), ("b1", (1500, 5000)), ("b2", (5000, 11000)), ("b3", (11000, 24001))):
            sl = torch.from_numpy((hz >= lo) & (hz < hi))
            for fn, fr in (("v", v), ("u", ~v)):
                if fr.any():
                    den = ps[fr][:, sl].sum()
                    out[f"{key}|kept_{bn}{fn}"] = float((m[fr][:, sl] * ps[fr][:, sl]).sum() / (den + 1e-10))
                    out[f"{key}|ideal_{bn}{fn}"] = float((irm[fr][:, sl] * ps[fr][:, sl]).sum() / (den + 1e-10))
                    out[f"{key}|share_{bn}{fn}"] = float(den / (ps.sum() + 1e-10))
    return out


def gapsets(opt: dict, masks: list[str]) -> dict:
    import zlib
    import librosa
    import soundfile as sf
    M = MK.load(ROOT / "results" / masks[0] / "last.pt", DEV)
    pool = SEP.bg_split()["dev"]["music"]
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())["ref"]
    cv = [(s["speaker"], T.FEAT / f"cv/{s['speaker']}_{c['name']}.wav") for s in [x for x in man["speakers"] if x["split"] == "devA"][:64] for c in s["src"][:2] if dv[f"{s['speaker']}_{c['name']}"]["text_vs_orig"] <= 0.1]
    vctk = [(q.name.split("_")[0], q) for q in sorted((T.FEAT / "vctk").glob("*.wav"))]
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    va = SEP.utterances(split["val"])
    ok = [w for (q, w) in va if w.parent.parent.name == "snv_e1" and SEP.xop_wav(w).exists()]
    g = np.random.default_rng(51)
    pick = [ok[i] for i in g.choice(len(ok), size=min(int(opt.get("nf", 72)), len(ok)), replace=False)]
    iro = [(w.parent.name, w) for w in pick]
    xop = [(w.parent.name, SEP.xop_wav(w)) for w in pick]
    sets = {"cv": (cv, False), "vctk": (vctk, False), "vctk_mp3": (vctk, True), "irodori": (iro, False), "irodori_mp3": (iro, True), "version": (xop, False), "version_mp3": (xop, True)}
    res: dict = {"prereg": "results/snv_p51/prereg.yaml", "mask": masks[0], "music_read": "full", "sets": {}}
    for name, (items, mp3) in sets.items():
        rows = []
        for k, (vid, path) in enumerate(items):
            x = librosa.load(str(path), sr=P.SR, mono=True)[0].astype(np.float64)
            if mp3:
                x = SEP.codec_mp3(x).astype(np.float64)
            mus = pool[zlib.crc32(f"{name.split('_')[0]}|{vid}|{k}".encode()) % len(pool)]
            rows.append({"voice": vid} | mask_stats(M, x, mus, 1 + k))
        per = {}
        for r in rows:
            per.setdefault(r["voice"], []).append(r)
        summ = {}
        for key in rows[0]:
            if "|" in key:
                summ[key] = round(float(np.mean([np.mean([r[key] for r in rs]) for rs in per.values()])), 4)
        summ["f0_med"] = round(float(np.median([r["f0_med"] for r in rows])), 1)
        res["sets"][name] = {"n_items": len(rows), "n_voices": len(per), "summary": summ}
        print(name, len(rows), len(per), {k: summ[k] for k in ("mus10|lb", "mus0|lb", "mus0|voice_kept", "mus0|noise_kept", "mus0|ideal_voice_kept", "mus0|ideal_noise_kept")}, flush=True)
    return res


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    mode = [x for x in sys.argv[1:] if "=" not in x][0]
    masks = opt.get("masks", "snv_p38_mask,snv_p37_mask").split(",")
    res = fam(opt, masks) if mode == "fam" else sig(opt, masks) if mode == "sig" else gate(opt, masks) if mode == "gate" else gfam(opt, masks) if mode == "gfam" else realmask(opt, masks) if mode == "realmask" else gapsets(opt, masks) if mode == "gapsets" else devA(opt, masks)
    out = ROOT / opt.get("out", f"results/snv_p40/pitch_{mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "real", "versions", "irodori")}, ensure_ascii=False, indent=1)[:6000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

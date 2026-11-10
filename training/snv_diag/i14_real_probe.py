"""段 I14 の後の切り分け(学習なし): 前段 D(paper_snv §5 M20)の出力と前段 v3 の特徴の食い違いを、試験の本物の区間(外郎売・namikawa・登録)・JVNV(スタジオの本物の男声)・検証の Irodori の声のきれいな音で並べる。
VCTK(開発の組の本物の男声・英語)も同じ物差しで並べる。伴奏の無い区間では前段 v3 の特徴がきれいな音の値 = 食い違いは D の代価。量 = 包絡の L1(c0・c1〜c12・c13〜c24・周期性 4)・有声の一致と率・高さの誤差(両方有声のフレームのセントの中央値・100 セントを越える率・符号つきの中央値)。
録音の性質 = 長時間の平均スペクトルの帯域ごとの割合(dB)・静かなフレーム(c0 の下 10 %)と有声のフレームの c0 の差。
    cd training && SNV_DEV=cpu uv run --no-sync python snv_diag/i14_real_probe.py [sep=snv_i14_sep[@ck]] [n_iro=4] [groups=vctk,jvnv,...] [out=results/snv_i14/real_probe.json]
tf=id,lp16,lp11,lf,lp16+lf = 検証の Irodori と JVNV の音を乱して(lpN = N kHz より上を消す・lf = 20〜120 Hz の雑音を音声の −15 dB で足す)、D 対 乱した音の前段 v3 と、前段 v3 の動き(元の音 対 乱した音)を並べる(out 既定 results/snv_i14/real_probe_tf.json)。
mask=<tag>[@ck] = 前段 D の代わりに、前段 v3 に時間周波数の重みを掛けた特徴(train_snv_mask・paper_snv §5 M23)を前段 v3 と比べる(重みの恒等さ)。
mode=lowf0 = 学習の男声のうち高さの中央値が 130 Hz 未満の声(各 3 本)で D 対 前段 v3(out 既定 results/snv_i14/real_probe_lowf0.json)。
mode=hpf = 前段 v3 に 4 次の Butterworth の高域通過(60・80 Hz・因果)を掛けたときの有声の率と反転と包絡の動き(外郎売・namikawa_3〜5・VCTK・JVNV・検証の Irodori・out 既定 results/snv_i14/front_v3_hpf.json)。
mode=dhpf = D の入力にだけ同じ高域通過(60・100 Hz)を掛けた D 対 元の音の前段 v3(D の外れが 100 Hz より下の音から来るか・out 既定 results/snv_i14/d_hpf.json)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p0_front as P
import train_snv_p1 as T
import train_snv_sep as SEP

ROOT = Path(__file__).resolve().parents[2]
FEAT = T.FEAT
FEAT_FN = SEP.features
JV = ROOT / "data/jvnv/jvnv_v1"
WAVDIR = SEP.WAVDIR
BANDS = ((0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 16000), (16000, 24000))
CENTS = 1200 / np.log(2)


def ltas(x: np.ndarray) -> dict:
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=240, window="hann", center=False)) ** 2
    fr = np.fft.rfftfreq(2048, 1 / P.SR)
    p = S.mean(1)
    tot = p.sum() + 1e-20
    return {f"{lo}-{hi}": round(float(10 * np.log10(p[(fr >= lo) & (fr < hi)].sum() / tot + 1e-12)), 2) for lo, hi in BANDS}


def compare(ref: dict, d: dict) -> dict:
    n = min(ref["y"].shape[1], d["y"].shape[1])
    yr, yd = ref["y"][:, :n], d["y"][:, :n]
    vr, vd = ref["v"][:n] > 0.5, d["v"][:n] > 0.5
    both = vr & vd
    dl = (d["lf"][:n] - ref["lf"][:n])[both] * CENTS
    c0 = yr[0]
    q = np.quantile(c0, 0.1)
    return {
        "frames": int(n),
        "y_l1": float(np.abs(yd - yr).mean()),
        "y_c0": float(np.abs(yd[0] - yr[0]).mean()),
        "y_c1_12": float(np.abs(yd[1:13] - yr[1:13]).mean()),
        "y_c13_24": float(np.abs(yd[13:25] - yr[13:25]).mean()),
        "y_per": float(np.abs(yd[25:29] - yr[25:29]).mean()),
        "y_l1_voiced": float(np.abs(yd[:, vr] - yr[:, vr]).mean()) if vr.any() else None,
        "y_l1_unvoiced": float(np.abs(yd[:, ~vr] - yr[:, ~vr]).mean()) if (~vr).any() else None,
        "c0_bias_voiced": float((yd[0, vr] - yr[0, vr]).mean()) if vr.any() else None,
        "vacc": float((vr == vd).mean()),
        "vfrac_ref": float(vr.mean()),
        "vfrac_d": float(vd.mean()),
        "miss": float((vr & ~vd).sum() / max(1, vr.sum())),
        "false": float((~vr & vd).sum() / max(1, (~vr).sum())),
        "cents_med": float(np.median(np.abs(dl))) if both.any() else None,
        "cents_gt100": float((np.abs(dl) > 100).mean()) if both.any() else None,
        "cents_signed": float(np.median(dl)) if both.any() else None,
        "f0_med_ref": float(np.median(ref["f0"][:n][vr])) if vr.any() else None,
        "floor_c0": float(np.mean(c0[c0 <= q]) - np.mean(c0[vr])) if vr.any() else None,
    }


def agg(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        xs = [r[k] for r in rows if r[k] is not None]
        if xs and isinstance(xs[0], (int, float)):
            out[k] = round(float(np.median(xs)), 4)
    return out


def transform(x: np.ndarray, tf: str, seed: int) -> np.ndarray:
    y = x.astype(np.float64)
    for t in tf.split("+"):
        if t.startswith("lp"):
            X = np.fft.rfft(y)
            X[np.fft.rfftfreq(len(y), 1 / P.SR) > float(t[2:]) * 1000] = 0
            y = np.fft.irfft(X, len(y))
        elif t == "lf":
            g = np.random.default_rng(seed)
            N = np.fft.rfft(g.standard_normal(len(y)))
            fr = np.fft.rfftfreq(len(y), 1 / P.SR)
            N[(fr < 20) | (fr > 120)] = 0
            n = np.fft.irfft(N, len(y))
            y = y + n * np.sqrt((y ** 2).mean() / ((n ** 2).mean() + 1e-20)) * 10 ** (-15 / 20)
    return np.clip(y, -1, 1).astype(np.float32)


def utt_of(x: np.ndarray) -> dict:
    n = P.n_frames(len(x))
    f = P.analyze(x[: n * P.H].astype(np.float32))
    f0, env, per = f["f0"], f["env"], f["per"]
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    return {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0.astype(np.float32)}


def run_tf(D, name: str, x: np.ndarray, tf: str, dev: str) -> dict:
    n = P.n_frames(len(x))
    x = x[: n * P.H]
    xt = transform(x, tf, zlib_seed(name))
    ref, reft = utt_of(x), utt_of(xt)
    d = SEP.features(D, xt, dev)
    return {"name": name} | compare(reft, d) | {"v3_shift": compare(ref, reft), "d_vs_clean": compare(ref, d)}


def zlib_seed(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode())


def lowf0(D, sname: str, dev: str) -> dict:
    split = json.loads((ROOT / "results" / sname / "split.json").read_text())
    rows = []
    for v in split["train"]:
        if not v.startswith("em"):
            continue
        ps = sorted((FEAT / "e1" / v).glob("t*.npz"))[:3]
        f0 = np.concatenate([np.load(p)["f0"] for p in ps])
        if np.median(f0[f0 > 0]) >= 130:
            continue
        for p in ps:
            rows.append(run(D, f"{v}/{p.stem}", T.load_utt(p), librosa.load(str(WAVDIR["e1"] / v / f"{p.stem}.wav"), sr=P.SR, mono=True)[0], dev))
    return {"summary": agg(rows) | {"n": len(rows)}, "rows": rows}


def hpf() -> dict:
    from scipy.signal import butter, sosfilt
    sets = {"uirou": [FEAT / "real" / f"uirou_{i}.wav" for i in range(6)],
            "namikawa_nonbgm": [FEAT / "real" / f"namikawa_{i}.wav" for i in (3, 4, 5)],
            "vctk": sorted((FEAT / "vctk").glob("*.wav"))[::4],
            "jvnv": [JV / p.stem.split("_")[0] / p.stem.split("_")[1] / "regular" / f"{p.stem}.wav" for p in sorted((FEAT / "jvnv").glob("*.npz"))[::3]],
            "irodori_val": [WAVDIR["e1"] / v / f"t0{j}.wav" for v in ("em594", "em595", "em596", "em597", "em598", "em599", "ef000", "ef016", "ef032") for j in (0, 1)]}
    res: dict = {}
    for fc in (60, 80):
        sos = butter(4, fc, "highpass", fs=P.SR, output="sos")
        for name, ws in sets.items():
            r = []
            for w in ws:
                x = librosa.load(str(w), sr=P.SR, mono=True)[0]
                a, b = utt_of(x), utt_of(sosfilt(sos, x.astype(np.float64)).astype(np.float32))
                n = min(a["y"].shape[1], b["y"].shape[1])
                va, vb = a["v"][:n] > 0.5, b["v"][:n] > 0.5
                r.append({"vfrac": float(va.mean()), "vfrac_hp": float(vb.mean()), "flip": float((va != vb).mean()), "y_l1": float(np.abs(a["y"][:, :n] - b["y"][:, :n]).mean()), "c0": float(np.abs(a["y"][0, :n] - b["y"][0, :n]).mean())})
            res[f"hp{fc}|{name}"] = {k: round(float(np.median([x[k] for x in r])), 4) for k in r[0]} | {"n": len(r)}
            print(f"hp{fc}|{name}", res[f"hp{fc}|{name}"], flush=True)
    return res


def dhpf(D, dev: str) -> dict:
    from scipy.signal import butter, sosfilt
    sets = {"vctk": [(p.stem, p.with_suffix(".wav")) for p in sorted((FEAT / "vctk").glob("*.npz"))[::3]],
            "jvnv": [(p.stem, JV / p.stem.split("_")[0] / p.stem.split("_")[1] / "regular" / f"{p.stem}.wav") for p in sorted((FEAT / "jvnv").glob("*.npz"))[::3]],
            "real_nonbgm": [(n, FEAT / "real" / f"{n}.wav") for n in [f"uirou_{i}" for i in range(6)] + [f"namikawa_{i}" for i in (3, 4, 5)]],
            "irodori_val": [(f"{v}/t0{j}", WAVDIR["e1"] / v / f"t0{j}.wav") for v in ("em594", "em595", "em596", "em597", "em598", "em599", "ef000", "ef016") for j in (0, 1)]}
    res: dict = {}
    for fc in (0, 60, 100):
        for g, items in sets.items():
            rows = []
            for nm, w in items:
                x = librosa.load(str(w), sr=P.SR, mono=True)[0]
                n = P.n_frames(len(x))
                x = x[: n * P.H]
                ref = utt_of(x)
                xh = x if fc == 0 else sosfilt(butter(4, fc, "highpass", fs=P.SR, output="sos"), x.astype(np.float64)).astype(np.float32)
                rows.append(compare(ref, SEP.features(D, xh, dev)))
            res[f"hp{fc}|{g}"] = agg(rows) | {"n": len(rows)}
            s_ = res[f"hp{fc}|{g}"]
            print(f"hp{fc}|{g}", {k: s_.get(k) for k in ("y_l1", "y_c0", "y_c1_12", "vacc", "miss", "false", "cents_med", "cents_signed")}, flush=True)
    return res


def run(D, name: str, ref: dict, x: np.ndarray, dev: str) -> dict:
    n = ref["y"].shape[1]
    x = x[: n * P.H]
    if len(x) < n * P.H:
        x = np.pad(x, (0, n * P.H - len(x)))
    d = FEAT_FN(D, x.astype(np.float32), dev)
    return {"name": name} | compare(ref, d) | {"ltas": ltas(x)} | ({"mask_mean": d["mask_mean"]} if "mask_mean" in d else {})


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    sname, _, sck = opt.get("sep", "snv_i14_sep").partition("@")
    dev = "cpu"
    import torch
    torch.set_num_threads(int(opt.get("threads", 8)))
    if opt.get("mode") == "hpf":
        (ROOT / opt.get("out", "results/snv_i14/front_v3_hpf.json")).write_text(json.dumps(hpf(), ensure_ascii=False, indent=1))
        return 0
    global FEAT_FN
    if "mask" in opt:
        import train_snv_mask as MK
        mname, _, mck = opt["mask"].partition("@")
        D, FEAT_FN = MK.load(ROOT / "results" / mname / f"{mck or 'last'}.pt", dev), MK.features
        if "dec" in opt:
            D.dec = opt["dec"]
    else:
        D = SEP.load(ROOT / "results" / sname / f"{sck or 'last'}.pt", dev)
    if opt.get("mode") == "dhpf":
        (ROOT / opt.get("out", "results/snv_i14/d_hpf.json")).write_text(json.dumps(dhpf(D, dev), ensure_ascii=False, indent=1))
        return 0
    if opt.get("mode") == "lowf0":
        r = lowf0(D, sname, dev)
        print("lowf0", json.dumps(r["summary"], ensure_ascii=False), flush=True)
        (ROOT / opt.get("out", "results/snv_i14/real_probe_lowf0.json")).write_text(json.dumps(r, ensure_ascii=False, indent=1))
        return 0
    if "tf" in opt:
        split = json.loads((ROOT / "results" / sname / "split.json").read_text())
        vals = [v for v in split["val"] if v.startswith("em")] + [v for v in split["val"] if v.startswith("ef")][:6]
        srcs = {"irodori_val": [(f"{v}/{p.stem}", WAVDIR["e1"] / v / f"{p.stem}.wav") for v in vals for p in sorted((FEAT / "e1" / v).glob("t*.npz"))[: int(opt.get("n_iro", 2))]],
                "jvnv": [(p.stem, JV / p.stem.split("_")[0] / p.stem.split("_")[1] / "regular" / f"{p.stem}.wav") for p in sorted((FEAT / "jvnv").glob("*.npz"))]}
        res: dict = {"sep": opt.get("sep", "snv_i14_sep"), "summary": {}, "rows": {}}
        for tf in opt["tf"].split(","):
            for g, items in srcs.items():
                rows = [run_tf(D, nm, librosa.load(str(w), sr=P.SR, mono=True)[0], tf, dev) for nm, w in items if w.exists()]
                key = f"{g}|{tf}"
                res["rows"][key] = rows
                res["summary"][key] = agg(rows) | {"n": len(rows), "v3_shift": agg([r["v3_shift"] for r in rows]), "d_vs_clean": agg([r["d_vs_clean"] for r in rows])}
                s = res["summary"][key]
                pick = lambda d: {k: d.get(k) for k in ("y_l1", "y_c0", "y_c1_12", "y_c13_24", "y_per", "vacc", "miss", "false", "cents_med", "cents_gt100", "cents_signed")}
                print(key, "D対v3(乱した音)", pick(s), "| v3 の動き", pick(s["v3_shift"]), flush=True)
        (ROOT / opt.get("out", "results/snv_i14/real_probe_tf.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    groups: dict = {"real_nonbgm": [], "real_bgm": [], "real_enroll": [], "jvnv": [], "vctk": [], "irodori_val": []}
    only = set(opt["groups"].split(",")) if "groups" in opt else set(groups)
    for p in sorted((FEAT / "vctk").glob("*.npz")) if "vctk" in only else []:
        if p.with_suffix(".wav").exists():
            groups["vctk"].append(run(D, p.stem, T.load_utt(p), librosa.load(str(p.with_suffix(".wav")), sr=P.SR, mono=True)[0], dev))
    for sp in ("uirou", "namikawa") if only & {"real_nonbgm", "real_bgm", "real_enroll"} else ():
        for i in range(6):
            nm = f"{sp}_{i}"
            g = "real_bgm" if sp == "namikawa" and i < 3 else "real_nonbgm"
            groups[g].append(run(D, nm, T.load_utt(FEAT / "real" / f"{nm}.npz"), librosa.load(str(FEAT / "real" / f"{nm}.wav"), sr=P.SR, mono=True)[0], dev))
        nm = f"{sp}_enroll"
        groups["real_enroll"].append(run(D, nm, T.load_utt(FEAT / "real" / f"{nm}.npz"), librosa.load(str(FEAT / "real" / f"{nm}.wav"), sr=P.SR, mono=True)[0], dev))
    for p in sorted((FEAT / "jvnv").glob("*.npz")) if "jvnv" in only else []:
        spk, emo = p.stem.split("_")[:2]
        w = JV / spk / emo / "regular" / f"{p.stem}.wav"
        if w.exists():
            groups["jvnv"].append(run(D, p.stem, T.load_utt(p), librosa.load(str(w), sr=P.SR, mono=True)[0], dev))
    split = json.loads((ROOT / "results" / sname / "split.json").read_text())
    n_iro = int(opt.get("n_iro", 4))
    vals = [v for v in split["val"] if v.startswith("em")] + [v for v in split["val"] if v.startswith("ef")][:6]
    for v in vals if "irodori_val" in only else []:
        for p in sorted((FEAT / "e1" / v).glob("t*.npz"))[:n_iro]:
            w = WAVDIR["e1"] / v / f"{p.stem}.wav"
            if w.exists():
                groups["irodori_val"].append(run(D, f"{v}/{p.stem}", T.load_utt(p), librosa.load(str(w), sr=P.SR, mono=True)[0], dev))
    res = {"sep": opt.get("sep", "snv_i14_sep"), "summary": {}, "ltas": {}, "rows": groups}
    for g, rows in groups.items():
        if not rows:
            continue
        res["summary"][g] = agg(rows) | {"n": len(rows)}
        res["ltas"][g] = {b: round(float(np.median([r["ltas"][b] for r in rows])), 2) for b in rows[0]["ltas"]}
        print(g, json.dumps(res["summary"][g], ensure_ascii=False), flush=True)
        print("  ltas", json.dumps(res["ltas"][g]), flush=True)
    for r in groups["real_nonbgm"] + groups["real_bgm"] + groups["real_enroll"]:
        print(r["name"], {k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in ("y_l1", "y_c0", "y_c1_12", "y_c13_24", "y_per", "vacc", "vfrac_ref", "vfrac_d", "cents_med", "cents_gt100", "cents_signed", "f0_med_ref", "floor_c0")}, flush=True)
    out = ROOT / opt.get("out", "results/snv_i14/real_probe.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

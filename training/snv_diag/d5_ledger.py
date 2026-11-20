"""段 D5(paper_snv rev41 §6・results/snv_d5/prereg.yaml・学習なし): 覆いの台帳。声ごとに軸の量を測り、開発の組の本物の男声が合成の族の域に入る割合(cover)と、族が開発の組の域に入る割合(mass)を数える。
量: 高さ(f0_med・f0_p5・f0_span = 100 Hz からの半音)/ 声道と音色(vtl = 有声の平均の包絡の 300〜5000 Hz の形(傾きを除く)を E1 の男声の平均へ合わせる周波数の伸縮 α = E1 の男声の平均に対する声道の長さの比・
tilt = 有声の 1〜3 kHz − 300〜1 kHz dB・vtl_lpc = LPC のフォルマントの間隔から cm(lpc=1 のときだけ測る・記録 results/snv_d5/ledger_lpc)。vtl は 2 つの物差しとも較正を通らなかった(results/snv_d5/ledger_calib*.json)ので、覆いの主の量に入れない)/
話し方(rate = 有声 1 秒あたりの c0 の山・段 I10 と同じ定義・rate_lv = 100〜6000 Hz の音量の山・lvl_span)/ 前段 v3 の特徴の上の録音の量(floor・env_q_*・env_v_*・per0〜3・vq・fall = 有声の終わりの後に 20 dB 以上落ちる所で −5 dB から −15 dB までの時間)/ hum(静かな時間の 50 / 60 Hz の線の突出)。
組: cv(Common Voice の開発の男声・split を持つ)・jvnv・vctk・e1(F0)・f1(M18 の族 = data/snv_p_feat_rec2 の版 + 包絡の伸縮 α と傾き・vtl は e1 の値 / α)・d3(段 D3 の dv)・
dir:<名>:<data の下の置き場>({声}/{発話}.wav)。声ごとの量は results/snv_d5/ledger/{組}.json。
    cd training && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python snv_diag/d5_ledger.py pops=e1,cv,jvnv,vctk,f1,d3 [lpc=1](e1 を最初に = vtl の基準)
    cd training && uv run --no-sync python snv_diag/d5_ledger.py mode=calib
    cd training && uv run --no-sync python snv_diag/d5_ledger.py mode=mora      # 文のある組に mora_rate(モーラ / 最初と最後の有声の間の秒)を足す
    cd training && uv run --no-sync python snv_diag/d5_ledger.py mode=table fams=e1,f1 dev=cv:devA [out=results/snv_d5/ledger.json]
"""
from __future__ import annotations

import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pae as PA
import snv_p0_front as P
import snv_p0c_anchor as AN

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/snv_p_feat"
OUT = ROOT / "results/snv_d5/ledger"
SR = P.SR
BANDS = ((0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 16000), (16000, 24000))
BN = ("b0", "b100", "b300", "b1k", "b3k", "b6k", "b10k", "b16k")
E1_UTT = ("t03", "t04", "t05", "t06", "t07", "t08")
AXES = {"pitch": ["f0_med", "f0_p5"], "tract": ["tilt"], "speak": ["rate", "f0_span", "lvl_span"],
        "record": ["floor"] + [f"env_q_{b}" for b in BN] + [f"env_v_{b}" for b in BN] + [f"per{i}" for i in range(4)] + ["fall", "hum"]}
_C: dict = {}


def consts() -> tuple:
    if not _C:
        _, dct = PA._np_consts()
        fm = AN.mel_hz()
        _C["D"] = dct[:25].T * (20 / np.log(10))
        _C["grp"] = [np.flatnonzero((fm >= lo) & (fm < hi)) for lo, hi in BANDS]
        _C["core"] = np.flatnonzero((fm >= 300) & (fm < 3000))
        _C["wide"] = np.flatnonzero((fm >= 100) & (fm < 6000))
        _C["tilt"] = (dct[:25] @ np.linspace(-0.5, 0.5, dct.shape[0])).astype(np.float64)
    return _C["D"], _C["grp"], _C["core"], _C["wide"]


def level(L: np.ndarray) -> np.ndarray:
    return 10 * np.log10((10 ** (L[consts()[3]] / 10)).mean(0))


def feat_stats(fs: list[dict]) -> dict:
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    D, grp, core, _ = consts()
    f0 = np.concatenate([f["f0"] for f in fs]).astype(np.float64)
    L = D @ np.concatenate([f["env"] for f in fs], 1).astype(np.float64)
    per = np.clip(np.concatenate([f["per"] for f in fs], 1), 0, 1.5)
    v = f0 > 0
    if v.sum() < 200:
        return {}
    lv = level(L)
    q = lv <= np.percentile(lv, 10)
    vcore = float(L[core][:, v].mean())
    st = lambda f: float(12 * np.log2(f / 100.0))
    fv = f0[v]
    r = {"f0_med": st(np.median(fv)), "f0_p5": st(np.percentile(fv, 5)), "f0_span": st(np.percentile(fv, 95)) - st(np.percentile(fv, 5)),
         "floor": float(np.median(lv[q]) - np.median(lv[v])), "lvl_span": float(np.percentile(lv[v], 95) - np.percentile(lv[v], 5)),
         "vq": float(v[q].mean()), "vrate": float(v.mean()), "tilt": float(L[grp[3]][:, v].mean() - L[grp[2]][:, v].mean())}
    for nme, g in zip(BN, grp):
        r[f"env_v_{nme}"] = float(L[g][:, v].mean() - vcore)
        r[f"env_q_{nme}"] = float(L[g][:, q].mean() - vcore)
    for i in range(4):
        r[f"per{i}"] = float(per[i, v].mean())
    r["mu"] = np.concatenate([f["env"] for f in fs], 1)[:, v].astype(np.float64).mean(1).round(4).tolist()
    pk = pk2 = 0
    vs, falls = 0.0, []
    for f in fs:
        vu = f["f0"] > 0
        lu = level(D @ f["env"].astype(np.float64))
        pk += len(find_peaks(np.where(vu, uniform_filter1d(f["env"][0].astype(np.float64), 5), -1e9), prominence=3.0, distance=12)[0])
        pk2 += len(find_peaks(np.where(vu, uniform_filter1d(lu, 5), -1e9), prominence=4.0, distance=12)[0])
        vs += vu.sum() / 200.0
        sm = uniform_filter1d(lu, 3)
        for t in np.flatnonzero(vu[:-1] & ~vu[1:]):
            seg = sm[t:t + 61]
            if len(seg) == 61 and seg.min() <= seg[0] - 20:
                falls.append((int(np.argmax(seg <= seg[0] - 15)) - int(np.argmax(seg <= seg[0] - 5))) * 5.0)
    r["rate"], r["rate_lv"] = pk / vs, pk2 / vs
    r["fall"] = float(np.median(falls)) if len(falls) >= 3 else float("nan")
    return r


VTL_GRID = np.round(np.arange(0.70, 1.501, 0.01), 2)


def vtl_fit(mu: np.ndarray, ref: np.ndarray) -> float:
    _, dct = PA._np_consts()
    fm = AN.mel_hz()
    band = (fm >= 300) & (fm <= 5000)
    X = np.stack([np.ones(int(band.sum())), np.arange(int(band.sum()), dtype=np.float64)], 1)
    det = lambda L: L - X @ np.linalg.lstsq(X, L, rcond=None)[0]
    Lt = det((dct[:25].T @ np.asarray(ref, np.float64))[band])
    return float(min(VTL_GRID, key=lambda a: float(np.mean((det((dct[:25].T @ AN.warp_env(np.asarray(mu, np.float32), float(a)).astype(np.float64))[band]) - Lt) ** 2))))


def vtl_ref() -> np.ndarray:
    return np.load(OUT / "vtl_ref.npy")


def vtl_of(xs: list[np.ndarray], f0s: list[np.ndarray], cap: int = 1500) -> float:
    win = np.hamming(250)
    n_all = sum(int((f > 0).sum()) for f in f0s)
    step = max(1, n_all // cap)
    out = []
    for x, f0 in zip(xs, f0s):
        y = librosa.resample(x.astype(np.float64), orig_sr=SR, target_sr=10000)
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])
        for t in np.flatnonzero(f0 > 0)[::step]:
            s = t * 50 - 100
            if s < 0 or s + 250 > len(y):
                continue
            seg = y[s:s + 250] * win
            if float((seg ** 2).sum()) < 1e-10:
                continue
            try:
                a = librosa.lpc(seg, order=10)
            except Exception:
                continue
            if not np.all(np.isfinite(a)):
                continue
            rt = np.roots(a)
            rt = rt[np.imag(rt) > 0]
            fr = np.angle(rt) * 10000 / (2 * np.pi)
            bw = -10000 / np.pi * np.log(np.abs(rt) + 1e-12)
            F = np.sort(fr[(fr > 200) & (fr < 4900) & (bw < 600)])
            if len(F) >= 4:
                i = np.arange(1, 5)
                out.append(35000.0 / (4 * float((F[:4] * (2 * i - 1)).sum() / ((2 * i - 1) ** 2).sum())))
    return float(np.median(out)) if len(out) >= 50 else float("nan")


def hum_of(xs: list[np.ndarray]) -> float:
    from scipy.signal import stft
    pws = []
    for x in xs:
        if len(x) < 8192:
            continue
        f, _, Z = stft(x.astype(np.float64), SR, nperseg=8192, noverlap=6144, window="hann", boundary=None, padded=False)
        pws.append(np.abs(Z) ** 2)
    if not pws:
        return float("nan")
    pw = np.concatenate(pws, 1)
    e = pw[(f >= 100) & (f < 6000)].sum(0)
    d = 10 * np.log10(pw[:, e <= np.percentile(e, 20)].mean(1) + 1e-20)

    def line(f0: float) -> float:
        vals = []
        for h in (1, 2, 3):
            c = int(np.argmin(np.abs(f - h * f0)))
            vals.append(float(d[c - 1:c + 2].max() - np.median(np.r_[d[c - 6:c - 2], d[c + 3:c + 7]])))
        return float(np.mean(vals))

    return max(line(50.0), line(60.0))


def load48(p: Path) -> np.ndarray:
    return librosa.load(str(p), sr=SR, mono=True)[0].astype(np.float32)


def feat_of(pop: str, voice: str, wav: Path, x: np.ndarray | None = None) -> dict:
    std = {"e1": FEAT / "e1" / voice / f"{wav.stem}.npz", "cv": FEAT / "cv" / f"{wav.stem}.npz", "jvnv": FEAT / "jvnv" / f"{wav.stem}.npz"}.get(pop.split(":")[0])
    dst = std if std is not None else FEAT / "d5_cache" / (pop.split(":")[1] if pop.startswith("dir:") else pop) / f"{voice}_{wav.stem}.npz"
    if not dst.exists():
        x = load48(wav) if x is None else x
        f = P.analyze(x[: P.n_frames(len(x)) * P.H])
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **{k: v.astype(np.float32) for k, v in f.items()})
    d = np.load(dst)
    return {k: d[k] for k in ("f0", "env", "per")}


def work(job: tuple) -> tuple:
    pop, voice, wavs, extra, lpc = job
    torch.set_num_threads(1)
    xs = [load48(Path(w)) for w in wavs]
    fs = [feat_of(pop, voice, Path(w), x) for w, x in zip(wavs, xs)]
    r = feat_stats(fs)
    if r:
        r["hum"] = hum_of(xs)
        if lpc:
            r["vtl_lpc"] = vtl_of(xs, [f["f0"] for f in fs])
    return voice, r | extra


def work_f1(job: tuple) -> tuple:
    voice, k = job
    g = np.random.default_rng(zlib.crc32(f"d5f1|{voice}|{k}".encode()))
    alpha = float(np.exp(np.linspace(np.log(0.75), np.log(1.33), 33))[g.integers(33)])
    tau = float(g.uniform(-1, 1))
    consts()
    fs = []
    for t in E1_UTT:
        p = ROOT / f"data/snv_p_feat_rec2/e1/{voice}/{t}_k{k}.npz"
        if p.exists():
            d = np.load(p)
            fs.append({"f0": d["f0"], "per": d["per"], "env": (AN.warp_env(d["env"], alpha) + tau * _C["tilt"][:, None]).astype(np.float32)})
    r = feat_stats(fs) if fs else {}
    if r:
        r["hum"] = float("nan")
    return f"{voice}_k{k}", r | {"alpha": alpha, "tau": tau, "sex": "M" if voice.startswith("em") else "F"}


def jobs(pop: str) -> list:
    if pop == "cv":
        man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
        return [(pop, s["speaker"], [str(FEAT / f"cv/{s['speaker']}_{c['name']}.wav") for c in s["enroll"] + s["src"]], {"split": s["split"], "age": s["age"]}) for s in man["speakers"]]
    if pop == "jvnv":
        JV = ROOT / "data/jvnv/jvnv_v1"
        return [(pop, s, [str(p) for p in sorted(JV.glob(f"{s}/*/regular/*_0[12].wav"))], {}) for s in ("M1", "M2")]
    if pop == "vctk":
        V = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
        males = [ln.split()[0] for ln in (V / "speaker-info.txt").read_text().splitlines()[1:] if len(ln.split()) > 2 and ln.split()[2] == "M"]
        out = []
        for s in males:
            fs = sorted((V / "wav48" / f"p{s}").glob("*.wav"))
            if len(fs) >= 38:
                out.append((pop, f"p{s}", [str(p) for p in fs[30:38]], {}))
        return out
    if pop == "e1":
        return [(pop, d.name, [str(d / f"{t}.wav") for t in E1_UTT], {"sex": "M" if d.name.startswith("em") else "F"}) for d in sorted((ROOT / "data/snv_e1").glob("e[fm]*"))]
    if pop == "d3":
        return [(pop, d.name, [str(p) for p in sorted(d.glob("t*.wav"))], {"sex": "M"}) for d in sorted((ROOT / "data/snv_d3").glob("dv*"))]
    if pop.startswith("dir:"):
        _, _, rel = pop.split(":")
        return [(pop, d.name, [str(p) for p in sorted(d.glob("*.wav"))], {}) for d in sorted((ROOT / "data" / rel).iterdir()) if d.is_dir()]
    raise SystemExit(f"unknown pop {pop}")


def path_of(pop: str) -> Path:
    return OUT / f"{pop.split(':')[1] if pop.startswith('dir:') else pop}.json"


def measure(pop: str, lpc: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    with ProcessPoolExecutor(11) as ex:
        if pop == "f1":
            js = [(d.name, k) for d in sorted((ROOT / "data/snv_p_feat_rec2/e1").glob("e[fm]*")) for k in (0, 1)]
            it = ex.map(work_f1, js, chunksize=8)
        else:
            it = ex.map(work, [j + (lpc,) for j in jobs(pop)], chunksize=2)
        for i, (voice, r) in enumerate(it):
            res[voice] = r
            if (i + 1) % 100 == 0:
                print(pop, i + 1, flush=True)
    if pop == "e1":
        np.save(OUT / "vtl_ref.npy", np.mean([r["mu"] for r in res.values() if r.get("sex") == "M" and "mu" in r], 0))
    ref = vtl_ref()
    for r in res.values():
        if "mu" in r:
            r["vtl"] = vtl_fit(np.array(r["mu"]), ref)
    path_of(pop).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(pop, "done", len(res), "空", sum(1 for r in res.values() if "f0_med" not in r), flush=True)


def vals(pop: str, key: str, filt: str | None = None) -> np.ndarray:
    name, _, sub = pop.partition(":") if not pop.startswith("dir:") else (pop, "", "")
    d = json.loads(path_of(name).read_text())
    ok = lambda r: sub == "" or r.get("split") == sub or r.get("sex") == sub
    x = np.array([r.get(key, np.nan) for r in d.values() if "f0_med" in r and ok(r)], float)
    return x[np.isfinite(x)]


def cover(dev: np.ndarray, fam: np.ndarray) -> dict:
    if len(dev) < 2 or len(fam) < 10:
        return {"cover": None, "mass": None, "ok": None}
    lo, hi = np.percentile(fam, [2.5, 97.5])
    dl, dh = np.percentile(dev, [5, 95])
    c, m = float(((dev >= lo) & (dev <= hi)).mean()), float(((fam >= dl) & (fam <= dh)).mean())
    return {"cover": round(c, 3), "mass": round(m, 3), "ok": bool(c >= 0.90 and m >= 0.20), "fam_lo": round(float(lo), 3), "fam_med": round(float(np.median(fam)), 3), "fam_hi": round(float(hi), 3),
            "dev_p5": round(float(dl), 3), "dev_med": round(float(np.median(dev)), 3), "dev_p95": round(float(dh), 3)}


def table(fams: list[str], dev: str, out: Path) -> None:
    res: dict = {"dev": dev, "spec": "cover ≥ 0.90 かつ mass ≥ 0.20", "families": {}}
    for fam in fams:
        t: dict = {"axes": {}}
        for ax, keys in AXES.items():
            rows = {k: cover(vals(dev, k), vals(fam, k)) for k in keys}
            bad = [k for k, r in rows.items() if r["ok"] is False]
            t["axes"][ax] = {"covered": not bad and all(r["ok"] is not None for r in rows.values()), "uncovered": bad, "unmeasured": [k for k, r in rows.items() if r["ok"] is None], "stats": rows}
        t["extra"] = {k: cover(vals(dev, k), vals(fam, k)) for k in ("mora_rate", "rate_lv", "vq")}
        res["families"][fam] = t
        print("==", fam, "対", dev, flush=True)
        for ax, a in t["axes"].items():
            print(f"  {ax}: {'覆えている' if a['covered'] else '覆えていない'} {a['uncovered']} 未測定 {a['unmeasured']}")
            for k, r in a["stats"].items():
                if r["ok"] is not None:
                    print(f"    {k:12s} cover {r['cover']:.2f} mass {r['mass']:.2f}  族 [{r['fam_lo']:.2f}, {r['fam_med']:.2f}, {r['fam_hi']:.2f}]  開発 [{r['dev_p5']:.2f}, {r['dev_med']:.2f}, {r['dev_p95']:.2f}]")
        for k, r in t["extra"].items():
            if r["ok"] is not None:
                print(f"  (追加・仕様の外){k:10s} cover {r['cover']:.2f} mass {r['mass']:.2f}  族 [{r['fam_lo']:.2f}, {r['fam_med']:.2f}, {r['fam_hi']:.2f}]  開発 [{r['dev_p5']:.2f}, {r['dev_med']:.2f}, {r['dev_p95']:.2f}]")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))


def mora(pops: list[str]) -> None:
    from snv_a0_measure import kana
    small = set("ャュョァィゥェォヮ")
    nm = lambda t: sum(1 for c in kana(t) if c not in small)
    text: dict = {}
    for s_ in json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())["speakers"]:
        text[("cv", s_["speaker"])] = [(FEAT / f"cv/{s_['speaker']}_{c['name']}.npz", c["sentence"]) for c in s_["enroll"] + s_["src"]]
    for pop, mp, sub in (("e1", "data/snv_e1/manifest.json", "e1"), ("d3", "data/snv_d3/manifest.json", None)):
        for v in json.loads((ROOT / mp).read_text())["voices"]:
            fs = [f for f in v["files"] if pop != "e1" or f["t"] in E1_UTT]
            text[(pop, v["voice"])] = [((FEAT / "e1" / v["voice"] / f"{f['t']}.npz") if pop == "e1" else (FEAT / "d5_cache" / pop / f"{v['voice']}_{f['t']}.npz"), f["text"]) for f in fs]
    jt = {ln.split("|")[0]: ln.split("|")[2].strip() for ln in (ROOT / "data/jvnv/jvnv_v1/transcription.csv").read_text().splitlines() if ln.count("|") >= 2}
    for spk in ("M1", "M2"):
        text[("jvnv", spk)] = [(q, jt[q.stem.split("_", 1)[1]]) for q in sorted((FEAT / "jvnv").glob(f"{spk}_*_regular_0[12].npz")) if q.stem.split("_", 1)[1] in jt]
    for pop in pops:
        d = json.loads(path_of(pop).read_text())
        for voice, r in d.items():
            n, sec = 0, 0.0
            for q, t in text.get((pop, voice), []):
                if not q.exists():
                    continue
                v = np.flatnonzero(np.load(q)["f0"] > 0)
                if len(v) > 20:
                    n += nm(t)
                    sec += (v[-1] - v[0] + 1) / 200.0
            if sec > 0:
                r["mora_rate"] = round(n / sec, 3)
        path_of(pop).write_text(json.dumps(d, ensure_ascii=False, indent=1))
        x = np.array([r["mora_rate"] for r in d.values() if "mora_rate" in r])
        print(pop, len(x), "モーラ / 秒 [5 %, 50 %, 95 %]", np.percentile(x, [5, 50, 95]).round(2).tolist(), flush=True)


def calib() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import snv_d5_world as W
    import snv_rec_feat as RF
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    males = [v for v in split["train"] if v.startswith("em")][:20]
    res: dict = {}
    al = (0.9, 1.0, 1.1, 1.2)
    est = {a: [] for a in al}
    ref = vtl_ref()
    for v in males:
        x = load48(ROOT / f"data/snv_e1/{v}/t03.wav")
        for a in al:
            y = W.shift(x, SR, 1.0, a)
            f = P.analyze(y[: P.n_frames(len(y)) * P.H])
            est[a].append(vtl_fit(f["env"][:, f["f0"] > 0].astype(np.float64).mean(1), ref))
    E = np.array([est[a] for a in al])
    ok = np.isfinite(E).all(0)
    lr = np.log(E[:, ok] / E[1:2, ok])
    slope = float(np.polyfit(np.repeat(np.log(al), ok.sum()), lr.reshape(-1), 1)[0])
    res["vtl"] = {"alpha": list(al), "median_ratio": [round(float(np.nanmedian(e)), 3) for e in E], "slope_loglog": round(slope, 3), "n": int(ok.sum())}
    if path_of("e1").exists():
        m, f = vals("e1:M", "vtl"), vals("e1:F", "vtl")
        res["vtl"]["auc_male_gt_female"] = round(float(np.mean(m[:, None] > f[None, :])), 3)
        res["vtl"]["e1_male_median"], res["vtl"]["e1_female_median"] = round(float(np.median(m)), 2), round(float(np.median(f)), 2)
    res["vtl"]["pass"] = bool(0.7 <= slope <= 1.3 and res["vtl"].get("auc_male_gt_female", 1.0) >= 0.85)
    print("vtl", res["vtl"], flush=True)
    rts = (0.0, 0.15, 0.3, 0.5, 0.8)
    fl = {r: [] for r in rts}
    hm = {d: [] for d in (None, -50, -40, -30, -20)}
    for v in males:
        x = load48(ROOT / f"data/snv_e1/{v}/t03.wav").astype(np.float64)
        for r in rts:
            y = x
            if r > 0:
                h = RF.rir({"rt60": r, "drr_db": 3.0, "rir_seed": 1})
                y = np.fft.irfft(np.fft.rfft(x, len(x) + len(h)) * np.fft.rfft(h, len(x) + len(h)), len(x) + len(h))[: len(x)]
            f = P.analyze(y[: P.n_frames(len(y)) * P.H].astype(np.float32))
            fl[r].append(feat_stats([f]).get("fall", np.nan))
        rms = np.sqrt((x ** 2).mean())
        tt = np.arange(len(x)) / SR
        hsig = np.sin(2 * np.pi * 50 * tt) + 0.5 * np.sin(2 * np.pi * 100 * tt) + 0.3 * np.sin(2 * np.pi * 150 * tt)
        hsig /= np.sqrt((hsig ** 2).mean())
        for d in hm:
            hm[d].append(hum_of([(x if d is None else x + hsig * rms * 10 ** (d / 20)).astype(np.float32)]))
    fm = [float(np.nanmedian(fl[r])) for r in rts]
    res["fall"] = {"rt60": list(rts), "median_ms": fm, "pass": bool(all(b > a for a, b in zip(fm, fm[1:])))}
    hmm = [float(np.nanmedian(hm[d])) for d in hm]
    res["hum"] = {"level_db": ["none", -50, -40, -30, -20], "median_db": [round(h, 2) for h in hmm], "pass": bool(all(b > a for a, b in zip(hmm, hmm[1:])))}
    print("fall", res["fall"], "\nhum", res["hum"], flush=True)
    (ROOT / "results/snv_d5/ledger_calib.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    mode = opt.get("mode", "measure")
    if mode == "calib":
        calib()
    elif mode == "mora":
        mora(opt.get("pops", "cv,e1,d3,jvnv").split(","))
    elif mode == "table":
        table(opt.get("fams", "e1,f1").split(","), opt.get("dev", "cv:devA"), ROOT / opt.get("out", "results/snv_d5/ledger.json"))
    else:
        for pop in opt.get("pops", "e1,cv,jvnv,vctk,f1,d3").split(","):
            measure(pop, opt.get("lpc") == "1")
    return 0


if __name__ == "__main__":
    sys.exit(main())

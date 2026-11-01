"""D109(学習なし): 変換向けの軸ごとの関門。オーナー 10-11『解析的に合格しなければ、そもそも人間に聴かせないで下さい』・耳の不合格の理由 = ガビガビ・ざらつき / 目標の人に聞こえない / 抑揚・f0 がおかしい / こもり・機械的。
D94/D95(WavLM の二標本検定)は男声の時間の流れまで拾うので変換の合格線に使えない(D107)。ここでは話し手のリズムに依らない発話ごとの統計を軸ごとに作り、本物の女声の分布で閾値を先に固定する。
  f0 軸(全部の組を同じ推定器 = F1 + 確信度 0.5 で測る): jump = 有声区間内の 0.6 oct 超の跳び / 有声秒・range = 有声の半音の標準偏差・micro = 隣接する有声フレームの |Δ半音| の中央値・vfrac = 有声 / 発声中のフレーム
  ガビ軸: contrast = 倍音間コントラスト(eval_zsvc・有声・500–3000Hz)・amline = フレーム周期の包絡変調の線(eval_nvoc)・shimmer = 隣接する有声フレーム(10ms 窓)の |Δ音量 dB| の中央値(v2)
  こもり軸: hf = 4–10kHz / 0.1–4kHz のエネルギー比 dB(発声中)・tilt = 1/3 oct の長時間スペクトルの傾き dB/oct(300Hz–8kHz)・lmstd = log-mel 40 帯(〜10kHz)の時間方向の標準偏差の平均(機械的 = 動きの乏しさ)
  目標との差(目標の参照がある組だけ): ltas_d = 1/3 oct の長時間スペクトル(水準をそろえる)と目標の登録の参照との RMS 差 dB・dtilt / dhf = tilt / hf の目標の参照との差(v2: こもり = 目標より暗い)
  本物の分布の目標との差 = 同じ話者の他の発話(最大 6 本・30s)を参照に。v2 の統計は v1 の結果(ガビ・こもり軸が prodcand の C05/C07 を通した = 盲目)を受けて足した。
本物の分布 = 評価に使わない実の女声(female-dataset・元が 44.1kHz 以上)--n_pool 人の 1 発話(8s)。閾値は先に固定(調整しない):
  各統計で 組の発話のうち本物の [p2.5, p97.5](片側の統計は片側)の外が 15% 以下 かつ 組の中央値が本物の [p15, p85] の内。軸 = その統計が全部合格。
妥当性の確認: 本物の半分で閾値を作り残り半分・目標の実音声 cen が合格・耳の合格の写し合成(d61 ORA・d68 DYN・d62 TEACHER)が合格・耳の失格(d66 TAB・d73・d85_360k)が 1 軸以上で失格・
d85_360k C05/C07(prodcand_ab1・オーナーの理由 = 4 つ全部)が ガビ・f0・こもり軸で失格。同一性は別(d84 の学習外 LDA)。出力 = results/conv_p0/<tag>.json。
    uv run python d109_axis_gate.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
SETS = [("d37", "cen", True), ("d61", "ORA", True), ("d68", "DYN", True), ("d62", "TEACHER", True), ("d66", "TAB", False), ("d73", "BTAB", False),
        ("d73", "BW07", False), ("d73", "W10", False), ("d85_360k", "C05", False), ("d85_360k", "C07", False), ("d85_360k", "SRC", None)]
SIDE = {"jump": "hi", "range": "both", "micro": "both", "vfrac": "both", "contrast": "both", "amline": "hi", "shimmer": "both", "hf": "both", "tilt": "both", "lmstd": "both",
        "ltas_d": "hi", "dtilt": "both", "dhf": "both"}
AXES = {"f0": ["jump", "range", "micro", "vfrac"], "gabi": ["contrast", "amline", "shimmer"], "muffled": ["hf", "tilt", "lmstd"], "target": ["ltas_d", "dtilt", "dhf"]}
VER = 2
SEC = 8.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--n_pool", type=int, default=150)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--tag", default="d109")
    ap.add_argument("--no_base", action="store_true")
    a = ap.parse_args()
    import scipy.signal as ss
    import soundfile as sf
    import f0est as FE
    import nvoc as N
    from d6_unit_cov import load48
    from eval_nvoc import amline
    from eval_zsvc import contrast
    S = Path(a.s)
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())
    fps = N.SR / N.HOP
    cf = 1000.0 * 2.0 ** (np.arange(-10, 11) / 3.0)
    cf = cf[(cf >= 100) & (cf <= 10000)]
    fb = None

    def f0track(x: np.ndarray) -> np.ndarray:
        n = len(x) // N.HOP
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * N.HOP])[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        k = pr[:, :m].argmax(0)
        ii = np.clip(k[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        f = np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12))
        return np.where(vo[:m] & (c >= 0.5), f, 0.0)

    def ltas(x: np.ndarray, act: np.ndarray) -> np.ndarray:
        fq, _, Z = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
        P = np.abs(Z[:, : len(act)]) ** 2
        P = P[:, act[: P.shape[1]]].mean(1)
        out = np.array([P[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() for c in cf])
        d = 10 * np.log10(out + 1e-20)
        return d - d.mean()

    def active(x: np.ndarray) -> np.ndarray:
        n = len(x) // N.HOP
        e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
        return e > e.max() - 40

    def stats(x: np.ndarray, ref: np.ndarray | None) -> dict:
        nonlocal fb
        x = x[: int(SEC * N.SR)].astype(np.float32)
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        f = f0track(x)
        act = active(x)[: len(f)]
        f = f[: len(act)]
        v = f > 0
        st = 12 * np.log2(np.where(v, f, 1.0))
        both = v[1:] & v[:-1]
        dst = np.abs(np.diff(st))[both]
        r = {"jump": float((dst > 7.2).sum() / max(v.sum() / fps, 1e-3)), "range": float(st[v].std()) if v.sum() > 10 else float("nan"),
             "micro": float(np.median(dst)) if len(dst) else float("nan"), "vfrac": float(v.sum() / max(act.sum(), 1))}
        r["contrast"] = contrast(x.astype(np.float64), f.astype(np.float64))
        r["amline"] = amline(x)
        fq, _, Z = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
        P = np.abs(Z[:, : len(act)]) ** 2
        Pa = P[:, act[: P.shape[1]]]
        r["hf"] = float(10 * np.log10(Pa[(fq >= 4000) & (fq < 10000)].sum() / Pa[(fq >= 100) & (fq < 4000)].sum()))
        L = ltas(x, act)
        sel = (cf >= 300) & (cf <= 8000)
        r["tilt"] = float(np.polyfit(np.log2(cf[sel]), L[sel], 1)[0])
        if fb is None:
            import librosa
            fb = librosa.filters.mel(sr=N.SR, n_fft=2048, n_mels=40, fmin=50, fmax=10000)
        lm = 10 * np.log10(fb @ Pa + 1e-10)
        r["lmstd"] = float(lm.std(1).mean())
        fw = np.lib.stride_tricks.sliding_window_view(np.r_[np.zeros(N.HOP, np.float32), x], 2 * N.HOP)[:: N.HOP][: len(act)]
        ed = 10 * np.log10((fw * np.hanning(2 * N.HOP)).__pow__(2).mean(1) + 1e-12)
        vv = v[: len(ed)]
        de = np.abs(np.diff(ed))[vv[1:] & vv[:-1]]
        r["shimmer"] = float(np.median(de)) if len(de) else float("nan")
        if ref is not None:
            ra = active(ref)
            Lr = ltas(ref, ra)
            r["ltas_d"] = float(np.sqrt(((L - Lr) ** 2).mean()))
            r["dtilt"] = r["tilt"] - float(np.polyfit(np.log2(cf[sel]), Lr[sel], 1)[0])
            fr2, _, Zr = ss.stft(ref, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
            Pr = (np.abs(Zr[:, : len(ra)]) ** 2)[:, ra[: Zr.shape[1]]]
            r["dhf"] = r["hf"] - float(10 * np.log10(Pr[(fr2 >= 4000) & (fr2 < 10000)].sum() / Pr[(fr2 >= 100) & (fr2 < 4000)].sum()))
        else:
            r["ltas_d"] = r["dtilt"] = r["dhf"] = float("nan")
        return r

    cpath = S / "d109_cache.json"
    cache = json.loads(cpath.read_text()) if cpath.exists() else {}

    def cstats(key: str, xf, reff) -> dict:
        k = f"{VER}|{key}"
        if k not in cache:
            cache[k] = stats(xf(), reff())
        return cache[k]

    def norm(y: np.ndarray) -> np.ndarray:
        return (y / (np.sqrt((y ** 2).mean()) + 1e-9) * 0.05).astype(np.float32)

    refs = {}

    def tref(t: int) -> np.ndarray:
        if t not in refs:
            refs[t] = norm(load48(S / "d37/_work" / f"ref_{t:03d}.wav")[: int(30 * N.SR)])
        return refs[t]

    excl = set((S / "excl.txt").read_text().split()) if (S / "excl.txt").exists() else set()
    base = ROOT / "female-dataset"
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(109).shuffle(pool)
    real = []
    for spk in pool:
        if len(real) >= a.n_pool:
            break
        ws = sorted((base / spk).glob("*.wav"))
        if len(ws) < 3 or any(sf.info(str(w)).samplerate < 44100 for w in ws[:3]):
            continue
        ck = f"{VER}|pool|{spk}|{ws[len(ws) // 2].name}"
        if ck in cache:
            real.append(cache[ck])
            continue
        x = load48(ws[len(ws) // 2])
        if len(x) < 3 * N.SR:
            continue
        rest = np.concatenate([load48(w) for w in ws if w != ws[len(ws) // 2]][:6])[: int(30 * N.SR)]
        if len(rest) < 8 * N.SR:
            continue
        real.append(cstats(f"pool|{spk}|{ws[len(ws) // 2].name}", lambda: x, lambda: norm(rest)))
        if len(real) % 25 == 0:
            print("pool", len(real), flush=True)
    keys = list(SIDE)
    R = {k: np.array([r[k] for r in real]) for k in keys}

    def judge(vals: dict, Rk: dict) -> dict:
        out = {}
        for k in keys:
            rv, sv = Rk[k][np.isfinite(Rk[k])], vals[k][np.isfinite(vals[k])]
            if len(sv) == 0:
                continue
            lo, hi, l15, h15 = np.percentile(rv, [2.5, 97.5, 15, 85])
            md = float(np.median(sv))
            if SIDE[k] == "hi":
                of, mok = float(np.mean(sv > hi)), md <= h15
            else:
                of, mok = float(np.mean((sv < lo) | (sv > hi))), l15 <= md <= h15
            out[k] = {"median": round(md, 3), "real_p15_p85": [round(float(l15), 3), round(float(h15), 3)], "out_frac": round(of, 3), "pass": bool(of <= 0.15 and mok)}
        ax = {n: all(out[k]["pass"] for k in ks if k in out) for n, ks in AXES.items() if any(k in out for k in ks)}
        return {"stats": out, "axes": ax, "pass": all(ax.values())}

    rep = {}
    idx = np.arange(len(real))
    h = idx % 2 == 0
    rep["real_half_B"] = {"ear_pass": True, **judge({k: R[k][~h] for k in keys}, {k: R[k][h] for k in keys})}
    print("real_half_B", rep["real_half_B"]["axes"], flush=True)
    cpath.write_text(json.dumps(cache))
    sets = ([] if a.no_base else list(SETS)) + [(e.split(":")[0], e.split(":")[1], None) for e in a.extra]
    for d, c, ok in sets:
        rows = []
        pat = "cen_*.wav" if c == "cen" else f"*_{c}.wav"
        for f in sorted((S / d).glob(pat)):
            m = re.search(r"t(\d+)", f.name.split("_")[1]) if f.name.startswith("p") else re.search(r"_(\d+)", f.name)
            t = int(m.group(1)) if m else -1
            rows.append(cstats(f"set|{f}|{f.stat().st_mtime_ns}", lambda: load48(f), lambda: tref(t) if 0 <= t < 30 else None))
        vals = {k: np.array([r[k] for r in rows]) for k in keys}
        rep[f"{d}:{c}"] = {"ear_pass": ok, "n": len(rows), **judge(vals, R)}
        print(f"{d}:{c}", ok, rep[f"{d}:{c}"]["axes"], {k: v["median"] for k, v in rep[f"{d}:{c}"]["stats"].items()}, flush=True)
        cpath.write_text(json.dumps(cache))
    rep["real_pool_median"] = {k: round(float(np.nanmedian(R[k])), 3) for k in keys}
    lab = [v for v in rep.values() if isinstance(v, dict) and v.get("ear_pass") is not None]
    rep["valid"] = {"pass_sets_pass": all(v["pass"] for v in lab if v["ear_pass"]), "fail_sets_fail": all(not v["pass"] for v in lab if v["ear_pass"] is False)}
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("valid", rep["valid"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

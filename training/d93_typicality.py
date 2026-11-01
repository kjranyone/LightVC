"""D93(学習なし): 解析の合格 = 本物の女声の分布の内側か(典型性)。オーナー指示 2026-10-11『解析的に合格しなければ人間に聴かせない』。
本物の女声(female-dataset・評価の 150 人プールと D80 の目標を除く・1 話者 1 発話 8s)から特徴(約 50 次元)を取り、縮約つきガウスに当てて
マハラノビス距離 d² を出す。合格線 = 当てに使っていない本物の d² の 95 パーセンタイル。音量は特徴の前に RMS 0.05 に揃える。
特徴: 周期性 4 帯の平均と時間の標準偏差・倍音間コントラスト 5 帯・変調スペクトル(4 mel 群 × 5 変調帯)・長時間平均スペクトル 8 mel 群(全体を引く = 傾き / こもり)・
包絡の動き(帯 RMS 3・共分散の正規化固有値の上位 5)・f0(有声率・有声区間内の log f0 の隣差の中央値・0.6 oct 超の跳び / 秒)。
検証(耳のラベル): 合格 = d61 ORA・d66 DYN・d68 DYN・d62 全条件 / 失格 = d61 TABENV・d66 TAB/TABPF/INS・d68 TAB・d73 変換 3 条件・prodcand(d85_360k の C05・C07)。
出力 = results/conv_p0/d93.json(本物の d² の分布・合格線・刺激ごとの d²・合否と耳の一致)。
    uv run python d93_typicality.py --s <scratchpad> --workers 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
LEVEL = 0.05
G: dict = {}


def init() -> None:
    from d78_spk_ruler import init as i0
    i0()
    from d78_spk_ruler import G as G0
    G.update(G0)


def feat(path: str) -> np.ndarray | None:
    import librosa
    import soundfile as sf
    import torch
    from scipy.signal import butter, sosfiltfilt
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    from eval_zsvc import contrast
    from d77_ruler import logmel, modspec
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 48000:
        x = librosa.resample(x, orig_sr=sr, target_sr=48000).astype(np.float32)
    x = x[: 8 * 48000]
    if len(x) < 3 * 48000:
        return None
    n = len(x) // N.HOP
    x = x[: n * N.HOP]
    x = x * np.float32(LEVEL / max(float(np.sqrt((x ** 2).mean())), 1e-9))
    f0, _ = H.teacher_f0(x, n)
    f0 = f0[:n].astype(np.float64)
    v = f0 > 0
    if v.sum() < 50:
        return None
    xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
    per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()[:, :n]
    e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
    w = G["sm"]
    env = (w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:])[:, :n].astype(np.float64)
    with torch.no_grad():
        p = G["c1"](G["mf"](torch.from_numpy(T1.prime(x))[None]))[..., -n:].softmax(1)[0].numpy().astype(np.float64)
    p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
    er = env[1:25]
    res = er - C0.table(er, p.argmax(0), G["Cb"].shape[0], G["Cb"]) @ p
    rv = res[:, v] - res[:, v].mean(1, keepdims=True)
    fr = N.SR / N.HOP
    l2 = sosfiltfilt(butter(4, 2.0, fs=fr, output="sos"), res - res.mean(1, keepdims=True), axis=1)
    l8 = sosfiltfilt(butter(4, 8.0, fs=fr, output="sos"), res - res.mean(1, keepdims=True), axis=1)
    r0 = res - res.mean(1, keepdims=True)
    dyn = [np.sqrt((l2[:, v] ** 2).mean()), np.sqrt(((l8 - l2)[:, v] ** 2).mean()), np.sqrt(((r0 - l8)[:, v] ** 2).mean())]
    ev = np.sort(np.linalg.eigvalsh(rv @ rv.T / rv.shape[1]))[::-1]
    evn = (ev / ev.sum())[:5]
    import scipy.signal as ss
    fq, _, Z = ss.stft(x, 48000, nperseg=4096, noverlap=4096 - 240)
    L = np.log(np.abs(Z) + 1e-6)
    T = min(L.shape[1], n)
    ctr = []
    for lo, hi in [(100, 500), (500, 1500), (1500, 3000), (3000, 6000), (6000, 12000)]:
        b = (fq >= lo) & (fq < hi)
        Lb = L[b][:, :T][:, v[:T]]
        ctr.append(float(np.mean(np.percentile(Lb, 90, axis=0) - np.percentile(Lb, 10, axis=0))))
    lm = logmel(x)
    ms = modspec(lm)
    msg = [ms[a:b].mean(0) for a, b in [(0, 16), (16, 48), (48, 80), (80, 128)]]
    ltas = np.array([lm[a:a + 16].mean() for a in range(0, 128, 16)])
    ltas = ltas - ltas.mean()
    lf = np.log2(np.where(v, f0, 1.0))
    dv = np.abs(np.diff(lf))[(v[1:] & v[:-1])]
    jumps = float((dv > 0.6).sum() / (n / fr))
    fvec = np.concatenate([per[:, v].mean(1), per[:, v].std(1), ctr, np.concatenate(msg), ltas, dyn, evn,
                           [v.mean(), float(np.median(dv)) if len(dv) else 0.0, jumps]])
    return fvec.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--n_fit", type=int, default=400)
    ap.add_argument("--n_cal", type=int, default=200)
    a = ap.parse_args()
    S = Path(a.s)
    excl = set((S / "excl.txt").read_text().split()) if (S / "excl.txt").exists() else set()
    spks = sorted(d.name for d in (ROOT / "female-dataset").iterdir() if d.is_dir() and d.name not in excl)
    random.Random(93).shuffle(spks)
    real = []
    for s in spks:
        ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
        if ws:
            real.append(str(ws[len(ws) // 2]))
        if len(real) >= a.n_fit + a.n_cal:
            break
    lab = []
    for d, conds, ok in [("d61", ["ORA"], True), ("d66", ["DYN"], True), ("d68", ["DYN"], True), ("d62", ["TEACHER", "F1", "F1FILT"], True),
                         ("d61", ["TABENV"], False), ("d66", ["TAB", "TABPF", "INS"], False), ("d68", ["TAB"], False), ("d73", ["BTAB", "BW07", "W10"], False),
                         ("d85_360k", ["C05", "C07"], False)]:
        for c in conds:
            for f in sorted((S / d).glob(f"*_{c}.wav"))[:30]:
                lab.append((d, c, ok, str(f)))
    with Pool(a.workers, initializer=init) as pool:
        fr = pool.map(feat, real)
        fl = pool.map(feat, [x[3] for x in lab])
    R = np.array([f for f in fr if f is not None])
    Rf, Rc = R[: a.n_fit], R[a.n_fit:]
    mu = Rf.mean(0)
    sd = Rf.std(0) + 1e-9
    Z = (Rf - mu) / sd
    C = np.cov(Z.T)
    C = 0.8 * C + 0.2 * np.eye(C.shape[0])
    Ci = np.linalg.inv(C)
    d2 = lambda f: float(((f - mu) / sd) @ Ci @ ((f - mu) / sd))
    cal = np.array([d2(f) for f in Rc])
    thr = float(np.percentile(cal, 95))
    rows = []
    for (d, c, ok, p), f in zip(lab, fl):
        if f is None:
            continue
        q = d2(f)
        rows.append({"set": d, "cond": c, "ear_pass": ok, "d2": round(q, 2), "pass": q <= thr})
    summ = {"n_fit": len(Rf), "n_cal": len(Rc), "thr95": round(thr, 2), "real_cal_median": round(float(np.median(cal)), 2), "dim": int(R.shape[1])}
    by: dict = {}
    for r in rows:
        k = f"{r['set']}:{r['cond']}"
        by.setdefault(k, []).append(r)
    summ["groups"] = {k: {"ear_pass": v[0]["ear_pass"], "frac_pass": round(float(np.mean([r["pass"] for r in v])), 3), "d2_median": round(float(np.median([r["d2"] for r in v])), 2), "n": len(v)} for k, v in by.items()}
    tp = np.mean([r["pass"] for r in rows if r["ear_pass"]])
    tn = np.mean([not r["pass"] for r in rows if not r["ear_pass"]])
    summ["agree_ear_pass"] = round(float(tp), 3)
    summ["agree_ear_fail"] = round(float(tn), 3)
    print(json.dumps(summ, indent=1, ensure_ascii=False), flush=True)
    np.savez(ROOT / "results/conv_p0/d93_model.npz", mu=mu, sd=sd, Ci=Ci, thr=thr)
    (ROOT / "results/conv_p0/d93.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

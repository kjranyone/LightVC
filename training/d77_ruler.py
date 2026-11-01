"""D77(学習なし): 解析の耳の候補を、既にある耳の判定(写し合成・同じ発話の本物 SRC が基準)で検証する。
刺激の組(<s>/<dir>/pNNNN_tNNN_<cond>.wav または tNNN_<cond>.wav + SRC)ごとに、出力と SRC の距離を候補の物差しで測る:
  pesq(16k wb・−PESQ)・logmel(128 mel の対数・発話平均を引いた L1)・lsd(STFT 2048 の対数パワー 0〜8kHz・有声)・env(出力を SRC の教師 f0 で CheapTrick → DCT24 c1..24 の RMSE)・
  mod(128 mel の対数の時間軌道の変調スペクトル 0〜2/2〜8/8〜16/16〜32/32〜100Hz の対数エネルギーの L1・時間合わせ不要)・ctr(倍音間コントラストの差の絶対値)。
耳の制約(current/artic_vc.md の耳の記録):
  PASS = d61 ORA・d66 DYN・d68 DYN・d70 SELF・d62 TEACHER/F1/F1FILT ・ FAIL = d61 TABENV・d66 TAB/TABPF/INS・d68 TAB・d70 TAB・d66rv BASE/FT
  順位 = d68 DYN < SLOW < TAB < FAST・d70 SELF < XFER < TAB・d58 ORA < F1PY < RUSTCOPY(小さいほど良い)。
出力 = results/conv_p0/d77.json(刺激ごとの値 + 物差しごとの制約の充足)。
    uv run python d77_ruler.py --s <scratchpad> --workers 10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
SETS = {"d61": ["ORA", "TABENV"], "d66": ["DYN", "TAB", "TABPF", "INS"], "d68": ["DYN", "SLOW", "TAB", "FAST"], "d70": ["SELF", "XFER", "TAB"],
        "d66rv": ["BASE", "FT"], "d62": ["TEACHER", "F1", "F1FILT"], "d58": ["ORA", "F1PY", "RUSTCOPY"]}
PASS = [("d61", "ORA"), ("d66", "DYN"), ("d68", "DYN"), ("d70", "SELF"), ("d62", "TEACHER"), ("d62", "F1"), ("d62", "F1FILT")]
FAIL = [("d61", "TABENV"), ("d66", "TAB"), ("d66", "TABPF"), ("d66", "INS"), ("d68", "TAB"), ("d70", "TAB"), ("d66rv", "BASE"), ("d66rv", "FT")]
ORDERS = {"d68": ["DYN", "SLOW", "TAB", "FAST"], "d70": ["SELF", "XFER", "TAB"], "d58": ["ORA", "F1PY", "RUSTCOPY"]}
MBANDS = [0, 2, 8, 16, 32, 100]
G: dict = {}


def rd(p: Path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == 48000
    return x.mean(1)


def prep_ref(src: Path) -> dict:
    import f0hi as H
    import nvoc as N
    import pae as PA
    x = rd(src)
    n = len(x) // N.HOP
    x = x[: n * N.HOP]
    f0, _ = H.teacher_f0(x, n)
    f0 = f0[:n].astype(np.float64)
    return {"x": x, "n": n, "f0": f0, "env": env_of(x, f0, n)}


def env_of(x: np.ndarray, f0: np.ndarray, n: int) -> np.ndarray:
    import nvoc as N
    import pae as PA
    x = np.pad(x, (0, max(0, n * N.HOP - len(x))))[: n * N.HOP]
    xa = np.concatenate([np.zeros(PA.A, np.float32), x.astype(np.float32), np.zeros(PA.A, np.float32)])
    return PA.envelope(xa, f0)[:, :n].astype(np.float64)


def logmel(x: np.ndarray) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=x, sr=48000, n_fft=2048, hop_length=240, n_mels=128, fmax=16000, power=2.0)
    return np.log(m + 1e-8)


def stft_db(x: np.ndarray) -> np.ndarray:
    import scipy.signal as ss
    f, _, Z = ss.stft(x, 48000, nperseg=2048, noverlap=2048 - 240)
    return np.log(np.abs(Z[f <= 8000]) ** 2 + 1e-10)


def modspec(L: np.ndarray) -> np.ndarray:
    Lc = L - L.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(Lc, axis=1)) ** 2
    fq = np.fft.rfftfreq(L.shape[1], 240 / 48000)
    return np.stack([np.log(P[:, (fq >= MBANDS[b]) & (fq < MBANDS[b + 1])].mean(1) + 1e-8) for b in range(len(MBANDS) - 1)], 1)


def one(job: tuple) -> dict:
    from pesq import pesq
    import librosa
    from eval_zsvc import contrast
    from quality_gate import align
    set_, key, cond, path, srcp = job
    ref = prep_ref(Path(srcp))
    y = rd(Path(path))
    x = ref["x"]
    _, y = align(x, y)
    y = y[: len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    g = np.sqrt((x ** 2).mean() / max((y ** 2).mean(), 1e-12))
    y = y * g
    out = {"set": set_, "key": key, "cond": cond}
    try:
        out["pesq"] = -float(pesq(16000, librosa.resample(x, orig_sr=48000, target_sr=16000), librosa.resample(y, orig_sr=48000, target_sr=16000), "wb"))
    except Exception:
        out["pesq"] = float("nan")
    Lx, Ly = logmel(x), logmel(y)
    T = min(Lx.shape[1], Ly.shape[1])
    Lx, Ly = Lx[:, :T], Ly[:, :T]
    out["logmel"] = float(np.abs((Ly - Ly.mean()) - (Lx - Lx.mean())).mean())
    Sx, Sy = stft_db(x), stft_db(y)
    T2 = min(Sx.shape[1], Sy.shape[1], ref["n"])
    v = ref["f0"][:T2] > 0
    out["lsd"] = float(np.sqrt((((Sy[:, :T2] - Sy[:, :T2].mean()) - (Sx[:, :T2] - Sx[:, :T2].mean()))[:, v] ** 2).mean()))
    ey = env_of(y, ref["f0"], ref["n"])
    vv = ref["f0"] > 0
    out["env"] = float(np.sqrt(((ey[1:25] - ref["env"][1:25])[:, vv] ** 2).mean()))
    out["mod"] = float(np.abs(modspec(Ly) - modspec(Lx)).mean())
    out["ctr"] = float(abs(contrast(y, ref["f0"]) - contrast(x, ref["f0"])))
    return out


def init() -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    S = Path(a.s)
    jobs = []
    for d, conds in SETS.items():
        for f in sorted((S / d).glob("*_SRC.wav")):
            key = f.name[: -len("_SRC.wav")]
            for c in conds:
                p = S / d / f"{key}_{c}.wav"
                if p.exists():
                    jobs.append((d, key, c, str(p), str(f)))
    print("jobs", len(jobs), flush=True)
    rows = []
    with Pool(a.workers, initializer=init) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            rows.append(r)
            if i % 50 == 0:
                print(i, len(jobs), flush=True)
    metrics = ["pesq", "logmel", "lsd", "env", "mod", "ctr"]
    by = {}
    for r in rows:
        by.setdefault((r["set"], r["key"]), {})[r["cond"]] = r
    rep = {"n": len(rows), "metrics": {}}
    for m in metrics:
        pv = [r[m] for r in rows if (r["set"], r["cond"]) in PASS and np.isfinite(r[m])]
        fv = [r[m] for r in rows if (r["set"], r["cond"]) in FAIL and np.isfinite(r[m])]
        auc = float(np.mean([[1.0 if p < f else 0.5 if p == f else 0.0 for f in fv] for p in pv]))
        paired = []
        for (s_, k_), cs in by.items():
            for pc in [c for (ss_, c) in PASS if ss_ == s_]:
                for fc in [c for (ss_, c) in FAIL if ss_ == s_]:
                    if pc in cs and fc in cs:
                        paired.append(1.0 if cs[pc][m] < cs[fc][m] else 0.0)
        orders = {}
        for s_, od in ORDERS.items():
            taus = []
            for (ss_, k_), cs in by.items():
                if ss_ != s_ or not all(c in cs for c in od):
                    continue
                vals = [cs[c][m] for c in od]
                conc = [1.0 if vals[i] < vals[j] else 0.0 for i in range(len(od)) for j in range(i + 1, len(od))]
                taus.append(float(np.mean(conc)))
            means = {c: round(float(np.nanmean([cs[c][m] for (ss_, k_), cs in by.items() if ss_ == s_ and c in cs])), 4) for c in od}
            orders[s_] = {"pair_concordance": round(float(np.mean(taus)), 3), "means": means, "mean_order_ok": bool(all(means[od[i]] < means[od[i + 1]] for i in range(len(od) - 1)))}
        rep["metrics"][m] = {"auc_pass_lt_fail": round(auc, 3), "paired_pass_better": round(float(np.mean(paired)), 3), "pass_max": round(float(np.max(pv)), 4),
                             "fail_min": round(float(np.min(fv)), 4), "orders": orders}
        print(m, json.dumps(rep["metrics"][m], ensure_ascii=False), flush=True)
    rep["rows"] = rows
    (ROOT / "results/conv_p0/d77.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

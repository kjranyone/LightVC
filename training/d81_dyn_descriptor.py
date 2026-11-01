"""D81(学習なし): 本人の動きは「内容(単位・遷移)× 経過時間」で決まり、別の発話から予測できるか(内容で索引した話者の動きの記述子の成否)。
女声 TTS(neutral 10 文 + 4 つの話し方がある話者 --n 人): 登録 = 文 tK 以外(--enroll_styles の話し方)・試験 = tK_neutral。
包絡 env = CheapTrick → DCT24 → [w,1−2w,w] 平滑(製品の登録と同じ)・単位 = C1(prime)・表 T[u] = 登録の硬い単位の平均・残差 r = env − Σp T。
記述子(登録の r から): PH[φ] = 単位に入ってからの経過フレーム φ の段(0・1・2〜3・4〜7・8〜15・16+)ごとの平均・UPH[u, φ]・TPH[(u_前, u), φ](観測が --min 未満は 1 段粗い記述子へ戻す)。
試験: r̂(t) = 記述子[単位の列から因果に決まる (u_前, u, φ)]・指標 = 有声の r と r̂ の相関・env の RMSE(表だけ Σp T vs Σp T + r̂)。
基準: 同じ人の同じ文の別の話し方の r を DTW で合わせた相関(床)。出力 = results/conv_p0/d81.json。
    uv run python d81_dyn_descriptor.py --n 24
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
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/female_tts_corpus"
STYLES = ["neutral", "breathy", "low_tension", "soft", "warm"]
PHB = [0, 1, 2, 4, 8, 16, 10 ** 6]
G: dict = {}


def init() -> None:
    import torch
    torch.set_num_threads(1)
    import c1_content as CC
    import train_rvoc as TR
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])); c1.load_state_dict(st1["net"]); c1.eval()
    G["c1"], G["mf"] = c1, CC.MelFront()
    G["Cb"] = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    G["sm"] = TR.ckpt_env_smooth(sr_, None)


def analyse(x: np.ndarray) -> dict:
    import torch
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    n = len(x) // N.HOP
    x = x[: n * N.HOP].astype(np.float32)
    f0, _ = H.teacher_f0(x, n)
    f0 = f0[:n].astype(np.float64)
    xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
    e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
    w = G["sm"]
    env = (w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:])[:, :n].astype(np.float64)
    with torch.no_grad():
        p = G["c1"](G["mf"](torch.from_numpy(T1.prime(x))[None]))[..., -n:].softmax(1)[0].numpy().astype(np.float64)
    p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
    return {"env": env[1:25], "p": p, "u": p.argmax(0), "v": f0 > 0}


def ctx(u: np.ndarray) -> tuple:
    prev = np.zeros(len(u), int)
    ph = np.zeros(len(u), int)
    cur_prev, start = -1, 0
    for t in range(len(u)):
        if t > 0 and u[t] != u[t - 1]:
            cur_prev, start = u[t - 1], t
        prev[t] = cur_prev
        ph[t] = np.searchsorted(PHB, t - start, side="right") - 1
    return prev, ph


def one(job: tuple) -> dict:
    import conv_c0 as C0
    import librosa
    spk, k, styles, mn = job
    K = G["Cb"].shape[0]
    en = [analyse(load48(CORPUS / spk / f"t{i:02d}_{s}.wav")) for i in range(10) if i != k for s in styles]
    te = analyse(load48(CORPUS / spk / f"t{k:02d}_neutral.wav"))
    E = np.concatenate([a["env"] for a in en], 1)
    U = np.concatenate([a["u"] for a in en])
    T = C0.table(E, U, K, G["Cb"])
    acc = {"PH": {}, "UPH": {}, "TPH": {}}
    for a in en:
        r = a["env"] - T @ a["p"]
        prev, ph = ctx(a["u"])
        for t in np.nonzero(a["v"])[0]:
            for key, d in ((ph[t], acc["PH"]), ((a["u"][t], ph[t]), acc["UPH"]), ((prev[t], a["u"][t], ph[t]), acc["TPH"])):
                s = d.setdefault(key, [np.zeros(24), 0])
                s[0] += r[:, t]; s[1] += 1
    mean = {lv: {kk: v[0] / v[1] for kk, v in d.items() if v[1] >= (1 if lv == "PH" else mn)} for lv, d in acc.items()}
    r = te["env"] - T @ te["p"]
    prev, ph = ctx(te["u"])
    n = len(te["u"])
    preds = {lv: np.zeros((24, n)) for lv in ["PH", "UPH", "TPH"]}
    for t in range(n):
        a0 = mean["PH"].get(ph[t], np.zeros(24))
        a1 = mean["UPH"].get((te["u"][t], ph[t]), a0)
        a2 = mean["TPH"].get((prev[t], te["u"][t], ph[t]), a1)
        preds["PH"][:, t], preds["UPH"][:, t], preds["TPH"][:, t] = a0, a1, a2
    v = te["v"]
    out = {"spk": spk, "k": k, "rmse_tab": float(np.sqrt((r[:, v] ** 2).mean()))}
    for lv, pr in preds.items():
        out[f"rmse_{lv}"] = float(np.sqrt(((r - pr)[:, v] ** 2).mean()))
        out[f"corr_{lv}"] = float(np.mean([np.corrcoef(r[d, v], pr[d, v])[0, 1] for d in range(24) if pr[d, v].std() > 1e-9]))
        sh = np.zeros_like(pr)
        sh[:, 1:] = pr[:, :-1]
    fl = []
    for s in ["breathy", "low_tension", "soft", "warm"]:
        b = analyse(load48(CORPUS / spk / f"t{k:02d}_{s}.wav"))
        rb = b["env"] - T @ b["p"]
        _, wp = librosa.sequence.dtw(X=te["env"], Y=b["env"], metric="euclidean")
        i, j = wp[::-1, 0], wp[::-1, 1]
        rbt = np.zeros_like(r); cnt = np.zeros(n)
        for a_, b_ in zip(i, j):
            rbt[:, a_] += rb[:, b_]; cnt[a_] += 1
        rbt /= np.maximum(cnt, 1)
        fl.append(float(np.mean([np.corrcoef(r[d, v], rbt[d, v])[0, 1] for d in range(24)])))
    out["corr_floor_other_style"] = float(np.mean(fl))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--texts", default="0,3,6")
    ap.add_argument("--enroll_styles", default="neutral")
    ap.add_argument("--min", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tag", default="d81")
    a = ap.parse_args()
    spks = sorted(d.name for d in CORPUS.iterdir() if d.is_dir() and all((d / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in STYLES))
    random.Random(81).shuffle(spks)
    jobs = [(s, int(k), a.enroll_styles.split(","), a.min) for s in spks[: a.n] for k in a.texts.split(",")]
    rows = []
    with Pool(a.workers, initializer=init) as pool:
        for r in pool.imap_unordered(one, jobs):
            rows.append(r)
    keys = [k for k in rows[0] if k not in ("spk", "k")]
    summ = {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in keys}
    print(json.dumps(summ), flush=True)
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps({"args": vars(a), "summary": summ, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

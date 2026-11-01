"""D78(学習なし): 話者基準の自然さの物差し(同じ発話の本物を使わない = 変換にも使える)を、既にある耳の判定で検証する。
各刺激 y と、その目標話者の別の発話(評価の 30 目標 = <s>/d37/_work/ref_TTT.wav・TTS の組 = 話者 A の文 tK 以外の neutral)から作った基準 R の距離:
  mod = 128 mel の対数の時間軌道の変調スペクトル(0〜2/2〜8/8〜16/16〜32/32〜100Hz)の対数エネルギーの L1・ctr = 倍音間コントラストの差の絶対値・
  cov = 出力を解析し直した動き res(CheapTrick DCT24 − 同じ発話の単位平均の事後混合)の 24 次元の共分散の Förstner 距離・rms = res の帯(0〜2/2〜8/8Hz〜)ごとの RMS の対数比の絶対値の和。
基準は 8s ずつに区切って平均(mod・ctr・rms)/プール(cov)。解析の前に RMS を LEVEL(0.05)に揃える(2026-10-11 修正: 揃えないと小さい音で C1 の対数メルの床・f0 の有声判定が変わり、生の音量の Rust 出力(RMS 0.026)とピーク 0.9 に揃えた Python 出力(0.14)で同じ条件の共分散が 9.5 対 7.7 に食い違った)。同じ発話の本物 SRC の距離を並べて印字(自然の水準)。
耳の制約は D77 と同じ + 変換 d73(BTAB / BW07 / W10 = 全部失格)。出力 = results/conv_p0/d78.json。
    uv run python d78_spk_ruler.py --s <scratchpad> --workers 10
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from d77_ruler import FAIL, ORDERS, PASS, SETS, logmel, modspec, rd

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/female_tts_corpus"
SETS2 = {**SETS, "d73": ["SRC", "BTAB", "BW07", "W10"]}
FAIL2 = FAIL + [("d73", "BTAB"), ("d73", "BW07"), ("d73", "W10")]
G: dict = {}
LEVEL = 0.05


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


def feats(x: np.ndarray) -> dict:
    import torch
    from scipy.signal import butter, sosfiltfilt
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    from eval_zsvc import contrast
    n = len(x) // N.HOP
    x = x[: n * N.HOP].astype(np.float32)
    x = x * np.float32(LEVEL / max(float(np.sqrt((x ** 2).mean())), 1e-9))
    f0, _ = H.teacher_f0(x, n)
    f0 = f0[:n].astype(np.float64)
    xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
    e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
    w = G["sm"]
    env = (w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:])[:, :n].astype(np.float64)
    with torch.no_grad():
        p = G["c1"](G["mf"](torch.from_numpy(T1.prime(x))[None]))[..., -n:].softmax(1)[0].numpy().astype(np.float64)
    p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
    er = env[1:25]
    res = er - C0.table(er, p.argmax(0), G["Cb"].shape[0], G["Cb"]) @ p
    v = f0 > 0
    rv = res - res.mean(1, keepdims=True)
    fr = N.SR / N.HOP
    l2 = sosfiltfilt(butter(4, 2.0, fs=fr, output="sos"), rv, axis=1)
    l8 = sosfiltfilt(butter(4, 8.0, fs=fr, output="sos"), rv, axis=1)
    rms = np.array([np.sqrt((l2[:, v] ** 2).mean()), np.sqrt(((l8 - l2)[:, v] ** 2).mean()), np.sqrt(((rv - l8)[:, v] ** 2).mean())])
    R = rv[:, v]
    return {"mod": modspec(logmel(x)), "ctr": contrast(x, f0), "cov_sum": R @ R.T, "cov_n": R.shape[1], "rms": rms}


def ref_feats(x: np.ndarray) -> dict:
    seg = 8 * 48000
    fs = [feats(x[i: i + seg]) for i in range(0, max(1, len(x) - seg // 2), seg) if len(x[i: i + seg]) > 3 * 48000]
    return {"mod": np.mean([f["mod"] for f in fs], 0), "ctr": float(np.nanmean([f["ctr"] for f in fs])),
            "cov": sum(f["cov_sum"] for f in fs) / sum(f["cov_n"] for f in fs) + 1e-6 * np.eye(24), "rms": np.mean([f["rms"] for f in fs], 0)}


def dist(f: dict, r: dict) -> dict:
    from d74_res_cov import forstner
    C = f["cov_sum"] / max(f["cov_n"], 1) + 1e-6 * np.eye(24)
    return {"mod": float(np.abs(f["mod"] - r["mod"]).mean()), "ctr": float(abs(f["ctr"] - r["ctr"])), "cov": forstner(r["cov"], C),
            "rms": float(np.abs(np.log(f["rms"] / r["rms"])).sum())}


def ref_audio(S: Path, set_: str, key: str) -> np.ndarray:
    import re
    from d6_unit_cov import load48
    if set_ == "d70":
        info = {f"p{d['k']:04d}_t{d['k']:03d}": d for d in json.loads((S / "d70/d70.json").read_text())}[key]
        return np.concatenate([load48(CORPUS / info["A"] / f"t{i:02d}_neutral.wav") for i in range(10) if f"t{i:02d}" != info["text"]]).astype(np.float32)
    ti = int(re.search(r"t(\d+)", key).group(1))
    return rd(S / "d37/_work" / f"ref_{ti:03d}.wav")


def one(job: tuple) -> list:
    S, set_, key, items = job
    r = ref_feats(ref_audio(Path(S), set_, key))
    out = []
    for cond, path in items:
        d = dist(feats(rd(Path(path))), r)
        out.append({"set": set_, "key": key, "cond": cond, **d})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    S = Path(a.s)
    jobs = []
    for d, conds in SETS2.items():
        for f in sorted((S / d).glob("*_SRC.wav")):
            key = f.name[: -len("_SRC.wav")]
            items = [(c, str(S / d / f"{key}_{c}.wav")) for c in (conds if "SRC" in conds else ["SRC"] + conds) if (S / d / f"{key}_{c}.wav").exists()]
            jobs.append((str(S), d, key, items))
    print("jobs", len(jobs), flush=True)
    rows = []
    with Pool(a.workers, initializer=init) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            rows.extend(r)
            if i % 20 == 0:
                print(i, len(jobs), flush=True)
    rep = {"metrics": {}}
    by = {}
    for r in rows:
        by.setdefault((r["set"], r["key"]), {})[r["cond"]] = r
    for m in ["mod", "ctr", "cov", "rms"]:
        pv = [r[m] for r in rows if (r["set"], r["cond"]) in PASS and np.isfinite(r[m])]
        fv = [r[m] for r in rows if (r["set"], r["cond"]) in FAIL2 and np.isfinite(r[m])]
        sv = [r[m] for r in rows if r["cond"] == "SRC" and np.isfinite(r[m])]
        auc = float(np.mean([[1.0 if p < f else 0.5 if p == f else 0.0 for f in fv] for p in pv]))
        orders = {}
        for s_, od in ORDERS.items():
            means = {c: round(float(np.nanmean([cs[c][m] for (ss_, k_), cs in by.items() if ss_ == s_ and c in cs])), 4) for c in od}
            orders[s_] = {"means": means, "mean_order_ok": bool(all(means[od[i]] < means[od[i + 1]] for i in range(len(od) - 1)))}
        conv = {c: round(float(np.nanmean([cs[c][m] for (ss_, k_), cs in by.items() if ss_ == "d73" and c in cs])), 4) for c in ["SRC", "BTAB", "BW07", "W10"]}
        rep["metrics"][m] = {"auc_pass_lt_fail": round(auc, 3), "src_mean": round(float(np.mean(sv)), 4), "pass_mean": round(float(np.mean(pv)), 4),
                             "pass_p90": round(float(np.percentile(pv, 90)), 4), "fail_mean": round(float(np.mean(fv)), 4), "fail_p10": round(float(np.percentile(fv, 10)), 4),
                             "fail_frac_above_pass_p90": round(float(np.mean(np.array(fv) > np.percentile(pv, 90))), 3), "orders": orders, "conv_d73": conv}
        print(m, json.dumps(rep["metrics"][m], ensure_ascii=False), flush=True)
    rep["rows"] = rows
    (ROOT / "results/conv_p0/d78.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

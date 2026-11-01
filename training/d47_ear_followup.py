"""D47(診断・学習なし): 耳 warp_ab2(オーナー 2026-10-08『f0 を見失っている・全体的にガビガビ・いずれも聞き取りづらい』)の数値の裏取り。D37 の男声 TTS 60 文。
(1) f0: 製品の F1(因果)と教師 f0(f0hi)の比較 = 有声の一致・有声 ↔ 無声の切り替わりの回数(毎秒)・オクターブ誤り(|log2 比| > 0.5)・半音より大きいずれ。
    出力の f0: 変換の出力(TAB・G7・M5)を教師 f0 で測り、狙い(F1 の f0 × 目標の中央値への比)との一致を同じ指標で。
(2) ガビガビ: フレーム周期(200/400/600Hz)の包絡の変調の線(eval_nvoc.amline・本物 ≈ 0)を 元 → 本物の分析の写し合成(D44 ORA)→ 製品の写し合成(D44 PROD・D43 COPYM)→ 変換(D40 TAB・G7・M5)で。
    uv run python d47_ear_followup.py --s <scratchpad> --out ../results/conv_p0/d47_ear_followup.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def f0stats(ref: np.ndarray, est: np.ndarray) -> dict:
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    vr, ve = ref > 0, est > 0
    both = vr & ve
    lr = np.log2(np.where(both, est, 1.0) / np.where(both, ref, 1.0))[both]
    flick = lambda v: float(np.abs(np.diff(v.astype(int))).sum() / (n / 200.0))
    return {"v_agree": float((vr == ve).mean()), "v_miss": float((vr & ~ve).sum() / max(vr.sum(), 1)), "v_false": float((~vr & ve).sum() / max((~vr).sum(), 1)),
            "flick_ref_per_s": flick(vr), "flick_est_per_s": flick(ve), "octave_err": float((np.abs(lr) > 0.5).mean()) if len(lr) else 0.0,
            "semitone_err": float((np.abs(lr) > 1 / 12).mean()) if len(lr) else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import eval_nvoc as E
    import f0est as FE
    import f0hi as H
    import nvoc as N
    S = Path(a.s)
    dev = "cuda"
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    srcs = sorted((S / "d37").glob("p*_SRC.wav"))
    rep: dict = {"f0_src": [], "f0_out": {}, "amline": {}}
    keys = [f.name[: -len("_SRC.wav")] for f in srcs]
    tgt_mu = {}
    for ti in range(30):
        j = json.loads((S / "d37/_work" / f"enroll_{ti:03d}" / "enroll.json").read_text())
        tgt_mu[ti] = j["mu"]
    for f, k in zip(srcs, keys):
        x, _ = sf.read(f, dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0t, _ = H.teacher_f0(x, n)
        with torch.no_grad():
            v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x)[None].to(dev)], 1), n))
            f0f = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
        st = f0stats(f0t[:n], f0f)
        rep["f0_src"].append(st)
        vo = f0f > 0
        med = float(np.median(np.log(f0f[vo]))) if vo.any() else 0.0
        ti = int(k[7:10])
        aim = np.where(vo, np.exp(np.log(np.maximum(f0f, 1.0)) - med + tgt_mu[ti]), 0.0)
        for c, d in (("TAB", "d40"), ("G7", "d40"), ("M5", "d40")):
            y, _ = sf.read(S / d / f"{k}_{c}.wav", dtype="float32")
            m = len(y) // N.HOP
            fo, _ = H.teacher_f0(y[: m * N.HOP], m)
            rep["f0_out"].setdefault(c, []).append(f0stats(aim, fo))
        print(k, {kk: round(v, 3) for kk, v in st.items()}, flush=True)
    for c, path in (("SRC", "d37/{k}_SRC.wav"), ("ORA_py", "d44/{k}_ORA.wav"), ("PROD_py", "d44/{k}_PROD.wav"), ("COPYM_rust", "d43/{k}_COPYM.wav"),
                    ("TAB_rust", "d40/{k}_TAB.wav"), ("G7_rust", "d40/{k}_G7.wav"), ("M5_rust", "d40/{k}_M5.wav")):
        rep["amline"][c] = [float(E.amline(sf.read(S / path.format(k=k), dtype="float64")[0])) for k in keys]
    summ = {"f0_src": {kk: round(float(np.mean([s[kk] for s in rep["f0_src"]])), 4) for kk in rep["f0_src"][0]},
            "f0_out": {c: {kk: round(float(np.mean([s[kk] for s in v])), 4) for kk in v[0]} for c, v in rep["f0_out"].items()},
            "amline": {c: {"median": round(float(np.median(v)), 3), "mean": round(float(np.mean(v)), 3), "p90": round(float(np.percentile(v, 90)), 3)} for c, v in rep["amline"].items()}}
    rep["summary"] = summ
    print(json.dumps(summ, indent=1, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

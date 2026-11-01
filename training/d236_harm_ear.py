"""D236(学習なし・results/conv_p0/d236_prereg.yaml): 変換の耳の不合格は有声の倍音の明瞭さの不足で説明できるか。
物差し = 帯域ごとの倍音の山谷比 hnr1/2/3(pae.periodicity・出力の f0 = F1 + 確信度 0.5)・ケプストラムのピークの突出 cpp(16 kHz)・無音の床 floor(p5 − p95)。
ラベル = 耳のクリップ(rvoc_ab3 の合格の写し合成・cand_ab1 / prodcand_ab1 / convg1_ab1 / conv_ab1 の不合格の変換・R / source の実音声)+ dev 3 組の変換 180 本ずつ。閾値 = 本物の女声 150 人の [p15, p85]。
    uv run python d236_harm_ear.py --s <dev の親> --out ../results/conv_p0/d236.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"
SEC = 8.0
GOOD = {"hnr1": "hi", "hnr2": "hi", "hnr3": "hi", "cpp": "hi", "floor": "lo"}
SETS = ("dev", "dev2", "dev3")
DEVF = {"KNN": ("d135_real", "KNN"), "G03S10ZH": ("d114", "G03S10ZH"), "RRQSDC19LE": ("d139", "RRQSDC19LE")}
EARDIRS = {"cand_ab1": {"cand": "FAIL_cand_ab1", "base": "FAIL_cand_ab1", "knn": "KNN_ear"},
           "prodcand_ab1": {"cand_gamma05_color_ft360k": "FAIL_prodcand", "cand_gamma07_color_ft360k": "FAIL_prodcand"},
           "convg1_ab1": {"table_only": "FAIL_convg1", "warp_gamma07": "FAIL_convg1", "warp_gamma10": "FAIL_convg1"},
           "conv_ab1": {"same": "FAIL_conv_ab1", "other": "FAIL_conv_ab1"},
           "rvoc_ab3": {"rvoc": "PASS_rvoc", "bigvgan_v2": "PASS_bigvgan", "source": "EAR_REAL"}}
CAND_TRIALS = ("cand_01", "cand_02", "cand_03")
FAIL_GROUPS = ("FAIL_cand_ab1", "FAIL_prodcand", "FAIL_convg1", "FAIL_conv_ab1", "dev:G03S10ZH", "dev:RRQSDC19LE")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_pool", type=int, default=150)
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    import f0est as FE
    import nvoc as N
    import pae as PA
    from d6_unit_cov import load48
    S = Path(a.s)
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())

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

    def cpp16(x48: np.ndarray, act: np.ndarray) -> float:
        x = resample_poly(x48, 1, 3)
        L, H, NF = 640, 160, 1024
        fr = np.lib.stride_tricks.sliding_window_view(x, L)[::H] * np.hanning(L)
        sp = 10 * np.log10(np.abs(np.fft.rfft(fr, NF)) ** 2 + 1e-12)
        ce = 20 * np.log10(np.abs(np.fft.irfft(sp, NF))[:, : NF // 2] + 1e-12)
        q = np.arange(NF // 2) / 16000
        srch = (q >= 1 / 1000) & (q <= 1 / 60)
        reg = (q >= 0.001) & (q <= 0.025)
        A_ = np.vstack([q[reg], np.ones(reg.sum())]).T
        coef = np.linalg.lstsq(A_, ce[:, reg].T, rcond=None)[0]
        ip = np.flatnonzero(srch)[ce[:, srch].argmax(1)]
        c = ce[np.arange(len(ce)), ip] - (coef[0] * q[ip] + coef[1])
        ta = np.arange(len(c)) * H * 3 // N.HOP
        ok = act[np.clip(ta, 0, len(act) - 1)]
        return float(np.median(c[ok])) if ok.sum() > 5 else float("nan")

    def stats(x: np.ndarray) -> dict:
        x = x[: int(SEC * N.SR)].astype(np.float32)
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        f = f0track(x)
        n = len(f)
        e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
        act = e > e.max() - 40
        xa = np.pad(x[: n * N.HOP], (PA.A, PA.A))
        per = PA.periodicity(torch.from_numpy(xa)[None].to(dev), torch.from_numpy(f.astype(np.float32))[None].to(dev))[0].cpu().numpy() * 30
        v = (f > 0) & act
        r = {f"hnr{i + 1}": float(np.median(per[i, v])) if v.sum() > 10 else float("nan") for i in range(3)}
        r["cpp"] = cpp16(x, act)
        r["floor"] = float(np.percentile(e, 5) - np.percentile(e, 95))
        r["vfrac"] = float(v.sum() / max(act.sum(), 1))
        return r

    excl = set((S / "excl.txt").read_text().split())
    for f in list(S.glob("*/devset.json")) + [S / "eval_targets.json"]:
        if f.exists():
            excl |= set(json.loads(f.read_text())["targets"])
    base = ROOT / "female-dataset"
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(109).shuffle(pool)
    rows: list = []
    nreal = 0
    for spk in pool:
        if nreal >= a.n_pool:
            break
        ws = sorted((base / spk).glob("*.wav"))
        if len(ws) < 3 or any(sf.info(str(w)).samplerate < 44100 for w in ws[:3]):
            continue
        x = load48(ws[len(ws) // 2])
        if len(x) < 3 * N.SR:
            continue
        rows.append({"grp": "REAL_POOL", "id": spk} | stats(x))
        nreal += 1
    print("pool", nreal, flush=True)
    for d, mp in EARDIRS.items():
        key = json.loads((EB / d / "_key_聴取後に開く.json").read_text())["trials"]
        for t, v in key.items():
            if d == "cand_ab1" and t not in CAND_TRIALS:
                continue
            for L, form in v["map"].items():
                if form in mp:
                    rows.append({"grp": mp[form], "id": f"{d}/{t}/{L}", "form": form} | stats(load48(EB / d / t / f"{L}.wav")))
            if (EB / d / t / "R.wav").exists():
                rows.append({"grp": "EAR_REAL", "id": f"{d}/{t}/R"} | stats(load48(EB / d / t / "R.wav")))
    print("ear", len(rows) - nreal, flush=True)
    for D in SETS:
        for nm, (dd, c) in DEVF.items():
            for f in sorted((S / D / dd).glob(f"*_{c}.wav")):
                rows.append({"grp": f"dev:{nm}", "id": f"{D}/{f.name}"} | stats(load48(f)))
        print(D, len(rows), flush=True)
    rep: dict = {"rows": rows}
    real = {m: np.array([r[m] for r in rows if r["grp"] == "REAL_POOL"], float) for m in GOOD}
    rep["real_p15_p85"] = {m: [round(float(np.nanpercentile(v, 15)), 3), round(float(np.nanpercentile(v, 85)), 3)] for m, v in real.items()}
    grps = sorted({r["grp"] for r in rows})
    rep["medians"] = {g: {m: round(float(np.nanmedian([r[m] for r in rows if r["grp"] == g])), 3) for m in list(GOOD) + ["vfrac"]} | {"n": sum(r["grp"] == g for r in rows)} for g in grps}
    bad = lambda m, val: val < rep["real_p15_p85"][m][0] if GOOD[m] == "hi" else val > rep["real_p15_p85"][m][1]
    rep["axis"] = {}
    for m in GOOD:
        pass_ok = all(not bad(m, rep["medians"][g][m]) for g in ("PASS_rvoc", "PASS_bigvgan"))
        det = {g: bad(m, rep["medians"][g][m]) for g in FAIL_GROUPS}
        rep["axis"][m] = {"pass_ok": pass_ok, "fail_detected": det, "axis_ok": pass_ok and all(det.values())}
    rep["trial1"] = {r.get("form", "R"): {m: round(r[m], 2) for m in GOOD} for r in rows if r["id"].startswith("cand_ab1/cand_01/")}
    ok = [m for m in GOOD if rep["axis"][m]["axis_ok"]]
    rep["verdict"] = "HARM_EXPLAINS" if any(m != "floor" for m in ok) else "FLOOR_ONLY" if ok else "NONE"
    print(json.dumps({k: rep[k] for k in ("real_p15_p85", "medians", "axis", "trial1", "verdict")}, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

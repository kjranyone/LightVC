"""D9: 変換の包絡の不自然さ(単位のちらつき)の測定(学習なし)。
元の声(日本語男声 TTS)を C1 の硬い単位 → 目標の表(参照 20s)→ 因果 5 フレーム平均 ê にしたときの (1) 単位の切り替え頻度、
(2) 包絡の変調スペクトル(0–4・4–15・15–50Hz の帯域ごとの変動の rms・c1..c24 平均)を、自然な女声の実の包絡(目標自身の別発話の CheapTrick 包絡)と比べる。
処方の候補: 切り替えのヒステリシス(因果: 新しい単位の事後が今の単位を margin 上回る状態が h フレーム続いたら切り替え = h フレームの遅れ)。
    uv run python d9_flicker.py --out ../results/conv_p0/d9_flicker.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import TTS_LEAK, load48

ROOT = Path(__file__).resolve().parent.parent
FPS = 200


def band_rms(E: np.ndarray, lo: float, hi: float) -> float:
    if E.shape[1] < 64:
        return float("nan")
    if lo <= 0:
        sos = butter(4, hi, "lowpass", fs=FPS, output="sos")
    else:
        sos = butter(4, [lo, hi], "bandpass", fs=FPS, output="sos")
    y = sosfiltfilt(sos, E - E.mean(1, keepdims=True), axis=1)
    return float(np.sqrt((y ** 2).mean()))


def hysteresis(p: np.ndarray, h: int, margin: float) -> np.ndarray:
    """p [n, K] 事後 → 因果のヒステリシスつきの単位列。"""
    cur = int(p[0].argmax())
    cand, run = -1, 0
    out = np.empty(len(p), int)
    for t in range(len(p)):
        b = int(p[t].argmax())
        if b != cur and p[t, b] > p[t, cur] + margin:
            run = run + 1 if b == cand else 1
            cand = b
            if run >= h:
                cur, run = b, 0
        else:
            run = 0
        out[t] = cur
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_tg", type=int, default=12)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_c3 as C3
    import train_rvoc as TR
    dev = "cuda"
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    sm_w = 0.25

    def sm3(E):
        e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
        return sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:]

    @torch.no_grad()
    def post(x):
        n = len(x) // N.HOP
        return c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].T.cpu().numpy()

    def envelope(x):
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        return sm3(PA.envelope(xa, f0)), f0

    hs = TR.held_speakers()
    held = [s for s in sorted(hs) if s != "unknown" and s not in TTS_LEAK and (ROOT / "female-dataset" / s).is_dir()][: a.n_tg]
    mrows = [r for r in json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())["rows"] if r["ok"] and r["src"] == "tts_m"]
    by: dict = {}
    for r in mrows:
        by.setdefault(r["spk"], []).append(r["wav"])
    msp = sorted(by)
    rep: dict = {}
    acc: dict = {}

    def add(name, E, sw=None):
        d = acc.setdefault(name, {"0-4": [], "4-15": [], "15-50": [], "switch_per_s": []})
        d["0-4"].append(band_rms(E, 0, 4)); d["4-15"].append(band_rms(E, 4, 15)); d["15-50"].append(band_rms(E, 15, 50))
        if sw is not None:
            d["switch_per_s"].append(sw)

    for i, s in enumerate(held):
        ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        rest = [w for w in ws if w not in ref][:4]
        if not rest:
            continue
        xr = np.concatenate([load48(w) for w in ref])
        xr = xr[: len(xr) // N.HOP * N.HOP]
        Er, _ = envelope(xr)
        pr = post(xr)
        T = C0.table(Er[1:25, : len(pr)], pr.argmax(1), K, Cb)
        xn = np.concatenate([load48(w) for w in rest])[: 10 * 48000]
        xn = xn[: len(xn) // N.HOP * N.HOP]
        En, f0n = envelope(xn)
        vo = f0n > 0
        pn = post(xn)
        un = pn.argmax(1)
        add("female_real_env", En[1:25])
        add("female_self_TAB", C3.causal_avg(torch.from_numpy(T[:, un].astype(np.float32))[None], 5)[0].numpy(), float((np.diff(un) != 0).mean() * FPS))
        m = msp[i % len(msp)]
        xm = np.concatenate([load48(w) for w in sorted(by[m])[:3]])[: 10 * 48000]
        xm = xm[: len(xm) // N.HOP * N.HOP]
        pm = post(xm)
        um = pm.argmax(1)
        add("male_TAB", C3.causal_avg(torch.from_numpy(T[:, um].astype(np.float32))[None], 5)[0].numpy(), float((np.diff(um) != 0).mean() * FPS))
        Em, _ = envelope(xm)
        add("male_real_env", Em[1:25])
        for h, mg in ((2, 0.0), (3, 0.05), (4, 0.1)):
            uh = hysteresis(pm, h, mg)
            add(f"male_TAB_hyst_h{h}_m{mg}", C3.causal_avg(torch.from_numpy(T[:, uh].astype(np.float32))[None], 5)[0].numpy(), float((np.diff(uh) != 0).mean() * FPS))
        print(i, s, {k: round(float(np.nanmean(v["4-15"])), 3) for k, v in acc.items()}, flush=True)
    for k, v in acc.items():
        rep[k] = {b: round(float(np.nanmean(x)), 4) for b, x in v.items() if x}
    nat = rep["female_real_env"]
    for k in rep:
        rep[k]["ratio_to_natural"] = {b: round(rep[k][b] / nat[b], 3) for b in ("0-4", "4-15", "15-50")}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

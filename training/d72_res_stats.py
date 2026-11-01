"""D72(診断・学習なし): 製品で写す動き(F3 の包絡 − 操作者の表·p)が、女声の本物の動き(CheapTrick − 同じ発話の単位平均·p)と同じ大きさ・同じ変調周波数の配分を持つか。
  FEM_REAL = 女声 30 目標(D37 の cen)の本物の動き(D68 の定義)・FEM_F3 = 同じ発話の F3 の包絡 − 本人の Rust 登録の表·p(製品の式で操作者 = 本人)・
  FEM_F3_INS / MALE_F3_INS = F3 の包絡 − 同じ発話の F3 の単位平均·p(本物と同じ定義・表の誤差を含まない)・FEM_F3vsREAL = FEM_F3 と FEM_REAL の次元ごとの相関(F3 が本物の動きをどれだけ再現するか)・MALE_F3 = D56 の男声 60 文の F3 − 操作者の表·p(製品で写す動き)。
帯ごとの RMS(0〜2・2〜8・8 Hz〜)を印字。出力 = results/conv_p0/d72.json。
    uv run python d72_res_stats.py --s <scratchpad>
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

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_rvoc as TR
    S = Path(a.s)
    dev = "cuda"
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    sf3 = torch.load(ROOT / "results/f3_1/last.pt", map_location="cpu", weights_only=False)
    f3 = LV.EnvEst(sf3["cfg"]["ch"], tuple(sf3["cfg"]["dils"])).to(dev); f3.load_state_dict(sf3["net"]); f3.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    fr = N.SR / N.HOP
    s2, s8 = butter(4, 2.0, fs=fr, output="sos"), butter(4, 8.0, fs=fr, output="sos")
    rd = lambda d: np.fromfile(d / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64)

    def bands(r: np.ndarray) -> list:
        r = r - r.mean(1, keepdims=True)
        l2 = sosfiltfilt(s2, r, axis=1)
        l8 = sosfiltfilt(s8, r, axis=1)
        return [float(np.sqrt((l2 ** 2).mean())), float(np.sqrt(((l8 - l2) ** 2).mean())), float(np.sqrt(((r - l8) ** 2).mean())), float(np.sqrt((r ** 2).mean()))]

    def front(x: np.ndarray, n: int):
        xt = torch.from_numpy(x)[None].to(dev)
        with torch.no_grad():
            e3 = 10.0 * f3(mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1)))[0].cpu().numpy()[:, :n].astype(np.float64)
            p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        e3 = np.pad(e3, ((0, 0), (0, max(0, n - e3.shape[1]))), mode="edge")[:, :n]
        return e3, p

    acc: dict = {"FEM_REAL": [], "FEM_F3": [], "FEM_F3_INS": [], "MALE_F3": [], "MALE_F3_INS": []}
    corr, corr_ins = [], []
    for ti in range(30):
        x, _ = sf.read(S / "d37" / f"cen_{ti:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
        e3, p = front(x, n)
        er = env[1:25]
        rr = er - C0.table(er, p.argmax(0), K, Cb) @ p
        rf = e3[1:25] - rd(S / "d53_en" / f"en_{ti:03d}") @ p
        rfi = e3[1:25] - C0.table(e3[1:25], p.argmax(0), K, Cb) @ p
        acc["FEM_REAL"].append(bands(rr)); acc["FEM_F3"].append(bands(rf)); acc["FEM_F3_INS"].append(bands(rfi))
        corr.append(float(np.mean([np.corrcoef(rr[d] - rr[d].mean(), rf[d] - rf[d].mean())[0, 1] for d in range(24)])))
        corr_ins.append(float(np.mean([np.corrcoef(rr[d], rfi[d])[0, 1] for d in range(24)])))
    for f in sorted((S / "d56").glob("p*_SRC.wav")):
        pi = int(f.name[1:5])
        x, _ = sf.read(f, dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        e3, p = front(x, n)
        acc["MALE_F3"].append(bands(e3[1:25] - rd(S / "d56/_work" / f"op_{pi:04d}") @ p))
        acc["MALE_F3_INS"].append(bands(e3[1:25] - C0.table(e3[1:25], p.argmax(0), K, Cb) @ p))
    rep = {k: dict(zip(["rms_0_2Hz", "rms_2_8Hz", "rms_8Hz_up", "rms_all"], [round(float(v), 4) for v in np.mean(np.asarray(x), 0)])) for k, x in acc.items()}
    rep["FEM_F3_vs_REAL_corr_mean"] = round(float(np.mean(corr)), 4)
    rep["FEM_F3_INS_vs_REAL_corr_mean"] = round(float(np.mean(corr_ins)), 4)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    (ROOT / "results/conv_p0/d72.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

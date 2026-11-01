"""D112(学習なし): D109 v2 で製品の変換(d110 C07F5)に残る失格 = こもり軸の tilt(長時間スペクトルの傾きが本物の女声より暗い・表だけの TAB でも暗い)の出どころ。
発声中のフレームを有声 / 無声(F1 + 確信度 0.5 = D109 と同じ)に分け、それぞれの 1/3 oct の長時間スペクトルの傾き(300Hz–8kHz)と 4–10kHz / 0.1–4kHz 比、無声の割合を測る。
  有声の傾きが本物並みで無声の割合だけ少ない → 子音の作り方(無声を有声で鳴らす)・有声の傾きが暗い → 有声の包絡か励起。
組 = 目標の実音声 cen・写し合成 d102 ORA / ALLF4・元の男声 SRC・d110 TAB / C07 / C07F5。出力 = results/conv_p0/d112.json。
    uv run python d112_tilt_split.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="名前=ディレクトリ/パターン(<s> 相対)だけを測る")
    ap.add_argument("--tag", default="d112")
    a = ap.parse_args()
    import scipy.signal as ss
    import f0est as FE
    import nvoc as N
    from d6_unit_cov import load48
    S = Path(a.s)
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(s1["net"]); f1.eval()
    fr_ = FE.Front().to(dev)
    cf = 1000.0 * 2.0 ** (np.arange(-10, 11) / 3.0)
    cf = cf[(cf >= 300) & (cf <= 8000)]

    def voiced(x: np.ndarray) -> np.ndarray:
        n = len(x) // N.HOP
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * N.HOP])[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy()[:, :n]
        k = pr.argmax(0)
        ii = np.clip(k[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        c = np.take_along_axis(pr, ii, 0).sum(0)
        return vo & (c >= 0.5)

    def tilt_hf(P: np.ndarray, fq: np.ndarray) -> tuple:
        if P.shape[1] < 5:
            return float("nan"), float("nan")
        m = P.mean(1)
        L = 10 * np.log10(np.array([m[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() for c in cf]) + 1e-20)
        return float(np.polyfit(np.log2(cf), L, 1)[0]), float(10 * np.log10(m[(fq >= 4000) & (fq < 10000)].sum() / m[(fq >= 100) & (fq < 4000)].sum()))

    sets = {"cen": sorted((S / "d37").glob("cen_*.wav")), "ORA": sorted((S / "d102").glob("*_ORA.wav")), "ALLF4": sorted((S / "d102").glob("*_ALLF4.wav")),
            "SRC": sorted((S / "d110").glob("*_SRC.wav")), "TAB": sorted((S / "d110").glob("*_TAB.wav")), "C07": sorted((S / "d110").glob("*_C07.wav")),
            "C07F5": sorted((S / "d110").glob("*_C07F5.wav"))}
    if a.only is not None:
        sets = {e.split("=")[0]: sorted(S.glob(e.split("=")[1])) for e in a.only}
    rep = {}
    for nm, fs in sets.items():
        acc: dict = {}
        for p in fs:
            x = load48(p).astype(np.float32)[: 8 * N.SR]
            x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
            v = voiced(x)
            n = len(v)
            e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
            act = e > e.max() - 40
            fq, _, Z = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
            P = np.abs(Z[:, :n]) ** 2
            m = min(P.shape[1], n)
            P, v, act = P[:, :m], v[:m], act[:m]
            tv, hv = tilt_hf(P[:, act & v], fq)
            tu, hu = tilt_hf(P[:, act & ~v], fq)
            ta, ha = tilt_hf(P[:, act], fq)
            for k, val in {"tilt_all": ta, "hf_all": ha, "tilt_voiced": tv, "hf_voiced": hv, "tilt_unvoiced": tu, "hf_unvoiced": hu,
                           "unvoiced_frac": float((act & ~v).sum() / max(act.sum(), 1)),
                           "unvoiced_energy_share": float(P[:, act & ~v].sum() / max(P[:, act].sum(), 1e-20))}.items():
                acc.setdefault(k, []).append(val)
        rep[nm] = {k: round(float(np.nanmedian(v)), 3) for k, v in acc.items()}
        print(nm, json.dumps(rep[nm]), flush=True)
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

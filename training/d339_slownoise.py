"""D339: 生成の床の運び手 = 発話ごとの遅い音色の揺らぎの不足(D334 S1・D335 の 0.125〜1 Hz で 4〜12 kHz −1.2〜−1.4 dB・D337 予測できない・D338 表の誤差でない)。
学習し直さずに、本物の揺らぎと同じ統計(固定の定数)を持つ乱数の遅い揺らぎを ppgvoc18 の包絡の行に足し、SR の v1⊥ が下がるかを見る。
統計: e2e/fem の実女声(保留を除く・表のある話者)の残差(包絡 − Σp T / 10)の 0.5 s の移動平均からクリップの平均を引いた揺らぎ → 主成分 → 成分ごとの平均のパワースペクトル。
生成: 成分ごとに白色雑音にその振幅スペクトルをかけて戻す(乱数の位相・種は項目ごとに固定)→ 主成分の基底で 24 次元へ・σ 倍。診断(製品にするなら因果な固定のフィルタで作る)。
    uv run python d339_slownoise.py --ckpt ../results/diag_ppgvoc18/last_ema.pt --out ../results/conv_p0/d339
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P

SEG = 2048


BAND = {"w": (100, 0)}


def resid(c: np.ndarray, lg: np.ndarray, T: np.ndarray) -> np.ndarray:
    """残差の帯: BAND["w"] = (長い窓, 短い窓) のフレーム数。短い窓 0 = 遅い揺らぎ(長い窓の移動平均 − 区間の平均・D339)・> 0 = 中間の帯(短い窓の平均 − 長い窓の平均・D364)。"""
    p = torch.from_numpy(lg.T.copy()).float().softmax(0)
    r = torch.from_numpy(c[:, 1:25].T.copy()).float() - torch.from_numpy(T).float() @ p / 10
    lw, sw = BAND["w"]
    s = F.avg_pool1d(F.pad(r[None], (lw // 2, lw - lw // 2 - 1), mode="replicate"), lw, 1)[0].numpy()
    if sw > 0:
        m = F.avg_pool1d(F.pad(r[None], (sw // 2, sw - sw // 2 - 1), mode="replicate"), sw, 1)[0].numpy()
        return m - s
    return s - s.mean(1, keepdims=True)


def fit_stats(ncomp: int, exclude: set) -> tuple:
    """e2e/fem の実女声(exclude の話者を除く・表のある話者・SEG 以上)の遅い揺らぎ → (基底 B [24, ncomp]・成分ごとの平均の振幅スペクトル・成分ごとの標準偏差・情報)。"""
    E2 = P.H / "e2e"
    TD = P.ROOT / "data/rvoc_tables_real"
    D = []
    for m in json.loads((E2 / "meta.json").read_text())["fem"]:
        f = E2 / "fem" / f"{m['key']}.cond"
        if m["spk"] in exclude or not f.exists() or not (TD / f"{m['spk']}.npz").exists():
            continue
        c, lg = P.load_dump(E2 / "fem", m["key"])
        if len(c) >= SEG:
            D.append(resid(c, lg, np.load(TD / f"{m['spk']}.npz")["T"]))
    X = np.concatenate(D, 1)
    U, S, _ = np.linalg.svd(X @ X.T / X.shape[1])
    B = U[:, :ncomp]
    psd = np.zeros((ncomp, SEG // 2 + 1))
    for d in D:
        psd += np.abs(np.fft.rfft((B.T @ d[:, :SEG]) * np.hanning(SEG)[None], axis=1)) ** 2
    tgt = np.sqrt((B.T @ X).var(1))
    return B, np.sqrt(psd / len(D)), tgt, {"clips": len(D), "explained": (S[:ncomp] / S.sum()).round(3).tolist(), "drift_rms": round(float(np.sqrt((X ** 2).mean())), 4)}


def make_noise(B: np.ndarray, amp: np.ndarray, tgt: np.ndarray, T_: int, seed: int) -> np.ndarray:
    """成分ごとに白色雑音へ平均の振幅スペクトルをかけ(乱数の位相・SEG ごと)・成分の標準偏差を学習の揺らぎにそろえ・基底で 24 次元へ。"""
    rng = np.random.default_rng(seed)
    n = int(np.ceil(T_ / SEG)) * SEG
    out = np.zeros((B.shape[1], n))
    for k in range(B.shape[1]):
        for o in range(0, n, SEG):
            out[k, o:o + SEG] = np.fft.irfft(amp[k] * np.exp(2j * np.pi * rng.random(SEG // 2 + 1)), SEG)
    z = out[:, :T_]
    z = z / (z.std(1, keepdims=True) + 1e-9) * tgt[:, None]
    return (B @ z).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    ap.add_argument("--ncomp", type=int, default=8)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    held = P.e2e_held()
    B, amp, tgt, info = fit_stats(a.ncomp, {h[0] for h in held})
    print(json.dumps(info), flush=True)
    np.savez(O / "slow_stats.npz", B=B, amp=amp, tgt=tgt)
    noise = lambda T_, seed: make_noise(B, amp, tgt, T_, seed)

    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX["on"], P.CTX["ch"], P.CTX["adapt"] = bool(st.get("ctx", False)), int(st.get("ctx_ch", 192)), bool(st.get("adapt", False))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    sr, _ = P.eval_items(held, "table", dev)
    groups = []
    with torch.no_grad():
        for sg in a.sigmas:
            od = O / f"N{sg}"
            od.mkdir(exist_ok=True)
            for i, (nm, c, lg, spk) in enumerate(sr):
                cond, exc, s_ = P.cond_exc(g, c, lg, spk, dev)
                nz = torch.from_numpy(noise(cond.shape[-1], 339 + i)).to(dev)[None]
                cond = torch.cat([cond[:, :1], cond[:, 1:25] + sg * nz, cond[:, 25:]], 1)
                y = g(cond, exc, s_)[0].cpu().numpy()[N.DELAY:]
                sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
            groups.append(f"N{sg}={od}/h*_SR.wav")
            print("rendered", sg, flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())

"""nvoc の評価: held24(除外話者の実音声 8s×24)のコピー合成を合格錨 BigVGAN v2 と同じ物差しで測る。

  logmel  多尺度 log-mel L1(train_s1_1.MEL_SPECS・遅延補正 y[DELAY:] vs x[:−DELAY])。錨 BigVGAN 0.195
  pesq    広帯域 PESQ(16kHz・元音声を参照)
  hf_db   6–16kHz 帯のエネルギー差(出力 − 元・dB・有音フレーム平均)。負 = こもり
  future  学習済み重み・実音声での未来不変性(編集点より前の出力が変わらないか)

    CUDA_VISIBLE_DEVICES=0 uv run python eval_nvoc.py --ckpt ../results/nvoc1/last.pt --out ../results/nvoc1/eval_last.json
    CUDA_VISIBLE_DEVICES=0 uv run python eval_nvoc.py --anchor --out ../results/nvoc1/anchor_bigvgan.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1

ROOT = Path(__file__).resolve().parent.parent
_MELS: dict = {}


def held_items() -> list[dict]:
    import s0_artic as S
    from train_ddsp_vc import index, load48
    spk, _, ev = index()
    out = []
    for it in S.held24():
        k = next((kk for kk in ev if kk.endswith("/" + it["spk"])), None)
        if k is None:
            continue
        z, w, _ = next(r for r in spk[k] if r[0].stem == it["stem"])
        x = load48(w)[:8 * N.SR]
        x = x[:len(x) // N.HOP * N.HOP]
        f0 = np.load(z)["f0"].astype(np.float32)[:len(x) // N.HOP]
        f0 = np.pad(f0, (0, len(x) // N.HOP - len(f0)))
        out.append({"stem": it["stem"], "x": x, "f0": f0})
    return out


def hf_db(y: np.ndarray, x: np.ndarray) -> float:
    import scipy.signal as ss
    f, _, Y = ss.stft(y, N.SR, nperseg=2048, noverlap=2048 - 480)
    _, _, X = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - 480)
    b = (f >= 6000) & (f <= 16000)
    ey = (np.abs(Y[b]) ** 2).sum(0) + 1e-12
    ex = (np.abs(X[b]) ** 2).sum(0) + 1e-12
    tot = (np.abs(X) ** 2).sum(0) + 1e-12
    act = 10 * np.log10(tot) > 10 * np.log10(tot.max()) - 40
    return float(np.mean(10 * np.log10(ey[act]) - 10 * np.log10(ex[act])))


def amline(w: np.ndarray) -> float:
    """フレーム周期(200/400/600Hz)の包絡変調の線(2–12kHz の 3 帯の平均・近傍比 dB)。本物 ≈ 0。耳の「ガビガビ」(2026-10-01)。"""
    import scipy.signal as ss
    out = []
    for lo, hi in ((2000, 4000), (4000, 8000), (8000, 12000)):
        sos = ss.butter(6, [lo, hi], btype="band", fs=N.SR, output="sos")
        env = np.abs(ss.hilbert(ss.sosfiltfilt(sos, w.astype(np.float64))))
        env = ss.resample_poly(env, 1, 12)
        f, P = ss.welch(env - env.mean(), fs=N.SR / 12, nperseg=4096)
        pk, bg = 0.0, 0.0
        for h in (200, 400, 600):
            on = (f >= h - 3) & (f <= h + 3)
            nb = (f >= h - 40) & (f <= h + 40) & ~on
            pk += P[on].sum()
            bg += P[nb].mean() * on.sum()
        out.append(10 * np.log10(pk / max(bg, 1e-20)))
    return float(np.mean(out))


def metrics(y: np.ndarray, x: np.ndarray, delay: int, dev: str) -> dict:
    from pesq import pesq
    from scipy.signal import resample_poly
    if dev not in _MELS:
        _MELS[dev] = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    n = min(len(y) - delay, len(x))
    yy, xx = y[delay:delay + n].astype(np.float32), x[:n].astype(np.float32)
    lm = float(logmel_l1(torch.from_numpy(yy).to(dev)[None, None], torch.from_numpy(xx).to(dev)[None, None], _MELS[dev]))
    try:
        pq = float(pesq(16000, resample_poly(xx, 1, 3), resample_poly(yy, 1, 3), "wb"))
    except Exception:
        pq = float("nan")
    return {"logmel": lm, "pesq": pq, "hf_db": hf_db(yy, xx), "am_db": amline(yy)}


def run_nvoc(model: N.NVoc, x: np.ndarray, f0: np.ndarray, dev: str, seed: int = 0) -> np.ndarray:
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(1, len(x), generator=g).to(dev)
    xin = torch.cat([torch.zeros(1, N.WIN - N.HOP), torch.from_numpy(x)[None]], -1).to(dev)
    with torch.no_grad():
        return model(xin, torch.from_numpy(f0)[None].to(dev), noise)[0].cpu().numpy()


def summarize(rows: list[dict]) -> dict:
    return {k: {"mean": round(float(np.nanmean([r[k] for r in rows])), 4), "median": round(float(np.nanmedian([r[k] for r in rows])), 4)}
            for k in rows[0]}


def held_eval(model: N.NVoc, dev: str, items: list[dict]) -> dict:
    model.eval()
    rows = [metrics(run_nvoc(model, it["x"], it["f0"], dev), it["x"], N.DELAY, dev) for it in items]
    return summarize(rows)


def future_probe(model: N.NVoc, items: list[dict], dev: str) -> dict:
    a, b = items[0], items[1]
    n = min(len(a["x"]), len(b["x"]), 4 * N.SR) // N.HOP * N.HOP
    x, f0 = a["x"][:n].copy(), a["f0"][:n // N.HOP].copy()
    cut = 200
    y = run_nvoc(model, x, f0, dev)
    x2, f2 = x.copy(), f0.copy()
    x2[cut * N.HOP:] = b["x"][cut * N.HOP:n]
    f2[cut:] = b["f0"][cut:n // N.HOP]
    y2 = run_nvoc(model, x2, f2, dev)
    d = np.abs(y2 - y)
    ch = np.nonzero(d > 1e-6 * max(1e-3, float(np.abs(y).max())))[0]
    if len(ch) == 0:
        return {"inconclusive": True}
    first = int(ch[0])
    return {"inconclusive": False, "edit_sample": cut * N.HOP, "first_change": first,
            "lookahead_samples": max(0, cut * N.HOP - first), "note": "出力ブロック t は入力 (t+1)·HOP 到着時に確定・出力の時刻 m は入力 m − DELAY に対応"}


def anchor(items: list[dict], dev: str) -> dict:
    import librosa
    import bigvgan
    from bigvgan.env import AttrDict
    from bigvgan.meldataset import get_mel_spectrogram
    snaps = Path.home() / ".cache/huggingface/hub/models--nvidia--bigvgan_v2_44khz_128band_512x/snapshots"
    snap = sorted(snaps.iterdir())[-1]
    voc = bigvgan.BigVGAN(AttrDict(json.loads((snap / "config.json").read_text())), use_cuda_kernel=False)
    voc.load_state_dict(torch.load(snap / "bigvgan_generator.pt", map_location="cpu")["generator"])
    voc.remove_weight_norm()
    voc = voc.eval().to(dev)
    rows = []
    for it in items:
        x44 = librosa.resample(it["x"].astype(np.float64), orig_sr=N.SR, target_sr=44100).astype(np.float32)
        with torch.no_grad():
            mel = get_mel_spectrogram(torch.from_numpy(x44)[None].to(dev), voc.h)
            y44 = voc(mel).squeeze().cpu().numpy().astype(np.float64)
        y = librosa.resample(y44, orig_sr=44100, target_sr=N.SR)[:len(it["x"])]
        rows.append(metrics(y, it["x"], 0, dev))
    return summarize(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--random", action="store_true", help="未学習の既定構成(起動前の未来不変性プローブ)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = held_items()
    rep: dict = {"held_n": len(items)}
    if a.anchor:
        rep["bigvgan_v2_44k"] = anchor(items, dev)
    else:
        if a.random:
            torch.manual_seed(0)
            m = N.NVoc().to(dev)
            rep["step"] = 0
        else:
            st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
            c = st["cfg"]
            m = N.NVoc(ch=c["ch"], kernels=tuple(tuple(k) for k in c["kernels"]), dils=tuple(c["dils"])).to(dev)
            m.load_state_dict(st["ema"])
            rep["step"] = int(st["step"])
            rep["held24"] = held_eval(m, dev, items)
        rep["future_invariance_real_audio"] = future_probe(m, items, dev)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

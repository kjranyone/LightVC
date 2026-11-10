"""paper_snv rev29 段 I15(M21)の背景の透かし: 前段 D の学習の背景(MUSAN の音楽と雑音・FMA の器楽・操作者の音)に、Irodori の生成と同じ SilentCipher の透かしを足した版を作る。
学習の声(Irodori の生成)には全て透かしがあり、背景と本物の操作者の声には無い = 段 I14 の D は透かしを声の手がかりにした(results/snv_i15/pre_codec.json・pre_bgwm.json)。背景にも同じ透かしを足し、透かしが声と背景を分けないようにする(入力側だけ・声の透かしは外さない)。
床 = 色つき雑音(パワー ∝ f^−β・β 0 / 0.5 / 1 / 1.5 / 2・各 8 本 × 20 s)の透かし付きの束(data/snv_sep/wm/floor_*.wav)= 学習の入力の無音をデジタルの 0 にしない。
合成の手拍子 = 減衰する雑音の破裂(τ 3〜25 ms)を 2 次の帯域通過(中心 0.6〜3 kHz・Q 0.7〜3)と素通しで混ぜ、短い部屋の尾を足した拍を 0.1〜0.6 s おきに並べた 20 s の列 32 本(data/snv_sep/wm/clap_*.wav・透かし付き)= 操作者の手の音の合成の源(入力側だけ)。
rev33(段 I18): FSD50K の開発の組の手の音(data/snv_sep/fsd50k_dev_sources.json)があれば、その透かしの版も作る。
出力 = data/snv_sep/wm/{元のファイルの stem}.wav(44.1 kHz・PCM16・元と同じ長さ)と data/snv_sep/wm/index.json(元の道 → 透かしの版)。
    cd ../../Irodori-TTS && HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-sync python ../LightVC/training/snv_sep_wm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
OUT = ROOT / "data/snv_sep/wm"
WSR = 44100
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 60 * WSR


def sources() -> list[str]:
    ms = json.loads((ROOT / "data/snv_p_feat_rec2/sources.json").read_text())
    fm = json.loads((ROOT / "data/snv_sep/fma_sources.json").read_text())
    op = json.loads((ROOT / "data/snv_sep/operator_sources.json").read_text())
    fd = ROOT / "data/snv_sep/fsd50k_dev_sources.json"
    fsd = set(json.loads(fd.read_text())["pools"]["op"]) if fd.exists() else set()
    return sorted(set(ms["pools"]["music"]) | set(ms["pools"]["noise"]) | set(fm["pools"]["music"]) | set(op["pools"]["train"]) | set(op["pools"]["dev"]) | fsd)


def floors() -> list[tuple[str, np.ndarray]]:
    out = []
    for beta in (0.0, 0.5, 1.0, 1.5, 2.0):
        for k in range(8):
            g = np.random.default_rng((int(beta * 10), k))
            n = 20 * WSR
            X = np.fft.rfft(g.standard_normal(n))
            f = np.fft.rfftfreq(n, 1 / WSR)
            X *= np.where(f > 20, (np.maximum(f, 20) / 1000.0) ** (-beta / 2), 0.0)
            x = np.fft.irfft(X, n)
            out.append((f"floor_b{int(beta * 10):02d}_{k}", (x / (np.sqrt((x ** 2).mean()) + 1e-12) * 0.05).astype(np.float32)))
    return out


def claps() -> list[tuple[str, np.ndarray]]:
    from scipy.signal import butter, sosfilt
    out = []
    for k in range(32):
        g = np.random.default_rng((77, k))
        n = 20 * WSR
        x = np.zeros(n)
        t = int(g.uniform(0.05, 0.3) * WSR)
        while t < n - WSR // 2:
            L = int(0.25 * WSR)
            tt = np.arange(L) / WSR
            burst = g.standard_normal(L) * np.exp(-tt / g.uniform(0.003, 0.025))
            fc, q = g.uniform(600, 3000), g.uniform(0.7, 3.0)
            band = sosfilt(butter(2, [fc / (1 + 0.5 / q), min(fc * (1 + 0.5 / q), WSR / 2 - 100)], "bandpass", fs=WSR, output="sos"), burst)
            a = g.uniform(0.2, 0.8)
            e = a * band / (np.abs(band).max() + 1e-9) + (1 - a) * burst / (np.abs(burst).max() + 1e-9)
            e += g.standard_normal(L) * np.exp(-tt / g.uniform(0.05, 0.2)) * 10 ** (-20 / 20) * np.abs(e).max()
            x[t:t + L] += e * 10 ** (g.uniform(-6, 0) / 20)
            t += int(g.uniform(0.1, 0.6) * WSR)
        out.append((f"clap_{k:02d}", (x / (np.abs(x).max() + 1e-9) * 0.5).astype(np.float32)))
    return out


def main() -> int:
    sys.path.insert(0, str(IRODORI))
    from irodori_tts.watermark import SilentCipherWatermarker
    wm = SilentCipherWatermarker(device=DEV)
    assert wm.ready, "SilentCipher が使えない"
    OUT.mkdir(parents=True, exist_ok=True)
    ip = OUT / "index.json"
    index = json.loads(ip.read_text()) if ip.exists() else {}

    def mark(x: np.ndarray) -> np.ndarray:
        parts = []
        for s in range(0, len(x), CHUNK):
            seg = torch.from_numpy(x[s:s + CHUNK].astype(np.float32))[None].to(DEV)
            parts.append(wm.encode_batch([seg], sample_rate=WSR)[0].reshape(-1).float().cpu().numpy()[: seg.shape[-1]])
        return np.concatenate(parts)

    for name, x in floors() + claps():
        q = OUT / f"{name}.wav"
        if not q.exists():
            sf.write(q, np.clip(mark(x), -1, 1), WSR, subtype="PCM_16")
        index[name] = str(q)
    srcs = sources()
    for n, p in enumerate(srcs):
        q = OUT / f"{Path(p).stem}.wav"
        if p in index and q.exists():
            continue
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != WSR:
            x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, WSR)[0].numpy()
        sf.write(q, np.clip(mark(x), -1, 1), WSR, subtype="PCM_16")
        index[p] = str(q)
        if (n + 1) % 100 == 0:
            ip.write_text(json.dumps(index, indent=0))
            print("marked", n + 1, "/", len(srcs), flush=True)
    ip.write_text(json.dumps(index, indent=0))
    print("done", len(index), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

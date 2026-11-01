"""D687 の組の準備(推論だけ)。A = devvctkm を関門の形に(操作者の参照 op_PPPP.wav = ref_<spk> へのリンク・前置き pre_<key>.wav = 参照の末尾 12 s + SRC 8 s)。
B = devff2n = devff2 の入力に外郎売の最も静かなフレームの平均パワースペクトルの定常の雑音を −37.6 dB(発話の 95 パーセンタイルのフレームのパワー比)で足す。規則は results/conv_p0/d687/prereg.yaml。
    uv run python d687_build.py
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
SR, N, HOP = 48000, 2048, 480
FLOOR_DB = -37.6


def load(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    assert sr == SR, p
    return x.mean(1)


def frames_ms(x: np.ndarray) -> np.ndarray:
    fr = np.lib.stride_tricks.sliding_window_view(x, N)[::HOP] * np.hanning(N)
    return (fr ** 2).mean(1)


def noise_psd() -> np.ndarray:
    S = H / "d487_uiro"
    acc, n = np.zeros(N // 2 + 1), 0
    for p in sorted((S / "d56").glob("*_SRC.wav")) + sorted((S / "d56" / "_work").glob("op_*.wav")):
        x = load(p)
        fr = np.lib.stride_tricks.sliding_window_view(x, N)[::HOP] * np.hanning(N)
        e = 10 * np.log10((fr ** 2).mean(1) + 1e-20)
        sel = fr[e <= np.percentile(e, 5) + 3.0]
        acc += (np.abs(np.fft.rfft(sel, axis=1)) ** 2).sum(0)
        n += len(sel)
    return acc / n


def shaped(psd: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    m = 1 << int(np.ceil(np.log2(n)))
    f = np.interp(np.fft.rfftfreq(m, 1 / SR), np.fft.rfftfreq(N, 1 / SR), np.sqrt(psd))
    y = np.fft.irfft(f * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))), m)[:n]
    return y / np.sqrt(np.mean(y ** 2))


def add_noise(x: np.ndarray, psd: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p95 = np.percentile(frames_ms(x), 95) / 0.375
    return x + shaped(psd, len(x), rng) * np.sqrt(p95 * 10 ** (FLOOR_DB / 10))


def build_vctk() -> dict:
    S = H / "devvctkm"
    w = S / "d56" / "_work"
    n_link = n_pre = 0
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        key = f"p{r['pi']:04d}_t{r['ti']:03d}"
        op = w / f"op_{r['pi']:04d}.wav"
        if not op.exists():
            op.symlink_to(f"ref_{r['spk']}.wav")
            n_link += 1
        pre = w / f"pre_{key}.wav"
        if not pre.exists():
            ref = load(w / f"ref_{r['spk']}.wav")[-12 * SR:]
            ref = np.pad(ref, (12 * SR - len(ref), 0))
            sf.write(pre, np.concatenate([ref, load(S / "d56" / f"{key}_SRC.wav")[: 8 * SR]]).astype(np.float32), SR)
            n_pre += 1
    return {"vctk_op_links": n_link, "vctk_pre": n_pre}


def build_noise() -> dict:
    S, O = H / "devff2", H / "devff2n"
    (O / "d56" / "_work").mkdir(parents=True, exist_ok=True)
    shutil.copy(S / "devset.json", O / "devset.json")
    for d in ("d85_work", "rr_en", "d37"):
        if not (O / d).exists():
            (O / d).symlink_to(S / d)
    psd = noise_psd()
    np.save(O / "noise_psd.npy", psd)
    rng = np.random.default_rng(687)
    fl = []
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        key, pi = f"p{r['pi']:04d}_t{r['ti']:03d}", r["pi"]
        op = O / "d56" / "_work" / f"op_{pi:04d}.wav"
        if not op.exists():
            sf.write(op, add_noise(load(S / "d56" / "_work" / f"op_{pi:04d}.wav"), psd, rng).astype(np.float32), SR)
        pre = O / "d56" / "_work" / f"pre_{key}.wav"
        if not pre.exists():
            y = add_noise(load(Path(os.path.realpath(S / "d56" / "_work" / f"pre_{key}.wav"))), psd, rng)
            sf.write(pre, y.astype(np.float32), SR)
            sf.write(O / "d56" / f"{key}_SRC.wav", y[12 * SR:].astype(np.float32), SR)
        e = 10 * np.log10(frames_ms(load(O / "d56" / f"{key}_SRC.wav")) + 1e-20)
        fl.append(float(np.percentile(e, 5) - np.percentile(e, 95)))
    return {"devff2n_pairs": len(fl), "floor_db_median": round(float(np.median(fl)), 2)}


def main() -> int:
    rep = build_vctk() | build_noise()
    print(rep, flush=True)
    (Path(__file__).resolve().parent.parent / "results/conv_p0/d687/build.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

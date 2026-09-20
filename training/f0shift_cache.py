"""F0シフト増強用latentキャッシュ: WORLDフォルマント保存ピッチシフト→Y-S1再エンコード。

学習ペアからサンプルした発話を、発話毎のランダムシフト量(上方はcodec学習範囲の
約700Hzにキャップ)でWORLD再合成し、codecでエンコードして
data/f0shift_latent/<spk>/<stem>.pt = {z[T,32] half, st, ratio} を書く。
条件(mel/content/energy)は元音声のものを使うため、wavは保存しない。

phase1(CPU並列): WORLD再合成 -> data/f0shift_tmp/<spk>/<stem>.wav(int16 48k)
phase2(GPU): encode -> latent保存、tmp削除

    CUDA_VISIBLE_DEVICES=0 uv run python f0shift_cache.py --n 8000 --workers 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/female_real_feat"
F0FIX = ROOT / "data/female_real_f0fix"
TMP = ROOT / "data/f0shift_tmp"
OUT = ROOT / "data/f0shift_latent"
SR = 44100
CAP_HZ = 700.0


def shift_one(args: tuple[str, str, str, float]) -> tuple[str, str]:
    stem_pt, wav_out, f0fix_pt, st = args
    try:
        d = torch.load(stem_pt, map_location="cpu", weights_only=False)
        f0f = torch.load(f0fix_pt, map_location="cpu",
                         weights_only=False)["f0"].numpy()
        w44, _ = librosa.load(d["path"], sr=SR, mono=True)
        w64 = w44.astype(np.float64)
        f0, t = pyworld.harvest(w64, SR, f0_floor=65, f0_ceil=1000,
                                frame_period=5.0)
        f0 = pyworld.stonemask(w64, f0, t, SR)
        sp = pyworld.cheaptrick(w64, f0, t, SR)
        ap = pyworld.d4c(w64, f0, t, SR)
        r = 2.0 ** (st / 12.0)
        f0s = np.where(f0 > 0, f0 * r, f0)
        y = pyworld.synthesize(f0s, sp, ap, SR, frame_period=5.0)
        y48 = librosa.resample(y, orig_sr=SR, target_sr=48000)
        sf.write(wav_out, (np.clip(y48, -1, 1) * 32767.0).astype(np.int16), 48000)
        return (str(wav_out), f"{st:.3f}")
    except Exception as e:  # noqa: BLE001
        return ("", f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    stems = []
    for spk in sorted(SRC.iterdir()):
        if spk.is_dir():
            for f in spk.glob("*.pt"):
                if (F0FIX / spk.name / f.name).exists():
                    stems.append(f)
    rng.shuffle(stems)
    stems = stems[:a.n]

    jobs, meta = [], {}
    for f in stems:
        d = torch.load(f, map_location="cpu", weights_only=False)
        f0f = torch.load(F0FIX / f.parent.name / f.name, map_location="cpu",
                         weights_only=False)["f0"].numpy()
        fv = f0f[f0f > 60]
        if len(fv) < 60:
            continue
        med = float(np.median(fv))
        st_max = min(12.0, 12.0 * np.log2(CAP_HZ / med))
        st_min = max(-4.0, -12.0 * np.log2(med / 65.0))
        if st_max - st_min < 1.0:
            continue
        st = rng.uniform(st_min, st_max)
        od = TMP / f.parent.name
        od.mkdir(parents=True, exist_ok=True)
        wav_out = od / (f.stem + ".wav")
        if wav_out.exists():
            continue
        jobs.append((str(f), str(wav_out),
                     str(F0FIX / f.parent.name / f.name), st))
        meta[f.stem] = st
    print(f"  phase1 jobs {len(jobs)}", flush=True)
    t0 = time.time()
    errs = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (_, msg) in enumerate(ex.map(shift_one, jobs, chunksize=8)):
            if msg and ":" in msg:
                errs += 1
            if (i + 1) % 1000 == 0:
                print(f"  synth {i+1}/{len(jobs)} err {errs} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    from causal_codec import CausalCodec, HOP_LENGTH
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt",
                     map_location=dev)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()
    n_done = 0
    for spk_dir in sorted(TMP.iterdir()):
        if not spk_dir.is_dir():
            continue
        od = OUT / spk_dir.name
        od.mkdir(parents=True, exist_ok=True)
        for wav in sorted(spk_dir.glob("*.wav")):
            outf = od / (wav.stem + ".pt")
            if outf.exists():
                continue
            x, sr = sf.read(str(wav), dtype="float32")
            x = x / 32768.0
            n = len(x) // HOP_LENGTH * HOP_LENGTH
            if n < 48000:
                continue
            with torch.no_grad():
                z = codec.encode(torch.from_numpy(x[:n])[None, None]
                                 .to(dev))[0].transpose(0, 1).cpu()
            torch.save({"z": z.half(), "st": meta[wav.stem],
                        "ratio": 2.0 ** (meta[wav.stem] / 12.0)}, outf)
            n_done += 1
        for wav in spk_dir.glob("*.wav"):
            wav.unlink()
    summary = {"synth_jobs": len(jobs), "synth_errs": errs,
               "latents": n_done, "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "_summary.json").write_text(json.dumps(summary))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

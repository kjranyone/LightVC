"""Artic-A2 Step 2 用の特徴量をフルコーパスで事前計算する(製品と同じ解析: p24・解析窓の先読み 10ms)。

manifest(data/kansei_vc/manifests/canonical_utterances.tsv)の real_female・tts_base・tts_emotional_live・tts_male_ja の全発話について
  lar  [K,24] float16  LPC の LAR(48kHz・p24・H=120=2.5ms 毎・解析窓の先読み LA=480=10ms=中心化・出力遅延 D≥10ms の中に収まる)
  f0   [T]    float32  因果 YIN(窓 1536・hop 240・有声 d'<0.45・0=無声)
  lev  [K]    float16  LPC と同じ窓(先読み 10ms の 20ms 窓)のプリエンファシス後 RMS [dB]
  sr   元の標本化周波数(22.05k/32k 録音は帯域が切れている → 学習側で損失の帯域を分ける)
を data/artic_feat/<source_type>/<speaker>/<utt>.npz に保存する。既存ファイルは飛ばす(再開可能)。

    OMP_NUM_THREADS=1 uv run python prep_lar.py --workers 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "data/kansei_vc/manifests/canonical_utterances.tsv"
OUT = ROOT / "data/artic_feat"
TYPES = ("real_female", "tts_base", "tts_emotional_live", "tts_male_ja")
LA = 480


def jobs() -> list[tuple[str, str, str, int]]:
    out = []
    for r in csv.DictReader(open(MAN), delimiter="\t"):
        if r["source_type"] not in TYPES:
            continue
        wav = (ROOT / r["wav_path"].replace("../", "", 1)).resolve()
        out.append((r["source_type"], r["speaker_id"] or wav.parent.name, str(wav), int(r["sample_rate"] or 0)))
    return out


def one(j: tuple[str, str, str, int]) -> str:
    st, spk, wav, sr = j
    dst = OUT / st / spk / (Path(wav).stem + ".npz")
    if dst.exists():
        return "skip"
    try:
        import soundfile
        import librosa
        x, s0 = soundfile.read(wav, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if s0 != D.SR:
            x = librosa.resample(x, orig_sr=s0, target_sr=D.SR)
        x = x.astype(np.float64)
        lar = D.lar_frames(x, 24, la=LA)
        f0, _ = D.causal_yin(x, voi_max=0.45)
        xp = D.preemph(x)
        K = lar.shape[0]
        pad = np.concatenate([np.zeros(D.W - LA), xp, np.zeros(D.H + LA)])
        fr = np.lib.stride_tricks.sliding_window_view(pad, D.W)[0:K * D.H:D.H]
        lev = 10 * np.log10(np.maximum((fr ** 2).mean(1), 1e-12))
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, lar=lar.astype(np.float16), f0=f0.astype(np.float32), lev=lev.astype(np.float16), sr=np.int32(s0),
                 la=np.int32(LA))
        return "ok"
    except Exception as ex:
        return f"err {wav} {ex}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    js = jobs()
    print("jobs", len(js), flush=True)
    from multiprocessing import Pool
    n_ok = n_err = 0
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, js, chunksize=16)):
            if r.startswith("err"):
                n_err += 1
                print(r, flush=True)
            else:
                n_ok += 1
            if i % 5000 == 0:
                print("progress", i, "ok", n_ok, "err", n_err, flush=True)
    print("done ok", n_ok, "err", n_err, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

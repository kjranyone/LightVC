"""D135(評価の較正だけ・オーナー判断 10-12「評価の較正だけなら可」・製品の設計・学習・推論には使わない): 同一性の合格線(学習外 LDA 0.42)が
ゼロショットの変換で届きうる線かを、既存の強いゼロショット VC(kNN-VC・非因果・WavLM 層 6 の特徴を目標の参照フレームの k 近傍平均で置き換え HiFi-GAN で鳴らす・16kHz)で測る。
注意: 同一性の物差しも WavLM の特徴なので kNN-VC はこの物差しで有利に出やすい(循環)= 届かなければ合格線は非現実的・届けば物差し寄りの基準点として割り引いて読む。
組(製品と同じ参照の量):
  real = 実女声 30 目標 × 男声 60 組(D56 の SRC・参照 = d37/_work/ref_TTT.wav)→ <s>/d135_real/pNNNN_tNNN_KNN.wav(+ mkcen で中心)→ d4b_lda_verifier
  tts  = D106 の組(男声パラレル m000〜m023 → 女声 TTS の評価外 24 目標 × 5 文・参照 = 目標の別の 9 文の neutral)→ <s>/d106/m{i}_{A}_t{k}_KNN.wav → d118_identity_ceiling
    uv run python d135_knnvc_calib.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
FEM = ROOT / "data/female_tts_corpus"
MPAR = ROOT / "data/male_tts_v4l_par"
STY = ["neutral", "breathy", "low_tension", "soft", "warm"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--real_only", action="store_true")
    a = ap.parse_args()
    import torchaudio
    from scipy.signal import resample_poly
    from d6_unit_cov import load48
    S = Path(a.s)
    hub = Path.home() / ".cache/torch/hub/bshall_knn-vc_master"
    knn = torch.hub.load(str(hub), "knn_vc", source="local", prematched=True, pretrained=True, device="cuda")

    def w16(p: Path, sec: float | None = None) -> torch.Tensor:
        x = load48(p).astype(np.float32)
        if sec:
            x = x[: int(sec * 48000)]
        return torch.from_numpy(resample_poly(x, 1, 3).astype(np.float32))

    def conv(src: Path, refs: list) -> np.ndarray:
        q = knn.get_features(w16(src)[None])
        ms = knn.get_matching_set([w16(r)[None] for r in refs])
        y = knn.match(q, ms, topk=a.topk).cpu().numpy()
        return resample_poly(y, 3, 1).astype(np.float32)

    od = S / "d135_real"; od.mkdir(exist_ok=True)
    for f in sorted((S / "d56").glob("p*_SRC.wav")):
        ti = int(f.name[7:10])
        o = od / f"{f.stem[:-4]}_KNN.wav"
        if not o.exists():
            sf.write(o, conv(f, [S / "d37/_work" / f"ref_{ti:03d}.wav"]), 48000)
    print("real done", flush=True)
    if a.real_only:
        return 0
    males = sorted(d.name for d in MPAR.iterdir() if d.is_dir() and len(list(d.glob("f*.wav"))) == 10)[:24]
    fems = sorted(d.name for d in FEM.iterdir() if d.is_dir() and all((d / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in STY))
    random.Random(80).shuffle(fems)
    for mi, M in enumerate(males):
        A = fems[mi]
        for k in range(5):
            o = S / "d106" / f"{M}_{A}_t{k:02d}_KNN.wav"
            if not o.exists():
                sf.write(o, conv(MPAR / M / f"f{k:02d}.wav", [FEM / A / f"t{i:02d}_neutral.wav" for i in range(10) if i != k]), 48000)
    print("tts done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

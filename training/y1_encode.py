"""Y-1a: 女声フルコーパス + male を DACVAE latent へ一括事前計算（48k・25fps）。

出力: data/latent48/<corpus>/<spk>/<utt>.pt  = {z: [T,32], path, sr}
正規化: latent は dataset-global per-dim mean/std を別ファイルに出し
学習側で正規化（FM ABI・発話統計禁止）。

    cd Irodori-TTS && HF_HUB_OFFLINE=1 .venv/bin/python ../LightVC/training/y1_encode.py \
        [--limit 200]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Irodori-TTS"))
from irodori_tts.codec import DACVAECodec

LVC = Path("/home/kojirotanaka/kjranyone/LightVC")
OUT = LVC / "data/latent48"
SR = 48000

CORPORA = {
    "female_real": LVC / "female-dataset",
    "female_tts": LVC / "data/female_tts_corpus",
    "male_tts": LVC / "data/male_tts_corpus",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="話者あたり上限(0=全部)")
    ap.add_argument("--stats-sample", type=int, default=400,
                    help="per-dim 統計用に抽出する発話数")
    a = ap.parse_args()

    codec = DACVAECodec.load(device="cuda", deterministic_encode=True,
                             deterministic_decode=True, normalize_db=None)
    n_files, n_done = 0, 0
    stats_acc: list[torch.Tensor] = []
    t0 = time.time()
    for cname, root in CORPORA.items():
        spks = sorted([d for d in root.iterdir() if d.is_dir()])
        if a.limit:
            spks = spks[: a.limit]
        for spk in spks:
            od = OUT / cname / spk.name
            od.mkdir(parents=True, exist_ok=True)
            for wav in sorted(spk.glob("*.wav")):
                outf = od / (wav.stem + ".pt")
                n_files += 1
                if outf.exists():
                    continue
                try:
                    w, _ = librosa.load(str(wav), sr=SR, mono=True)
                    if len(w) < SR:                       # <1s はスキップ
                        continue
                    x = torch.from_numpy(w)[None]
                    z = codec.encode_waveform(x, SR)[0].cpu()
                    torch.save({"z": z.half(), "path": str(wav), "sr": SR}, outf)
                    n_done += 1
                    if len(stats_acc) < a.stats_sample:
                        stats_acc.append(z.float())
                except Exception as e:
                    print(f"  ERR {wav}: {e}", flush=True)
            if n_done % 500 < 5:
                print(f"  {cname}/{spk.name}: cum {n_done} ({time.time()-t0:.0f}s)",
                      flush=True)
    if stats_acc:
        Z = torch.cat(stats_acc, 0)
        mu, sd = Z.mean(0), Z.std(0).clamp(min=0.05)
        torch.save({"mu": mu, "sd": sd}, OUT / "abi.pt")
        print(f"  ABI: mu med {float(mu.median()):.4f} sd med {float(sd.median()):.4f}")
    print(f"\n  {n_done} utts encoded (skip {n_files - n_done}) -> {OUT}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

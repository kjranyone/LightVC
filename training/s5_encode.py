"""S1-5a: フルコーパスを Y-S1 codec で latent 化（自作 encoder・48k・100fps）。

出力: data/ys1_latent/<corpus>/<spk>/<utt>.pt = {z: [T,32] half, path}
統計: 全 latent 完走後に一括で per-dim mean/std（abi.pt）=前回の
「最初の400発話バグ」を繰り返さない。

    CUDA_VISIBLE_DEVICES=0 uv run python s5_encode.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/ys1_latent"
CORPORA = {
    "female_real": ROOT / "female-dataset",
    "male_tts": ROOT / "data/male_tts_corpus",
}


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ck.get("ema") or ck["net"])
    codec.eval()

    n_done = 0
    t0 = time.time()
    for cname, root in CORPORA.items():
        spks = sorted([d for d in root.iterdir() if d.is_dir()])
        for spk in spks:
            od = OUT / cname / spk.name
            od.mkdir(parents=True, exist_ok=True)
            for wav in sorted(spk.glob("*.wav")):
                outf = od / (wav.stem + ".pt")
                if outf.exists():
                    continue
                try:
                    x, sr = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
                    n = len(x) // HOP_LENGTH * HOP_LENGTH
                    if n < SAMPLE_RATE:
                        continue
                    with torch.no_grad():
                        z = codec.encode(
                            torch.from_numpy(x[:n].astype(np.float32))[None, None]
                            .to(dev))[0].transpose(0, 1).cpu()   # [T,32]
                    torch.save({"z": z.half(), "path": str(wav)}, outf)
                    n_done += 1
                except Exception as e:
                    print(f"  ERR {wav}: {e}", flush=True)
            if n_done and n_done % 2000 < 20:
                print(f"  {cname}/{spk.name}: cum {n_done} ({time.time()-t0:.0f}s)",
                      flush=True)
    # ABI 統計: サンプリングでなく全件走査は重いので 2000 発話の層化抽出
    import random
    rng = random.Random(0)
    all_pts = list(OUT.rglob("*.pt"))
    sel = rng.sample(all_pts, min(2000, len(all_pts)))
    acc = []
    for p in sel:
        try:
            z = torch.load(p, map_location="cpu", weights_only=False)["z"].float()
            if z.shape[0] > 4:
                acc.append(z)
        except Exception:
            continue
    Z = torch.cat(acc, 0)
    torch.save({"mu": Z.mean(0), "sd": Z.std(0).clamp(min=0.05)}, OUT / "abi.pt")
    print(f"\n  {n_done} utts -> {OUT}  abi: 2000 層化抽出 ({time.time()-t0:.0f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

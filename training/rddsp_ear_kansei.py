import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import json

from bigvgan.env import AttrDict
from bigvgan.meldataset import get_mel_spectrogram
from kansei_train import CACHE, DATA, SNAP, gain_match, octave_correct
from kansei_vocoder import KanseiVocoder

h = AttrDict(json.loads((SNAP / "config.json").read_text()))

SR = 44100
CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/kansei_30k_base.pt")
OUT = Path("../results/rddsp_ear")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    import librosa

    OUT.mkdir(parents=True, exist_ok=True)
    sd = torch.load(CKPT, map_location=DEV)
    gen = KanseiVocoder(causal=False).to(DEV).eval()
    gen.load_state_dict(sd["gen"] if "gen" in sd else sd)
    print(f"{CKPT} step={sd.get('step')} params={sum(p.numel() for p in gen.parameters())/1e6:.1f}M")

    uids = [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]
    for uid in uids:
        w = DATA / (uid + ".wav")
        if not w.exists():
            print(f"  MISSING {w}")
            continue
        x, _ = librosa.load(str(w), sr=SR, mono=True)
        f0 = octave_correct(np.load(CACHE / (uid + ".npz"))["f0"].astype(np.float32))
        nf = min(len(f0), len(x) // 512, int(8.0 * SR) // 512)
        gt = x[: nf * 512].astype(np.float32)
        mel = get_mel_spectrogram(torch.tensor(gt).unsqueeze(0), h).to(DEV)
        f0t = torch.tensor(f0[:nf]).unsqueeze(0).to(DEV)
        with torch.no_grad():
            y = gen(mel, f0t)[0].float().cpu().numpy()
        n = min(len(y), len(gt))
        sf.write(OUT / f"{uid}_gt.wav", gt[:n], SR)
        sf.write(OUT / f"{uid}_kansei.wav", gain_match(y[:n], gt[:n]), SR)
        print(f"  {uid}  {n/SR:.1f}s  rms_gt {np.sqrt((gt[:n]**2).mean()):.4f}")
    print(f"wrote -> {OUT.resolve()}")


if __name__ == "__main__":
    main()

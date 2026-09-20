import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from causal_mel import mel_la
from f0leak_probe import load_wav
from vocoder_a import CombISTFTHead

CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/vocoderA_c11/last.pt")
WAVS = sys.argv[2:] or sorted(str(p) for p in Path("../female-dataset").glob("af1ad5575a3fa383/*.wav"))[:3]


def main() -> None:
    torch.set_num_threads(4)
    sd = torch.load(CKPT, map_location="cpu")
    gen = CombISTFTHead(dim=512, n_blocks=8, nfft=512, hop=128, causal=True).eval()
    gen.load_state_dict(sd["gen"], strict=False)
    print(f"{CKPT}  step={sd.get('step')}  alpha={F.softplus(gen.alpha).item():.3f}  "
          f"f0range=[{gen.fmin:.1f},{gen.fmax:.1f}]Hz")
    for w in WAVS:
        x = torch.tensor(load_wav(w))[: 44100 * 4]
        mel = mel_la(x, n_fft=2048, hop=128, look=0)[0].unsqueeze(0)
        with torch.no_grad():
            h = gen.pre(F.pad(mel, (6, 0)))
            for b in gen.blocks:
                h = b(h)
            g = gen.f0head(F.pad(F.leaky_relu(h, 0.1), (6, 0))).float()
        svoi = torch.sigmoid(g[:, 1]).flatten().numpy()
        f0 = (gen.fmin + (gen.fmax - gen.fmin) * torch.sigmoid(g[:, 0])).flatten().numpy()
        hot = svoi > 0.1
        print(f"  {Path(w).name:28s} svoi mean {svoi.mean():.4f} p90 {np.percentile(svoi, 90):.4f} "
              f"| frames>0.1 {100 * hot.mean():5.1f}% | f0(hot) {f0[hot].mean() if hot.any() else float('nan'):6.1f}Hz")


if __name__ == "__main__":
    main()

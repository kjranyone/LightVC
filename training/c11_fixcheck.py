import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from causal_mel import mel_la
from f0leak_probe import load_wav
from vocoder_a import CausalISTFTHead, CombISTFTHead

CFG = dict(dim=512, n_blocks=8, nfft=512, hop=128, causal=True)
REF = "checkpoints/vocoderA_c10/c10_ref.pt"
WAV = sorted(str(p) for p in Path("../female-dataset/af1ad5575a3fa383").glob("*.wav"))[0]


def feats(m: CombISTFTHead, mel: torch.Tensor) -> torch.Tensor:
    x = m.pre(F.pad(mel, (6, 0)))
    for b in m.blocks:
        x = b(x)
    return m.f0head(F.pad(F.leaky_relu(x, 0.1), (6, 0))).float()


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(4)
    sd = torch.load(REF, map_location="cpu")["gen"]
    m = CombISTFTHead(**CFG).eval()
    m.load_state_dict(sd, strict=False)
    base = CausalISTFTHead(**CFG).eval()
    base.load_state_dict(sd)
    x = torch.tensor(load_wav(WAV))[: 44100 * 4]
    mel = mel_la(x, n_fft=2048, hop=128, look=0)[0].unsqueeze(0)

    with torch.no_grad():
        g = feats(m, mel)
    lg = g[:, 1].flatten()
    print(f"[1] init on TRAINED backbone + REAL speech: logit_svoi "
          f"min {lg.min():7.2f} mean {lg.mean():7.2f} max {lg.max():7.2f} | svoi max {torch.sigmoid(lg).max():.4f}")
    print(f"    (broken version measured min -16.32 / mean 3.76 / max 98.96 / svoi max 1.000)")

    with torch.no_grad():
        f0v = m.fmin + (m.fmax - m.fmin) * torch.sigmoid(g[:, 0:1])
        c = torch.cos(2 * torch.pi * m.fbin.view(1, -1, 1) / f0v.clamp(min=1.0))
        comb = F.softplus(m.alpha) * (c - c.mean(dim=1, keepdim=True))
    print(f"[2] zero-mean comb: per-frame bin-mean |{comb.mean(dim=1).abs().max():.2e}| "
          f"range [{comb.min():+.3f},{comb.max():+.3f}] (boosts at k*f0, cuts between, no energy bias)")

    m.warm = 1.0
    with torch.no_grad():
        gg = feats(m, mel)
        sv = (1.0 - m.warm) * torch.sigmoid(gg[:, 1:2]) + m.warm * m.warm_svoi
    print(f"[3] warm=1 -> svoi forced {sv.min().item():.3f}..{sv.max().item():.3f} (target {m.warm_svoi})")

    def f0grad(warm: float, bias1: float) -> float:
        mm = CombISTFTHead(**CFG)
        mm.load_state_dict(sd, strict=False)
        mm.warm = warm
        with torch.no_grad():
            mm.f0head.bias[1] = bias1
        y = mm(mel)
        (y - torch.randn_like(y)).abs().mean().backward()
        return mm.f0head.weight_v.grad[0].abs().sum().item()

    ga, gb = f0grad(1.0, -4.0), f0grad(0.0, -30.0)
    print(f"[4] f0 gradient  warm=1 {ga:.3e} | warm=0 & svoi~0 (the c11 dead state) {gb:.3e} "
          f"-> ratio {ga / max(gb, 1e-30):.1e}  (chicken-and-egg broken open)")

    m.warm = 0.0
    with torch.no_grad():
        m.f0head.bias[1] = -50.0
        d = (m(mel) - base(mel)).abs().max().item()
    print(f"[5] fallback svoi=0 vs c10: max|dy| {d:.3e}")

    m.load_state_dict(sd, strict=False)
    m.warm = 0.0
    mel2 = mel.clone()
    mel2[:, :, 200:] += 5.0
    with torch.no_grad():
        y1, y2 = m(mel), m(mel2)
    e = 200 * CFG["hop"]
    print(f"[6] causality: before boundary {(y1[:, :e] - y2[:, :e]).abs().max():.3e} | "
          f"after {(y1[:, e:] - y2[:, e:]).abs().max():.3e}")

    with torch.no_grad():
        m(mel)
        t0 = time.time()
        for _ in range(3):
            m(mel)
        dt = (time.time() - t0) / 3
    print(f"[7] RTF (4 threads CPU): {dt / 4.0:.3f}")


if __name__ == "__main__":
    main()

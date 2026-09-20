"""Multi-Scale Sub-Band CQT discriminator (arXiv:2311.14957). Targets the
over-smoothing / non-sharp-harmonic plateau that MPD+MRD leave (generator CAN
represent sharp harmonics -- overfit 3.0 -- but MPD/MRD do not reward harmonic
sharpness, so full-GAN plateaus at ~2.68). CQT = log-frequency, constant-Q ->
resolves harmonics/octaves -> pushes the generator toward sharp harmonics.
Generator-agnostic: added alongside MPD/MRD. Training-only (not shipped)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from nnAudio.features import CQT

LRELU = 0.1


class CQTSubDisc(nn.Module):
    """CQT sub-disc with per-octave sub-band pre-processing (paper's fix for CQT
    temporal desync = the grain). Each octave's real+imag processed independently
    then concatenated -> synchronized latents, before the main strided convs."""
    def __init__(self, sr=44100, B=24, fmin=110.0, octaves=7, hop=256):
        super().__init__()
        nb = B * octaves
        self.B, self.oct = B, octaves
        self.cqt = CQT(sr=sr, hop_length=hop, fmin=fmin, n_bins=nb, bins_per_octave=B,
                       output_format="Complex", verbose=False)
        Csb = 8
        self.sb = weight_norm(nn.Conv2d(2, Csb, (3, 9), padding=(1, 4)))   # per-octave sub-band
        C = 32
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(Csb, C, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(C, C, (3, 9), stride=(2, 1), dilation=(1, 1), padding=(1, 4))),
            weight_norm(nn.Conv2d(C, C, (3, 9), stride=(2, 1), dilation=(1, 2), padding=(1, 8))),
            weight_norm(nn.Conv2d(C, C, (3, 9), stride=(2, 1), dilation=(1, 4), padding=(1, 16))),
        ])
        self.post = weight_norm(nn.Conv2d(C, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):                      # x: [B, T] waveform
        with torch.autocast("cuda", enabled=False):
            c = self.cqt(x.float())            # [B, nb, T', 2]
        c = c.permute(0, 3, 1, 2).contiguous()  # [B, 2, nb, T]
        Bsz, _, nb, T = c.shape
        # per-octave sub-band: split nb into (oct, B), process each octave independently
        c = c.reshape(Bsz, 2, self.oct, self.B, T).permute(0, 2, 1, 3, 4).reshape(Bsz * self.oct, 2, self.B, T)
        c = F.leaky_relu(self.sb(c), LRELU)     # [Bsz*oct, Csb, B, T]
        Csb = c.shape[1]
        c = c.reshape(Bsz, self.oct, Csb, self.B, T).permute(0, 2, 1, 3, 4).reshape(Bsz, Csb, nb, T)
        fmap = [c]
        h = c
        for conv in self.convs:
            h = F.leaky_relu(conv(h), LRELU)
            fmap.append(h)
        h = self.post(h)
        fmap.append(h)
        return torch.flatten(h, 1, -1), fmap


class MSSubBandCQTDisc(nn.Module):
    def __init__(self, sr=44100, Bs=(24, 36, 48)):
        super().__init__()
        self.discs = nn.ModuleList([CQTSubDisc(sr=sr, B=B) for B in Bs])

    def forward(self, y, y_hat):               # [B,1,T] each
        yr, yg, fr, fg = [], [], [], []
        for d in self.discs:
            r, fmr = d(y.squeeze(1))
            g, fmg = d(y_hat.squeeze(1))
            yr.append(r); yg.append(g); fr.append(fmr); fg.append(fmg)
        return yr, yg, fr, fg


if __name__ == "__main__":
    d = MSSubBandCQTDisc().cuda()
    y = torch.randn(2, 1, 32768).cuda(); yh = torch.randn(2, 1, 32768, requires_grad=True).cuda()
    yr, yg, fr, fg = d(y, yh)
    print("scores", [tuple(s.shape) for s in yg], "| fmaps/disc", [len(f) for f in fg])
    yg[0].sum().backward()
    print("grad OK", yh.grad is not None, "| params",
          round(sum(p.numel() for p in d.parameters()) / 1e6, 2), "M")

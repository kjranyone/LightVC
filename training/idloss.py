"""出力側の同一性・内容の損失に使う、波形から勾配が通る埋め込み(C3・current/converter.md §3c)。全て凍結(重みは更新しない)・入力は 48kHz の波形 [B, L]。
  Ecapa    speechbrain ECAPA-TDNN(192 次元・正規化)
  WavlmSV  microsoft/wavlm-base-plus-sv(512 次元・正規化)
  ContentVec  content-vec の最終層(768 次元・フレーム 20ms)
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent


def to16(x48: torch.Tensor) -> torch.Tensor:
    return torchaudio.functional.resample(x48, 48000, 16000)


class Ecapa(torch.nn.Module):
    def __init__(self, dev: str):
        super().__init__()
        from speechbrain.inference.speaker import EncoderClassifier
        self.m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "pretrained_models/spkrec-ecapa-voxceleb"), run_opts={"device": dev})
        for p in self.m.mods.parameters():
            p.requires_grad_(False)
        self.m.eval()

    def forward(self, x48: torch.Tensor) -> torch.Tensor:
        y = to16(x48)
        wl = torch.ones(y.shape[0], device=y.device)
        feats = self.m.mods.compute_features(y)
        feats = self.m.mods.mean_var_norm(feats, wl)
        e = self.m.mods.embedding_model(feats, wl)
        return F.normalize(e.reshape(e.shape[0], -1), dim=-1)


class WavlmSV(torch.nn.Module):
    def __init__(self, dev: str):
        super().__init__()
        from transformers import WavLMForXVector
        self.m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval().float()
        for p in self.m.parameters():
            p.requires_grad_(False)

    def forward(self, x48: torch.Tensor) -> torch.Tensor:
        y = to16(x48)
        y = (y - y.mean(-1, keepdim=True)) / (y.std(-1, keepdim=True) + 1e-7)
        return F.normalize(self.m(input_values=y).embeddings, dim=-1)


class ContentVec(torch.nn.Module):
    def __init__(self, dev: str):
        super().__init__()
        from transformers import HubertModel
        self.m = HubertModel.from_pretrained("lengyue233/content-vec-best").to(dev).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    def forward(self, x48: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.m(to16(x48)).last_hidden_state.float(), dim=-1)


if __name__ == "__main__":
    import time
    dev = "cuda"
    x = (torch.randn(8, 96000, device=dev) * 0.05).requires_grad_(True)
    for name, M in (("ecapa", Ecapa), ("wavlm_sv", WavlmSV), ("contentvec", ContentVec)):
        m = M(dev)
        torch.cuda.synchronize(); t0 = time.time()
        e = m(x)
        loss = (e ** 2).sum() if e.dim() == 2 else e.mean()
        g, = torch.autograd.grad(loss, x)
        torch.cuda.synchronize()
        print(name, tuple(e.shape), "grad finite", bool(torch.isfinite(g).all()), "grad norm %.3g" % g.norm().item(), "%.2fs (B8 x 2s fwd+bwd)" % (time.time() - t0), "mem %.1fGB" % (torch.cuda.max_memory_allocated() / 1e9))

"""S1-5c: namikawa 変換レンダ（E/P/S → CFM 1-step → Y-S1 decoder → 波形）。

初めて VC の音が出る経路。prosody はまず固定 +17 半音（P の実装前の
診断位置づけ・P 実装後に差し替え）。

    CUDA_VISIBLE_DEVICES=0 uv run python s5_render.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH as HOP48
from train_cfmys import CFMYS, ar_noise
from train_vc_e import E1

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/s5_render"
SR48 = 48000
F0_FPS = 44100 / 512


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfm", default=str(ROOT / "results/s5_cfm/s5_cfm_best.pt"))
    ap.add_argument("--semitones", type=float, default=17.0)
    ap.add_argument("--K", type=int, default=1,
                    help="CFM Eulerサンプリングステップ数(interp学習モデルは8)")
    ap.add_argument("--out", default="namikawa_s5.wav")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.cfm, map_location=dev)
    cfm = CFMYS(dim=ck["args"].get("dim", 384), spk_in=ck["args"].get("spk_in", False)).to(dev).eval()
    cfm.load_state_dict(ck["net"])
    abi = ck["abi"]
    mu, sd = abi["mu"].to(dev), abi["sd"].to(dev)

    ek = torch.load(ROOT / "results/diag_e2/diag_e2_best.pt", map_location=dev)
    enet = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"]).to(dev).eval()
    enet.load_state_dict(ek["net"])
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                         map_location="cpu", weights_only=False)

    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()

    w, _ = librosa.load(str(ROOT / "namikawa.mp3"), sr=44100, mono=True)
    x = torch.from_numpy(w) * 32768.0
    f0, _ = SF.causal_f0(x)
    ratio = 2.0 ** (a.semitones / 12)
    f0s = torch.where(f0 > 0, f0 * ratio, f0)

    # 条件列: content(mel80@44.1k の E 出力・172.27fps) + lf0 + en → 100fps 因果
    # 2026-09-19監査修正: E出力とSF.causal_f0はどちらも 44100/256 fps。
    # 旧コードは両方を50fps/86.13fps扱いし、content 3.44倍・f0 2倍の時間伸長と
    # T100上限6000による60s打ち切りを起こしていた。
    mel = SF.mel(x)
    with torch.no_grad():
        content = enet(mel.to(dev)[None])[0]                  # [768, Tm] @172.27fps
    MEL_FPS = 44100 / SF.HOP_A
    T100 = int(len(w) * 48000 / 44100) // HOP48
    idx_c = ((torch.arange(T100, dtype=torch.float64) + 1.0)
             * MEL_FPS / 100.0 - 1.0).floor().clamp(0, content.shape[-1] - 1).long()
    c = content[:, idx_c.to(dev)]
    hop = 512
    rms = torch.sqrt(((x[: len(w) // hop * hop] / 32768.0)
                      .reshape(-1, hop) ** 2).mean(-1) + 1e-12)
    i_f = ((torch.arange(T100, dtype=torch.float64) + 1.0)
           * MEL_FPS / 100.0 - 1.0).floor().clamp(0, f0s.shape[-1] - 1).long()
    lf0 = torch.log(f0s[i_f].clamp(min=50.0) / 200.0)
    i_e = ((torch.arange(T100, dtype=torch.float64) + 1.0)
           * F0_FPS / 100.0 - 1.0).floor().clamp(0, rms.shape[-1] - 1).long()
    enl = torch.log(rms[i_e].clamp(min=1e-4))
    cond = torch.cat([c, lf0[None].to(dev), enl[None].to(dev)], 0)[None]

    s_ = spk_emb["ab97e212acbb6d6b"][None].to(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        z0 = ar_noise(T100, 0.9, g, dev, 1)
        zh = z0
        for k in range(a.K):
            t = torch.full((1,), k / a.K, device=dev)
            zh = zh + cfm(zh, cond, t, s_) / a.K
        zh = zh.clamp(-8, 8)
        z = (zh * sd[:, None] + mu[:, None])       # [1,32,T] 実 latent 空間
        stream = codec.decoder.stream()
        outs = [stream.decode_step(z[:, :, i:i + 1]) for i in range(T100)]
        y = torch.cat(outs, -1)[0, 0].cpu().numpy()
    soundfile.write(OUT / a.out, np.clip(y, -1, 1), SR48)
    print(f"  {a.out}: {T100} latent frames -> {len(y)/SR48:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

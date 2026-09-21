"""対象発話の汎用レンダプローブ: s5_renderと同一経路でsource/target/.content源を差し替え可能。

content源: e1 = mel→E1診断出力(172.27fps)、cv = featキャッシュの真ContentVec(50fps)。
時刻契約は各々のnative gridから100fpsへ因果写像(学習cond_ofと同一式)。
mel_in学習モデル(ck["cli"]["mel_in"])はcausal mel80/8を条件末尾へ追加。

    CUDA_VISIBLE_DEVICES=0 uv run python render_probe.py \
        --wav ../female-dataset/<spk>/<stem>.wav --target ab97e212acbb6d6b \
        --content e1 --K 8 --out ../results/diag_cfm_audit/fem_e1.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
from causal_codec import CausalCodec, SAMPLE_RATE
from train_cfmys import CFMYS, ar_noise
from train_vc_e import E1

ROOT = Path(__file__).resolve().parent.parent
SR48 = 48000
MEL_FPS = 44100 / SF.HOP_A
F0_FPS_E = 44100 / 512
CV_FPS = 50.0
MEL_SCALE = 8.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--content", choices=["e1", "cv"], default="e1")
    ap.add_argument("--cv-feat", default=None,
                    help="content=cv時のfeat .pt(未指定なら女声feat規約から推定)")
    ap.add_argument("--semitones", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--cfm", default=str(ROOT / "results/s7_cfm_itp/s7_cfm_itp_best.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg-w", type=float, default=1.0,
                    help="speaker CFG外挿幅(1.0=無効。cond/uncondを同seedでSamplingし外挿)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.cfm, map_location=dev)
    mel_in = bool(ck.get("cli", {}).get("mel_in", False))
    cfm = CFMYS(dim=ck["args"].get("dim", 384),
                cin=850 if mel_in else 770,
                spk_in=ck["args"].get("spk_in", False)).to(dev).eval()
    cfm.load_state_dict(ck["net"])
    mu, sd = ck["abi"]["mu"].to(dev), ck["abi"]["sd"].to(dev)
    ek = torch.load(ROOT / "results/diag_e2/diag_e2_best.pt", map_location=dev)
    enet = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"]).to(dev).eval()
    enet.load_state_dict(ek["net"])
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                         map_location="cpu", weights_only=False)
    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()

    w, _ = librosa.load(a.wav, sr=44100, mono=True)
    x = torch.from_numpy(w) * 32768.0
    T100 = int(len(w) * 100 / 44100)
    mel_full = SF.mel(x).to(dev) if mel_in else None

    if a.content == "e1":
        with torch.no_grad():
            content = enet(SF.mel(x).to(dev)[None])[0]
        idx_c = ((torch.arange(T100, dtype=torch.float64) + 1.0)
                 * MEL_FPS / 100.0 - 1.0).floor().clamp(0, content.shape[-1] - 1).long()
        c = content[:, idx_c.to(dev)]
    else:
        fp = a.cv_feat
        if fp is None:
            spk_dir = Path(a.wav).parent.name
            fp = ROOT / "data/female_real_feat" / spk_dir / (Path(a.wav).stem + ".pt")
        d = torch.load(fp, map_location="cpu", weights_only=False)
        cv = d["content"].T.float().to(dev)
        idx_c = ((torch.arange(T100, dtype=torch.float64) + 1.0)
                 * CV_FPS / 100.0 - 1.0).floor().clamp(0, cv.shape[-1] - 1).long()
        c = cv[:, idx_c.to(dev)]

    f0, _ = SF.causal_f0(x)
    ratio = 2.0 ** (a.semitones / 12)
    f0s = torch.where(f0 > 0, f0 * ratio, f0)
    hop = 512
    rms = torch.sqrt(((x[: len(w) // hop * hop] / 32768.0)
                      .reshape(-1, hop) ** 2).mean(-1) + 1e-12)
    i_f = ((torch.arange(T100, dtype=torch.float64) + 1.0)
           * MEL_FPS / 100.0 - 1.0).floor().clamp(0, f0s.shape[-1] - 1).long()
    lf0 = torch.log(f0s[i_f].clamp(min=50.0) / 200.0)
    i_e = ((torch.arange(T100, dtype=torch.float64) + 1.0)
           * F0_FPS_E / 100.0 - 1.0).floor().clamp(0, rms.shape[-1] - 1).long()
    enl = torch.log(rms[i_e].clamp(min=1e-4))
    parts = [c, lf0[None].to(dev), enl[None].to(dev)]
    if mel_in:
        mel_med = int(ck.get("cli", {}).get("mel_med", 0))
        m = mel_full
        if mel_med:
            k = mel_med
            pad = k // 2
            mp = np.pad(m.cpu().numpy(), ((pad, pad), (0, 0)), mode="edge")
            m = torch.from_numpy(np.median(
                np.lib.stride_tricks.sliding_window_view(mp, k, axis=0),
                axis=-1).copy()).float().to(dev)
        idx_m = ((torch.arange(T100, dtype=torch.float64) + 1.0)
                 * MEL_FPS / 100.0 - 1.0).floor().clamp(0, m.shape[-1] - 1).long()
        parts.append((m[:, idx_m.to(dev)] / MEL_SCALE).clamp(-6.0, 6.0))
    cond = torch.cat(parts, 0)[None]

    s_ = spk_emb[a.target][None].to(dev)

    def sample(s_vec):
        g = torch.Generator(device=dev).manual_seed(a.seed)
        with torch.no_grad():
            zh = ar_noise(T100, a.rho, g, dev, 1)
            for k in range(a.K):
                t = torch.full((1,), k / a.K, device=dev)
                zh = zh + cfm(zh, cond, t, s_vec) / a.K
        return zh.clamp(-8, 8)

    zh = sample(s_)
    if a.cfg_w != 1.0:
        zh_u = sample(torch.zeros_like(s_))
        zh = (zh_u + a.cfg_w * (zh - zh_u)).clamp(-8, 8)
        z = zh * sd[:, None] + mu[:, None]
        stream = codec.decoder.stream()
        y = torch.cat([stream.decode_step(z[:, :, i:i + 1])
                       for i in range(T100)], -1)[0, 0].cpu().numpy()
    soundfile.write(a.out, np.clip(y, -1, 1), SR48)
    print(f"  {a.out}: {T100} frames -> {len(y)/SR48:.1f}s "
          f"(content={a.content}, st={a.semitones}, K={a.K}, mel_in={mel_in})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

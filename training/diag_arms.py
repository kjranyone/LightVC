"""耳棄却の帰属切り分け arm 生成（診断・tag diag_ear）。

    cs   : 入力 mel + 素 f0            -> V  （V+励起の健康診断）
    cs17 : 入力 mel + f0*2^(17/12)     -> V  （f0 シフトのみの効果）
    pyv  : G mel + f0*2^(17/12)        -> V  （Python 製品相当・hysteresis あり）

    uv run python diag_arms.py --out ../results/diag_ear
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
import v2f as V2
from eval_g1 import render
from rddsp_gpu import mel_to_linear
from train_vc_e import E1
from train_vc_g import G1, resample_to, F0_FPS

ROOT = Path(__file__).resolve().parent.parent


def render_nm(mel80: torch.Tensor, f0: torch.Tensor, vnet, W, n: int, dev,
              noise_mix: float) -> torch.Tensor:
    T = mel80.shape[-1]
    from eval_g1 import V_MEL_ADAPT
    mel_lin = W @ (mel80.to(W.device) - V_MEL_ADAPT)
    g = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(n, generator=g)
    phi, f0u = SF.phase_of(f0, n)
    imp = torch.zeros(n); cnt = torch.zeros(n); nyq = 44100 / 2
    for s0 in range(0, SF.KMAX, 32):
        kk = torch.arange(s0 + 1, min(s0 + 33, SF.KMAX + 1), dtype=torch.float32)[:, None]
        m_ = (kk * f0u[None] < nyq).float()
        imp += (torch.cos(kk * phi[None]) * m_).sum(0)
        cnt += m_.sum(0)
    exc = (1.0 - noise_mix) * (imp / cnt.clamp(min=1.0).sqrt()) + noise_mix * z
    E = SF.cstft(exc.to(W.device), SF.NFFT_S, SF.HOP_S)
    Ts = SF.n_frames(n)
    ml_syn = SF.to_frames(mel_lin, Ts)
    m = min(Ts, E.shape[-1])
    H = (ml_syn[:, :m] - SF.MEL_REF).exp()
    P = E[:, :m] * H
    feat = torch.cat([ml_syn[None, :, :m], P.real[None], P.imag[None],
                      torch.log(P.abs()[None] + 1e-5)], 0)[None]
    with torch.no_grad():
        o = vnet(feat)
    S = torch.complex(o[0, 0], o[0, 1])
    return SF.cistft(S, n).cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "namikawa.mp3"))
    ap.add_argument("--out", default=str(ROOT / "results/diag_ear"))
    ap.add_argument("--e", default=str(ROOT / "results/diag_e2/diag_e2_best.pt"))
    ap.add_argument("--g", default=str(ROOT / "results/diag_ciptB/diag_ciptB_best.pt"))
    ap.add_argument("--vtag", default="v2f_prior_20260819_013905")
    ap.add_argument("--shift", type=float, default=17.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ek = torch.load(a.e, map_location=dev)
    enet = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"]).to(dev).eval()
    enet.load_state_dict(ek["net"])
    gk = torch.load(a.g, map_location=dev)
    gnet = G1(dim=gk["args"]["dim"], layers=gk["args"]["L"]).to(dev).eval()
    gnet.load_state_dict(gk["net"])
    vk = torch.load(ROOT / f"results/{a.vtag}/{a.vtag}_best.pt", map_location=dev)
    vnet = V2.V2F(cin=4, ch=vk["args"]["ch"], layers=vk["args"]["L"],
                  norm=vk["args"]["norm"]).to(dev).eval()
    vnet.load_state_dict(vk["net"])
    W = mel_to_linear(dev, nbin=SF.NFFT_S // 2 + 1)

    import librosa, soundfile
    w, _ = librosa.load(a.inp, sr=44100, mono=True)
    x = torch.from_numpy(w) * 32768.0
    n = x.shape[-1]
    mel = SF.mel(x)
    f0, _ = SF.causal_f0(x)
    ratio = 2.0 ** (a.shift / 12.0)
    f0s = torch.where(f0 > 0, f0 * ratio, f0)

    y_cs = render(mel, f0, vnet, W, n, dev)
    soundfile.write(out / "cs.wav", y_cs.clamp(-1, 1).numpy(), 44100)
    y_cs17 = render(mel, f0s, vnet, W, n, dev)
    soundfile.write(out / "cs17.wav", y_cs17.clamp(-1, 1).numpy(), 44100)
    y_nz0 = render_nm(mel, f0s, vnet, W, n, dev, 0.0)
    soundfile.write(out / "cs17_nz0.wav", y_nz0.clamp(-1, 1).numpy(), 44100)
    y_nz10 = render_nm(mel, f0s, vnet, W, n, dev, 0.1)
    soundfile.write(out / "cs17_nz10.wav", y_nz10.clamp(-1, 1).numpy(), 44100)

    with torch.no_grad():
        content = enet(mel.to(dev)[None])[0]
        lf0 = torch.log(f0s.clamp(min=50.0) / 200.0).to(dev)
        hop = 512
        nfrm = n // hop
        rms = torch.sqrt(((x[: nfrm * hop] / 32768.0).reshape(nfrm, hop) ** 2).mean(-1) + 1e-12)
        en_seq = resample_to(rms[None], mel.shape[-1], F0_FPS)[0]
        en = torch.log(en_seq.clamp(min=1e-4)).to(dev)
        t = mel.shape[-1]
        feat = torch.cat([content[:, :t], lf0[None, :t], en[None, :t]], 0)[None]
        mel_g = gnet(feat)[0].cpu()
    y_pyv = render(mel_g, f0s, vnet, W, n, dev)
    soundfile.write(out / "pyv.wav", y_pyv.clamp(-1, 1).numpy(), 44100)
    print(f"  cs / cs17 / cs17_nz0 / cs17_nz10 / pyv 生成（shift {a.shift:+.1f}）-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

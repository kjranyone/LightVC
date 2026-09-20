"""Rust VcStream の parity 参照を吐く: 無音頭 1024 ＋ 男声を batch 変換。

    uv run python make_vc_parity.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
import v2f as V2
from train_vc_g import G1, load_wav, wav_path_of, resample_to, F0_FPS
from train_vc_e import E1
from rddsp_gpu import mel_to_linear

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--e", default="../results/diag_e1/diag_e1_best.pt")
    ap.add_argument("--g", default="../results/diag_cartA/diag_cartA_best.pt")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cpu"

    ek = torch.load(a.e, map_location=dev)
    enet = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"]).to(dev).eval()
    enet.load_state_dict(ek["net"])
    gk = torch.load(a.g, map_location=dev)
    gnet = G1(dim=gk["args"]["dim"], layers=gk["args"]["L"]).to(dev).eval()
    gnet.load_state_dict(gk["net"])
    vtag = Path("/tmp/current_tag").read_text().strip()
    vk = torch.load(ROOT / f"results/{vtag}/{vtag}_best.pt", map_location=dev)
    vnet = V2.V2F(cin=4, ch=vk["args"]["ch"], layers=vk["args"]["L"],
                  norm=vk["args"]["norm"]).to(dev).eval()
    vnet.load_state_dict(vk["net"])
    W = mel_to_linear(dev, nbin=SF.NFFT_S // 2 + 1)

    src = sorted((ROOT / "data/male_feat/male_p226").glob("*.pt"))[0]
    d = torch.load(src, map_location="cpu", weights_only=False)
    w = load_wav(wav_path_of(d)) * 32768.0
    x = torch.cat([torch.zeros(1024), w])
    n = (x.shape[-1] // 256) * 256
    x = x[:n]
    shift = 15.0
    ratio = 2.0 ** (shift / 12.0)

    mel = SF.mel(x)
    t = mel.shape[-1]
    f0, _ = SF.causal_f0(x)
    f0s = torch.where(f0 > 0, f0 * ratio, f0)
    with torch.no_grad():
        content = enet(mel[None])[0]
        lf0 = torch.log(f0s.clamp(min=50.0) / 200.0)[:t]
        nfrm = n // 512
        rms = torch.sqrt(((x[: nfrm * 512] / 32768.0).reshape(nfrm, 512) ** 2)
                         .mean(-1) + 1e-12)
        en = torch.log(resample_to(rms[None], t, F0_FPS)[0].clamp(min=1e-4))
        feat = torch.cat([content[:, :t], lf0[None, :t], en[None, :t]], 0)[None]
        mel_g = gnet(feat)[0]

    # render (eval_g1.render と同一・z を書き出す)
    g_ = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(n, generator=g_)
    phi, f0u = SF.phase_of(f0s, n)
    imp = torch.zeros(n); cnt = torch.zeros(n); nyq = 22050.0
    for s0 in range(0, SF.KMAX, 32):
        kk = torch.arange(s0 + 1, min(s0 + 33, SF.KMAX + 1), dtype=torch.float32)[:, None]
        m_ = (kk * f0u[None] < nyq).float()
        imp += (torch.cos(kk * phi[None]) * m_).sum(0)
        cnt += m_.sum(0)
    exc = 0.7 * (imp / cnt.clamp(min=1.0).sqrt()) + 0.3 * z
    exc.numpy().astype(__import__("numpy").float32).tofile(out / "vcp_exc.bin")
    f0s.numpy().astype(__import__("numpy").float32).tofile(out / "vcp_f0s.bin")
    SF._fill(f0s).numpy().astype(__import__("numpy").float32).tofile(out / "vcp_fill.bin")
    f0u.numpy().astype(__import__("numpy").float32).tofile(out / "vcp_f0u.bin")
    phi.numpy().astype(__import__("numpy").float32).tofile(out / "vcp_phi.bin")
    E = SF.cstft(exc, SF.NFFT_S, SF.HOP_S)
    Ts = SF.n_frames(n)
    from eval_g1 import V_MEL_ADAPT
    ml_syn = SF.to_frames(W @ (mel_g - V_MEL_ADAPT), Ts)  # 80-mel 空間で補正
    m = min(Ts, E.shape[-1])
    H = (ml_syn[:, :m] - SF.MEL_REF).exp()
    P = E[:, :m] * H
    fv = torch.cat([ml_syn[None, :, :m], P.real[None], P.imag[None],
                    torch.log(P.abs()[None] + 1e-5)], 0)[None]
    fv[0].numpy().astype(np.float32).tofile(out / "vcp_feat.bin")   # [4, NBIN, T]
    with torch.no_grad():
        o = vnet(fv)
    S = torch.complex(o[0, 0], o[0, 1])
    y = SF.cistft(S, n)

    mel_g.numpy().astype(np.float32).tofile(out / "vcp_melg.bin")
    lf0[:t].numpy().astype(np.float32).tofile(out / "vcp_lf0.bin")
    en[:t].numpy().astype(np.float32).tofile(out / "vcp_en.bin")
    content[:, :t].numpy().astype(np.float32).tofile(out / "vcp_content.bin")
    mel[:, :t].numpy().astype(np.float32).tofile(out / "vcp_mel.bin")
    x.numpy().astype(np.float32).tofile(out / "vcp_x.bin")
    z.numpy().astype(np.float32).tofile(out / "vcp_z.bin")
    y.numpy().astype(np.float32).tofile(out / "vcp_y.bin")
    (out / "vcp_meta.json").write_text(json.dumps(
        {"n": int(n), "shift": shift, "e": a.e, "g": a.g, "v": vtag}))
    print(f"  n={n} ({n/44100:.1f}s) shift +{shift:.0f}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

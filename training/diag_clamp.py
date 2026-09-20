"""G OOD 暴走の安価な緩和 arm（tag diag_clamp・固定定数＝発話統計なし）。

- lf0 を女声コーパス [p1, p99] にクランプしてから G へ
- G 出力 mel の上天井ビンを held 女声 mel の per-bin p99 にクランプしてから V へ

    uv run python diag_clamp.py --out ../results/diag_clamp
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
from train_vc_g import G1, load_item, feat_util, FEATS, resample_to, F0_FPS

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "namikawa.mp3"))
    ap.add_argument("--out", default=str(ROOT / "results/diag_clamp"))
    ap.add_argument("--e", default=str(ROOT / "results/diag_e2/diag_e2_best.pt"))
    ap.add_argument("--g", default=str(ROOT / "results/diag_ciptB/diag_ciptB_best.pt"))
    ap.add_argument("--vtag", default="v2f_prior_20260819_013905")
    ap.add_argument("--shift", type=float, default=17.0)
    ap.add_argument("--topbins", type=int, default=10)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for root in FEATS:
        for spk in sorted(root.iterdir()):
            if spk.is_dir():
                files += sorted(spk.glob("*.pt"))
    spk_all = sorted({f.parent.name for f in files})
    held = set(spk_all[-24:])
    ev = [f for f in files if f.parent.name in held][:24]
    mels = []
    for f in ev:
        _, mel = load_item(f)
        mels.append(mel)
    M = torch.cat(mels, -1)
    bin_p99 = M.quantile(0.99, dim=-1)
    lf0_all = []
    for f in ev[:8]:
        d = torch.load(f, map_location="cpu", weights_only=False)
        f0 = d["f0"]
        vv = f0[f0 > 50]
        if vv.numel():
            lf0_all.append(torch.log(vv / 200.0))
    L = torch.cat(lf0_all)
    lf0_lo, lf0_hi = float(L.quantile(0.01)), float(L.quantile(0.99))
    print(f"  corpus lf0 [{lf0_lo:.3f},{lf0_hi:.3f}] / top{a.topbins}bin p99 "
          + str([round(float(b), 1) for b in bin_p99[-a.topbins:]]))

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
    en_raw = torch.log(f0s.clamp(min=50.0) / 200.0)

    with torch.no_grad():
        content = enet(mel.to(dev)[None])[0]
        hop = 512
        nfrm = n // hop
        rms = torch.sqrt(((x[: nfrm * hop] / 32768.0).reshape(nfrm, hop) ** 2).mean(-1) + 1e-12)
        en_seq = resample_to(rms[None], mel.shape[-1], F0_FPS)[0]
        en = torch.log(en_seq.clamp(min=1e-4)).to(dev)
        t = mel.shape[-1]
        for name, lf0 in [("pyc", en_raw), ("cl", en_raw.clamp(lf0_lo, lf0_hi).to(dev))]:
            feat = torch.cat([content[:, :t], lf0[None, :t].to(dev), en[None, :t]], 0)[None]
            mel_g = gnet(feat)[0].cpu()
            if name == "cl":
                mel_g = mel_g.clone()
                mel_g[-a.topbins:] = torch.minimum(mel_g[-a.topbins:],
                                                   bin_p99[-a.topbins:][:, None])
            torch.save({"mel_g": mel_g, "f0": f0s}, out / f"cols_{name}.pt")
            y = render(mel_g, f0s, vnet, W, n, dev)
            soundfile.write(out / f"{name}.wav", y.clamp(-1, 1).numpy(), 44100)
            print(f"  {name}.wav 生成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

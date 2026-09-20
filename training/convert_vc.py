"""男声 → 萌え声の変換レンダ（R-X の実行系・v1 クライアントの参照実装）。

    mic/wav → front(mel80, f0) → E(mel80)→content → G(content, f0+shift)→mel80' → V → 音声

f0 シフトは**半音単位の固定ノブ**（発話統計は使わない・出荷ゲート準拠）。

    uv run python convert_vc.py --in ../namikawa.mp3 --shift 12 \\
        --e ../results/diag_e1/diag_e1_best.pt --g ../results/diag_g1/diag_g1_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
import v2f as V2
from train_vc_g import G1
from train_vc_e import E1
from eval_g1 import render
from rddsp_gpu import mel_to_linear

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="/tmp/vc_out.wav")
    ap.add_argument("--e", required=True)
    ap.add_argument("--g", required=True)
    ap.add_argument("--shift", type=float, default=12.0,
                    help="f0 シフト（半音）。固定ノブ＝発話統計なし")
    ap.add_argument("--energy-gain", type=float, default=1.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

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

    import librosa, soundfile
    w, _ = librosa.load(a.inp, sr=44100, mono=True)
    x = torch.from_numpy(w) * 32768.0
    n = x.shape[-1]

    mel = SF.mel(x)                                    # [80, Ta] 製品 front
    f0, _ = SF.causal_f0(x)
    ratio = 2.0 ** (a.shift / 12.0)
    f0s = torch.where(f0 > 0, f0 * ratio, f0)          # 有声のみシフト

    with torch.no_grad():
        content = enet(mel.to(dev)[None])[0]           # [768, Ta]
        # G の入力系列（train_vc_g.feat_util と同じ変換・ただし f0 はシフト後）
        lf0 = torch.log(f0s.clamp(min=50.0) / 200.0).to(dev)
        # energy は学習時の定義そのもの（HOP512 非重畳フレーム RMS）を
        # 因果リサンプルで mel グリッドへ——近似ではなく同一定義にする。
        hop = 512
        nfrm = n // hop
        rms = torch.sqrt(((x[: nfrm * hop] / 32768.0).reshape(nfrm, hop) ** 2).mean(-1) + 1e-12)
        from train_vc_g import resample_to, F0_FPS
        en_seq = resample_to(rms[None], mel.shape[-1], F0_FPS)[0]
        en = (torch.log(en_seq.clamp(min=1e-4)) * a.energy_gain).to(dev)
        t = mel.shape[-1]
        feat = torch.cat([content[:, :t], lf0[None, :t], en[None, :t]], 0)[None]
        mel_g = gnet(feat)[0].cpu()

    y = render(mel_g, f0s, vnet, W, n, dev)       # render 出力は [-1,1] スケール
    soundfile.write(a.out, y.clamp(-1, 1).numpy(), 44100)
    print(f"  {a.out} に書き出し（shift {a.shift:+.1f} 半音）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

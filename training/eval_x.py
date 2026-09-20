"""R-X の判定: 男声 → cartridge G → V の出力が目標話者に化けたか。

判定（凍結・vc_eg.md）: mean ECAPA SECS(変換音, 目標重心) >= 0.5。
参照系: 上限 = 目標話者の held 発話 vs 重心 / 下限 = 生の男声 vs 重心。
f0 シフトは**学習データの中央値から決める定数**（cartridge のノブ。走行時統計なし）。

    uv run python eval_x.py --g ../results/diag_cartA/diag_cartA_best.pt \
        --e ../results/diag_e1/diag_e1_best.pt --spk ab97e212acbb6d6b
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
import v2f as V2
from train_vc_g import G1, load_wav, wav_path_of, resample_to, F0_FPS
from train_vc_e import E1
from eval_g1 import render
from rddsp_gpu import mel_to_linear

ROOT = Path(__file__).resolve().parent.parent


def median_voiced_f0(feat_files) -> float:
    vs = []
    for f in feat_files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        v = d["f0"][d["f0"] > 50]
        if len(v):
            vs.append(v)
    return float(torch.cat(vs).median())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", required=True)
    ap.add_argument("--e", required=True)
    ap.add_argument("--spk", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--vtag", type=str, default=None)
    ap.add_argument("--dump", type=str, default=None,
                    help="変換 wav を書き出すディレクトリ（任意）")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ek = torch.load(a.e, map_location=dev)
    enet = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"],
              look=ek["args"].get("look", 0)).to(dev).eval()
    enet.load_state_dict(ek["net"])
    gk = torch.load(a.g, map_location=dev)
    from train_vc_g import GS
    gcls = GS if gk["args"].get("arch") == "gs" else G1
    gnet = gcls(dim=gk["args"]["dim"], layers=gk["args"]["L"]).to(dev).eval()
    gnet.load_state_dict(gk["net"])
    spk_map = (torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                          map_location="cpu", weights_only=False)
               if gcls is GS else None)
    vtag = a.vtag or Path("/tmp/current_tag").read_text().strip()
    vk = torch.load(ROOT / f"results/{vtag}/{vtag}_best.pt", map_location=dev)
    vnet = V2.V2F(cin=4, ch=vk["args"]["ch"], layers=vk["args"]["L"],
                  norm=vk["args"]["norm"]).to(dev).eval()
    vnet.load_state_dict(vk["net"])
    W = mel_to_linear(dev, nbin=SF.NFFT_S // 2 + 1)

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/sb_ecapa",
        run_opts={"device": "cpu"})
    import librosa

    def emb_of(w44: torch.Tensor) -> torch.Tensor:
        w16 = librosa.resample(w44.numpy(), orig_sr=44100, target_sr=16000)
        e = ecapa.encode_batch(torch.from_numpy(w16)[None]).squeeze()
        return e / e.norm()

    tf = sorted((ROOT / "data/female_tts_feat" / a.spk).glob("*.pt"))
    if not tf:
        sys.exit(f"目標話者 {a.spk} の feat が無い")
    cen_files, held_files = tf[:20], tf[-5:]
    cen = []
    for f in cen_files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        cen.append(emb_of(load_wav(wav_path_of(d))))
    cen = torch.stack(cen).mean(0)
    cen = cen / cen.norm()

    up = []
    for f in held_files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        up.append(float((emb_of(load_wav(wav_path_of(d))) * cen).sum()))

    tgt_f0 = median_voiced_f0(tf[:40])

    mroot = ROOT / "data/male_feat"
    spks = sorted(d for d in mroot.iterdir() if d.is_dir())[:4]
    srcs = []
    for sd in spks:
        srcs += sorted(sd.glob("*.pt"))[: max(1, a.n // len(spks))]
    srcs = srcs[: a.n]

    lo, cv = [], []
    if a.dump:
        Path(a.dump).mkdir(parents=True, exist_ok=True)
    for f in srcs:
        d = torch.load(f, map_location="cpu", weights_only=False)
        src_f0 = median_voiced_f0(sorted(f.parent.glob("*.pt"))[:20])
        shift = round(12.0 * math.log2(tgt_f0 / src_f0))
        x = load_wav(wav_path_of(d)) * 32768.0
        n = x.shape[-1]
        lo.append(float((emb_of(x / 32768.0) * cen).sum()))

        mel = SF.mel(x)
        f0, _ = SF.causal_f0(x)
        ratio = 2.0 ** (shift / 12.0)
        f0s = torch.where(f0 > 0, f0 * ratio, f0)
        with torch.no_grad():
            content = enet(mel.to(dev)[None])[0]
            lf0 = torch.log(f0s.clamp(min=50.0) / 200.0).to(dev)
            hop = 512
            nfrm = n // hop
            rms = torch.sqrt(((x[: nfrm * hop] / 32768.0).reshape(nfrm, hop) ** 2)
                             .mean(-1) + 1e-12)
            en_seq = resample_to(rms[None], mel.shape[-1], F0_FPS)[0]
            en = torch.log(en_seq.clamp(min=1e-4)).to(dev)
            t = mel.shape[-1]
            feat = torch.cat([content[:, :t], lf0[None, :t], en[None, :t]],
                             0)[None]
            s_ = (spk_map[a.spk][None].to(dev)
                  if spk_map is not None else None)
            mel_g = (gnet(feat, s_)[0].cpu() if s_ is not None
                     else gnet(feat)[0].cpu())
        y = render(mel_g, f0s, vnet, W, n, dev)   # render 出力は [-1,1] スケール
        cv.append(float((emb_of(y) * cen).sum()))
        print(f"  {f.parent.name}/{f.stem}: shift {shift:+d}  "
              f"src-SECS {lo[-1]:.3f} -> conv-SECS {cv[-1]:.3f}", flush=True)
        if a.dump:
            import soundfile
            soundfile.write(f"{a.dump}/{f.parent.name}_{f.stem}.wav",
                            y.clamp(-1, 1).numpy(), 44100)

    m_up = sum(up) / len(up)
    m_lo = sum(lo) / len(lo)
    m_cv = sum(cv) / len(cv)
    print(f"\n  上限（本人 held） {m_up:.3f}   下限（生男声） {m_lo:.3f}   "
          f"変換 {m_cv:.3f}（合格線 0.50）")
    print("  =>", "PASS" if m_cv >= 0.50 else "FAIL")
    return 0 if m_cv >= 0.50 else 1


if __name__ == "__main__":
    raise SystemExit(main())

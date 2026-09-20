"""R-G1 の判定: G の mel80 を V に通した PESQ を copy-synthesis と比べる。

判定（凍結・vc_eg.md）: PESQ(G→V) >= 0.8 × PESQ(copy-syn→V)。
copy-syn ＝ gt の mel80 をそのまま V に通す（V の天井）。

    uv run python eval_g1.py --ckpt ../results/diag_g1/diag_g1_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import math

import ship_front as SF
import v2f as V2

# V (v2f) は [-1,1] スケール波形で学習済み (train_gvoc.to_gpu が /32767)。
# E/G 系は x32768 慣習の mel を使うので、V 界面でこの定数を引く。
# 実測: copy-syn PESQ 1.751 -> 2.146 (-ln32768 補正) / 2.154 (完全再計算)。
V_MEL_ADAPT = math.log(32768.0)
from train_vc_g import G1, GS, load_item, feat_util
from rddsp_gpu import mel_to_linear
from rddsp_loop import score_one

ROOT = Path(__file__).resolve().parent.parent


def render(mel80: torch.Tensor, f0: torch.Tensor, vnet, W, n: int, dev) -> torch.Tensor:
    T = mel80.shape[-1]
    # V は [-1,1] 学習スケール。80-mel 空間で補正してから W 写像
    # (W の後で引くと行和 != 1 のぶん不等価。PESQ 検証は 80 空間版で 2.146)
    mel_lin = W @ (mel80.to(W.device) - V_MEL_ADAPT)
    g = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(n, generator=g)
    phi, f0u = SF.phase_of(f0, n)
    import math
    imp = torch.zeros(n); cnt = torch.zeros(n); nyq = 44100 / 2
    for s0 in range(0, SF.KMAX, 32):
        kk = torch.arange(s0 + 1, min(s0 + 33, SF.KMAX + 1), dtype=torch.float32)[:, None]
        m_ = (kk * f0u[None] < nyq).float()
        imp += (torch.cos(kk * phi[None]) * m_).sum(0)
        cnt += m_.sum(0)
    exc = 0.7 * (imp / cnt.clamp(min=1.0).sqrt()) + 0.3 * z
    E = SF.cstft(exc.to(W.device), SF.NFFT_S, SF.HOP_S)
    # 合成グリッドへ
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--e", type=str, default=None)
    ap.add_argument("--vtag", type=str, default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    gk = torch.load(a.ckpt, map_location=dev)
    gcls = GS if gk["args"].get("arch") == "gs" else G1
    gnet = gcls(dim=gk["args"]["dim"], layers=gk["args"]["L"]).to(dev)
    if gcls is GS:
        gnet.load_state_dict(gk["net"])
    else:
        gnet.load_state_dict(gk["net"])
    gnet.eval()
    spk_map = None
    if gcls is GS:
        spk_map = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                             map_location="cpu", weights_only=False)

    vtag = a.vtag or Path("/tmp/current_tag").read_text().strip()
    vk = torch.load(ROOT / f"results/{vtag}/{vtag}_best.pt", map_location=dev)
    vnet = V2.V2F(cin=4, ch=vk["args"]["ch"], layers=vk["args"]["L"],
                  norm=vk["args"]["norm"]).to(dev).eval()
    vnet.load_state_dict(vk["net"])
    W = mel_to_linear(dev, nbin=SF.NFFT_S // 2 + 1)

    enet = None
    if a.e:
        from train_vc_e import E1
        ek = torch.load(a.e, map_location=dev)
        enet = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"],
                  look=ek["args"].get("look", 0)).to(dev).eval()
        enet.load_state_dict(ek["net"])
        print(f"  content: E-student {a.e}（cos {ek.get('eval_cos', '?')}）", flush=True)

    # held-out（train_vc_g と同じ末尾 24 話者）
    from train_vc_g import FEATS
    files = []
    for root in FEATS:
        for spk in sorted(root.iterdir()):
            if spk.is_dir():
                files += sorted(spk.glob("*.pt"))
    spk_all = sorted({f.parent.name for f in files})
    held = set(spk_all[-24:])
    ev = [f for f in files if f.parent.name in held][:a.n]

    ps_g, ps_c = [], []
    for f in ev:
        d, mel = load_item(f)
        n = None
        import librosa
        wav_path = ROOT / str(d["path"]).lstrip("./")
        if not wav_path.exists():
            wav_path = Path(str(d["path"]))
        w, _ = librosa.load(str(wav_path), sr=44100, mono=True)
        gt = torch.from_numpy(w) * 32768.0
        gt_ref = torch.from_numpy(w)          # PESQ 参照は [-1,1] (render 出力と同スケール)
        n = gt.shape[-1]
        x = feat_util(d, mel).to(dev)[None]
        with torch.no_grad():
            if enet is not None:
                x[:, :768] = enet(mel.to(dev)[None])
            s_ = (spk_map[d["speaker"]][None].to(dev)
                  if spk_map is not None and d.get("speaker") in spk_map else None)
            mg = gnet(x, s_)[0].cpu() if s_ is not None else gnet(x)[0].cpu()
        # 実 f0（front と同じ検出器）
        f0, _ = SF.causal_f0(gt)
        y_g = render(mg, f0, vnet, W, n, dev)
        y_c = render(mel, f0, vnet, W, n, dev)
        ps_g.append(score_one(gt_ref, y_g))
        ps_c.append(score_one(gt_ref, y_c))
        print(f"  {f.parent.name[:10]}: G→V {ps_g[-1]:.3f} / copy-syn {ps_c[-1]:.3f}",
              flush=True)
    mg_, mc_ = sum(ps_g) / len(ps_g), sum(ps_c) / len(ps_c)
    ratio = mg_ / mc_
    print(f"\n  G→V {mg_:.4f}  copy-syn {mc_:.4f}  比 {ratio:.3f}（合格線 0.80）")
    print("  =>", "PASS" if ratio >= 0.80 else "FAIL")
    return 0 if ratio >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())

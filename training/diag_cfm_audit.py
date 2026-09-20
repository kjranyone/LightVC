"""S1-5 段階1接続監査: 既存CFM重みを学習規約どおりの条件(ContentVec@50fps,
f0/energy@44100/512)で検査する。レンダ経路(s5_render)のfps不整合と分離して、
学習済み写像自体の性質を測るのが目的。新規学習なし・計算枠10分。

走行前固定の判断:
  D1 z0種感覚: seed_spread/residual < 0.05 なら z0不感性 = 平均回帰退化を支持
     (z0⊥z1のL2最適解は E[z1|c]-z0。出力は E[z1|c] に潰れる)
  D2 lf0効力: delta(+12st) > 10*seed_spread かつ decode F0 中央値シフトが
     要求値±3半音以内なら条件効力あり
  D4 t依存性: 同一z0で t=0/0.5/1 の v差。退化していれば t は無視される

    CUDA_VISIBLE_DEVICES=0 uv run python diag_cfm_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE
from train_cfmys import CFMYS, LAT, ROOT, COND_FPS, LAT_FPS, F0_FPS, ar_noise

OUT = ROOT / "results/diag_cfm_audit"
K_SEEDS = 8
T_MIN, T_MAX = 300, 600


def cond_of(d: dict, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    c_full = d["content"].T.float()
    t_src = c_full.shape[-1]
    idx = ((torch.arange(T, dtype=torch.float64) + 1.0)
           * COND_FPS / LAT_FPS - 1.0).floor().clamp(0, t_src - 1).long()
    c = c_full[:, idx]
    f0 = d["f0"].float()
    en = d["energy"].float()
    i_f = ((torch.arange(T, dtype=torch.float64) + 1.0)
           * F0_FPS / LAT_FPS - 1.0).floor().clamp(0, f0.shape[0] - 1).long()
    lf0 = torch.log(f0[i_f].clamp(min=50.0) / 200.0)
    enl = torch.log(en[i_f].clamp(min=1e-4))
    return torch.cat([c, lf0[None], enl[None]], 0), i_f


def shift_lf0(d: dict, i_f: torch.Tensor, semitones: float) -> torch.Tensor:
    f0 = d["f0"].float()
    r = 2.0 ** (semitones / 12.0)
    f0s = torch.where(f0 > 0, f0 * r, f0)
    return torch.log(f0s[i_f].clamp(min=50.0) / 200.0)


def gen_zh(net, z0, cond, s_, dev, K: int = 1) -> torch.Tensor:
    with torch.no_grad():
        if K <= 1:
            v = net(z0, cond, torch.ones(1, device=dev), s_)
            return (z0 + v).clamp(-8, 8)
        z = z0
        for k in range(K):
            t = torch.full((1,), k / K, device=dev)
            z = z + net(z, cond, t, s_) / K
        return z.clamp(-8, 8)


def z0_for(T: int, seed: int, dev) -> torch.Tensor:
    g = torch.Generator(device=dev).manual_seed(seed)
    return ar_noise(T, 0.9, g, dev, 1)


def render(codec, z_real: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        stream = codec.decoder.stream()
        y = torch.cat([stream.decode_step(z_real[:, :, i:i + 1])
                       for i in range(z_real.shape[-1])], -1)[0, 0].cpu().numpy()
    return np.clip(y, -1, 1)


def decode_f0(y: np.ndarray) -> dict:
    w44 = librosa.resample(y.astype(np.float64), orig_sr=SAMPLE_RATE,
                           target_sr=44100)
    f0, t = pyworld.harvest(w44, 44100, f0_floor=65, f0_ceil=1000,
                            frame_period=512 / 44100 * 1000)
    f0 = pyworld.stonemask(w44, f0, t, 44100)
    v = f0[f0 > 60.0]
    return {"f0_median": float(np.median(v)) if len(v) else 0.0,
            "voiced_ratio": float(len(v) / max(len(f0), 1))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results/s5_cfm/s5_cfm_best.pt"))
    ap.add_argument("--f0fix", action="store_true",
                    help="条件のf0を data/female_real_f0fix の再計算値で置換")
    ap.add_argument("--K", type=int, default=1, help="サンプリングEulerステップ数")
    ap.add_argument("--out", default="audit.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    abi = torch.load(LAT / "abi.pt", map_location=dev, weights_only=False)
    mu, sd = abi["mu"].to(dev), abi["sd"].to(dev)
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                         map_location="cpu", weights_only=False)

    feats, lats = [], {}
    for cname in ("female_real_feat", "female_tts_feat"):
        root = ROOT / "data" / cname
        for spk in sorted(root.iterdir()):
            if spk.is_dir():
                feats += sorted(spk.glob("*.pt"))
    for p in LAT.rglob("*.pt"):
        if p.name != "abi.pt":
            lats[p.stem] = p
    pairs = [f for f in feats if f.stem in lats and "female_real" in str(lats[f.stem])]
    spk_all = sorted({f.parent.name for f in pairs})
    held = set(spk_all[-24:])
    ev = [f for f in pairs if f.parent.name in held]
    utts = []
    for f in ev:
        z = torch.load(lats[f.stem], map_location="cpu", weights_only=False)["z"]
        if T_MIN <= z.shape[0] <= T_MAX:
            utts.append((f, z.float()))
        if len(utts) == 3:
            break
    print(f"  held utts: {[f.stem for f, _ in utts]}", flush=True)

    ck = torch.load(a.ckpt, map_location=dev)
    if "abi" in ck:
        mu, sd = ck["abi"]["mu"].to(dev), ck["abi"]["sd"].to(dev)
    prefix = Path(a.ckpt).parent.name + "_"
    net = CFMYS(dim=ck["args"].get("dim", 384), spk_in=ck["args"].get("spk_in", False)).to(dev).eval()
    net.load_state_dict(ck["net"])
    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()

    rep: dict = {"ckpt": a.ckpt, "f0fix": a.f0fix, "K": a.K,
                 "ck_step": ck.get("step"),
                 "utts": [], "D4": {}, "D2": {}, "D3": []}
    spread_all: list[float] = []

    for ui, (f, z1) in enumerate(utts):
        d = torch.load(f, map_location="cpu", weights_only=False)
        if a.f0fix:
            ff = ROOT / "data/female_real_f0fix" / f.parent.name / f.name
            if ff.exists():
                d = {**d, "f0": torch.load(ff, map_location="cpu",
                                           weights_only=False)["f0"]}
        T = z1.shape[0]
        cond, i_f = cond_of(d, T)
        cond = cond.to(dev)[None]
        s_ = spk_emb.get(d.get("speaker"))
        s_ = s_[None].to(dev) if s_ is not None else None
        z1n = ((z1 - mu.cpu()) / sd.cpu()).to(dev)

        zhs = [gen_zh(net, z0_for(T, 1000 + k, dev), cond, s_, dev, a.K)
               for k in range(K_SEEDS)]
        zh_bar = torch.stack(zhs).mean(0)
        spread = torch.stack([(zh - zh_bar).abs().mean(-1).mean(-1)
                              for zh in zhs]).mean() * sd.mean()
        residual = (zh_bar[0].transpose(0, 1) - z1n).abs().mean() * sd.mean()
        spread_all.append(float(spread))
        rep["utts"].append({
            "stem": f.stem, "speaker": d.get("speaker"), "T": T,
            "seed_spread": float(spread), "residual": float(residual),
            "ratio": float(spread / residual.clamp(min=1e-9)),
            "real_L1": float(((zh_bar * sd[:, None] + mu[:, None])[0]
                              .transpose(0, 1) - z1.to(dev)).abs().mean()),
        })
        print(f"  [{f.stem}] spread {float(spread):.4f} residual {float(residual):.4f}"
              f" ratio {float(spread / residual.clamp(min=1e-9)):.4f}", flush=True)

        if ui == 0:
            base = zhs[0]
            with torch.no_grad():
                z0 = z0_for(T, 1000, dev)
                vs = [net(z0, cond, torch.full((1,), t, device=dev), s_)
                      for t in (0.0, 0.5, 1.0)]
            rep["D4"] = {
                "v_max_abs_diff_0_vs_1": float((vs[0] - vs[2]).abs().max()),
                "v_mean_abs_diff_0_vs_1": float((vs[0] - vs[2]).abs().mean()),
                "v_mean_abs_diff_05_vs_1": float((vs[1] - vs[2]).abs().mean()),
            }

            for st_key, st in (("shift7", 7.0), ("shift12", 12.0)):
                cond_s = cond.clone()
                cond_s[0, -2] = shift_lf0(d, i_f, st).to(dev)
                zh_s = gen_zh(net, z0_for(T, 1000, dev), cond_s, s_, dev, a.K)
                delta = (zh_s - base).abs().mean(-1).mean(-1) * sd.mean()
                rep["D2"][st_key] = {
                    "delta": float(delta),
                    "delta_over_spread": float(delta / spread.clamp(min=1e-9)),
                }

            cond12 = cond.clone()
            cond12[0, -2] = shift_lf0(d, i_f, 12.0).to(dev)
            zh12 = gen_zh(net, z0_for(T, 1000, dev), cond12, s_, dev, a.K)
            for tag, zh in (("base", base), ("st12", zh12)):
                z_real = zh * sd[:, None] + mu[:, None]
                y = render(codec, z_real)
                soundfile.write(OUT / f"{prefix}{f.stem}_{tag}.wav", y,
                                SAMPLE_RATE)
                rep["D2"][f"decode_{tag}"] = decode_f0(y)

            zh0 = gen_zh(net, z0_for(T, 1000, dev), cond,
                         torch.zeros(1, 192, device=dev), dev, a.K)
            rep["D3"] = [{
                "delta_speaker_zero": float(
                    (zh0 - base).abs().mean(-1).mean(-1) * sd.mean()),
            }]

    rep["D1_spread_mean"] = float(np.mean(spread_all))
    rep["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / a.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

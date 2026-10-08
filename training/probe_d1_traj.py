"""F6機構の潜在レベル直接検査(0学習・2026-09-23)。

F6: コーラス=フレーム毎独立サンプリングの軌道非整合。予測: 並列CFM標本はGTより
フレーム差分エネルギーが大きく lag-1 自己相関が低い(ジッター)・AR標本はGTに近い。
統計(正規化潜在空間・N話者×3seed・T≤600):
  jit_ratio = mean||z_t−z_{t−1}||² / 同GT
  ac1       = 次元平均 lag-1 自己相関(標本) と GT の値
  seed_spread = seed間の標本距離 / GTとの距離(収縮・退化の検出)

    CUDA_VISIBLE_DEVICES=0 uv run python probe_d1_traj.py --arms d1_g0,d1_g0par
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d1_model import sample_frame_ar
from train_d1 import build_index, cond_of
from train_cfmys import LAT, F0FIX, sample_k
from eval_d1_g0 import load_arm, pick_utts

ROOT = Path(__file__).resolve().parent.parent


def jit(z: torch.Tensor) -> float:
    return float((z[:, 1:] - z[:, :-1]).pow(2).sum(0).mean())


def ac1(z: torch.Tensor) -> float:
    x = z - z.mean(1, keepdim=True)
    num = (x[:, 1:] * x[:, :-1]).sum(1)
    den = x.pow(2).sum(1).clamp(min=1e-9)
    return float((num / den).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--ckpts", default="",
                    help="腕ごとのbest/last(カンマ区切り・既定: cfmys=best, d1=last)")
    ap.add_argument("--n-spk", type=int, default=6)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/d1_g0/latent_traj_stats.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in a.seeds.split(",")]
    abi = torch.load(LAT / "abi.pt", map_location=dev, weights_only=False)
    MU, SD = abi["mu"].to(dev), abi["sd"].to(dev)
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt", map_location="cpu",
                         weights_only=False)
    pairs, lats, held_spk = build_index(0)
    utts = pick_utts(pairs, lats, held_spk, "train", a.n_spk)
    mel_cache: dict = {}

    def mel_of(f, d):
        if f not in mel_cache:
            import librosa
            from causal_mel import causal_mel
            wv, _ = librosa.load(d["path"], sr=44100, mono=True)
            mel_cache[f] = causal_mel(torch.from_numpy(wv) * 32768.0, n_fft=1024, hop=256,
                                      num_mels=80, sr=44100)[0].half()
        return mel_cache[f]

    items = []
    for f in utts:
        d = torch.load(f, map_location="cpu", weights_only=False)
        d = {**d, "f0": torch.load(F0FIX / f.parent.name / f.name, map_location="cpu",
                                   weights_only=False)["f0"]}
        z = torch.load(lats[f.stem], map_location="cpu", weights_only=False)["z"].float().T
        T = min(z.shape[1], 600)
        zn = ((z[:, :T].to(dev) - MU[:, None]) / SD[:, None]).clamp(-8, 8)
        items.append((f, d, T, zn))

    arms = a.arms.split(",")
    whs = a.ckpts.split(",") if a.ckpts else [""] * len(arms)
    out = {"gt": {"jit": float(np.mean([jit(it[3]) for it in items])),
                  "ac1": float(np.mean([ac1(it[3]) for it in items]))},
           "utts": [it[0].stem for it in items], "arms": {}}
    for tag, wh in zip(arms, whs):
        probe_wh = wh or ("best" if "cfm" in tag else "last")
        kind, net, ck = load_arm(tag, dev, probe_wh)
        cli = ck["cli"]
        jr, a1, spread = [], [], []
        for f, d, T, zn in items:
            use_mel = cli.get("mel80") if kind == "d1" else cli.get("mel_in")
            cond = cond_of(d, T, mel_of(f, d) if use_mel else None)[None].to(dev)
            zs = []
            for s in seeds:
                with torch.no_grad():
                    if kind == "d1":
                        zh = sample_frame_ar(net, cond, K=8, seed=s,
                                             z0_rho=float(cli.get("z0_rho", 0)))[0]
                    else:
                        s_ = None if cli.get("no_spk") else spk_emb.get(d.get("speaker"))
                        s_ = s_[None].to(dev) if s_ is not None else None
                        g = torch.Generator(device=dev).manual_seed(s)
                        mu, sd = ck["abi"]["mu"].to(dev), ck["abi"]["sd"].to(dev)
                        zr = sample_k(net, T, cli.get("rho", 0.9), g, dev, cond, s_, 8)[0]
                        zr = zr * sd[:, None] + mu[:, None]
                        zh = ((zr - MU[:, None]) / SD[:, None]).clamp(-8, 8)
                zs.append(zh)
                jr.append(jit(zh) / jit(zn))
                a1.append(ac1(zh))
            d_gt = float(np.mean([(z_ - zn).pow(2).mean().sqrt().item() for z_ in zs]))
            d_ss = float(np.mean([(zs[i] - zs[j]).pow(2).mean().sqrt().item()
                                  for i in range(len(zs)) for j in range(i + 1, len(zs))]))
            spread.append(d_ss / max(d_gt, 1e-6))
        out["arms"][tag] = {"ckpt": f"{probe_wh}@{int(ck['step'])}",
                            "jit_ratio_median": round(float(np.median(jr)), 3),
                            "jit_ratio_range": [round(float(min(jr)), 3), round(float(max(jr)), 3)],
                            "ac1_median": round(float(np.median(a1)), 3),
                            "seed_spread_over_gt_dist": round(float(np.median(spread)), 3)}
        print(tag, out["arms"][tag], flush=True)
    out["gt"] = {k: round(v, 3) for k, v in out["gt"].items()}
    p = ROOT / a.out
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("gt", out["gt"], "->", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

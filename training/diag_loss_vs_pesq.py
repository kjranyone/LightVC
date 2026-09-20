"""損失と PESQ が同じ方向を向いているかを測る（帰属の分岐点）。

200000 step の走行で V は一度も prior を超えなかった（PESQ 1.5552 -> 1.0760 -> 1.1576）。
候補は 3 つ: ①容量不足 ②損失と PESQ の不整合 ③加算複素残差の設計。

**②かどうかは 1 回の前向き計算で決まる**——学習済みネットと「残差ゼロ（＝prior そのもの）」を
同じ発話で並べ、損失と PESQ を両方出す。

  prior のほうが損失が高いのに PESQ も高い -> **②**（最適化は正しく、目的関数が耳と別方向）
  prior のほうが損失が低い                 -> 最適化が壊れている（②ではない）

    uv run python diag_loss_vs_pesq.py --ckpt ../results/<TAG>/<TAG>_best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_gvoc as TG
import ship_front as SF
import v1d as V
from rddsp_gpu import WavLMConvLoss, mrstft, mel_to_linear, safe_score, build as build_small


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.ckpt, map_location=dev)
    ar = ck["args"]
    net = V.V1D(dim=ar["dim"], L=ar["L"], k_in=ar["k_in"], k=ar["k"]).to(dev)
    net.load_state_dict(ck["net"])
    net.eval()

    TG.SHIP = True
    TG.PRIOR_FN = TG.ship_prior
    TG.FEATS = TG.v1d_feats
    W = mel_to_linear(dev, nbin=net.nbin)
    _, te_small = build_small(80, a.n)
    items = [TG.to_gpu_pre(dict(w=x["gt"].to(dev), mel=x["mel"].to(dev),
                                f0=x["f0"], _pre=True), dev, W) for x in te_small]

    lmos = WavLMConvLoss(dev)
    g = torch.Generator(device=dev).manual_seed(999)
    rows = []
    with torch.no_grad():
        for it in items:
            n = it["w"].shape[-1]
            pw = TG.ship_prior(it, g, dev)
            it["_P"] = pw
            f, P = TG.v1d_feats(it)
            S_net = net.apply(f[None], P[None])[0]
            for name, S in (("prior", P), ("net", S_net)):
                y = SF.cistft(S, n)
                gt = it["w"]
                ys, ts = y[None, TG.HOP * 2:-TG.HOP * 2], gt[None, TG.HOP * 2:-TG.HOP * 2]
                m = float(mrstft(ys, ts))
                lm = float(lmos(ys, ts))
                S2 = SF.cstft(y, SF.NFFT_S, SF.HOP_S)
                mm = min(S2.shape[-1], S.shape[-1])
                cons = float(((S2[..., :mm] - S[..., :mm]).abs().pow(2).sum().sqrt()
                              / S[..., :mm].abs().pow(2).sum().sqrt().clamp(min=1e-8)))
                pq = safe_score(gt.cpu(), y.cpu())
                rows.append({"arm": name, "mrstft": m, "wavlm": lm,
                             "consist": cons, "total": m + 100.0 * lm + cons,
                             "pesq": pq})

    out = {}
    for arm in ("prior", "net"):
        r = [x for x in rows if x["arm"] == arm]
        out[arm] = {k: float(np.mean([x[k] for x in r]))
                    for k in ("mrstft", "wavlm", "consist", "total", "pesq")}
    print(f"{'arm':6s} {'mrstft':>9s} {'wavlm':>9s} {'consist':>9s} "
          f"{'total':>10s} {'PESQ':>7s}")
    for arm in ("prior", "net"):
        o = out[arm]
        print(f"{arm:6s} {o['mrstft']:9.4f} {o['wavlm']:9.5f} {o['consist']:9.4f} "
              f"{o['total']:10.4f} {o['pesq']:7.4f}")
    dl = out["net"]["total"] - out["prior"]["total"]
    dp = out["net"]["pesq"] - out["prior"]["pesq"]
    print(f"\n  net − prior:  損失 {dl:+.4f}   PESQ {dp:+.4f}")
    if dl < 0 and dp < 0:
        print("  => **②損失と PESQ の不整合**。最適化は正しく効いている"
              "（損失は下がった）のに耳の代理指標は悪化した。目的関数が犯人。")
    elif dl > 0:
        print("  => 最適化が効いていない（学習済みのほうが損失が高い）。②ではない。")
    else:
        print("  => 損失も PESQ も改善。この発話集合では矛盾なし。")
    Path("../results/z0/diag_loss_vs_pesq.json").write_text(
        json.dumps({"per_arm": out, "d_total": dl, "d_pesq": dp,
                    "ckpt": a.ckpt, "n": a.n}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

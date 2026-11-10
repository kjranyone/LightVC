"""段 I7 の切り分け(paper_snv 系 2h (iv)): M10 の声どうしで平均した勾配のうち、声が自分の再現から受ける割合(自己の分)を学習と同じ束で測る。
束 = sample_grouped(16, 4)・対応 = 保存した型への帯の対応(δ 0.1)。フレームごとに 自己の分 = 同じ (組, 位置) に対応づけた自分のフレームの数 / 全員のフレームの数。
名目は 1 / K。位置を自分しか通らないフレームは自己の分 1(声ごとの straight-through と同じ押し)。
    cd training && uv run --no-sync python snv_diag/i7_self_share.py ckpt=results/snv_i7_m10/step10k.pt n=50 out=results/snv_i7/self_share_10k.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train_snv_i as I
import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    kw = dict(a.split("=", 1) for a in sys.argv[1:])
    ck = ROOT / kw.get("ckpt", "results/snv_i7_m10/step10k.pt")
    n = int(kw.get("n", "50"))
    out = ROOT / kw.get("out", "results/snv_i7/self_share_10k.json")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(ck, map_location="cpu", weights_only=False)
    tr = json.loads((ck.parent / "split.json").read_text())["train"]
    ctr = I.Corpus(tr)
    E = P11.EncS().to(dev)
    E.load_state_dict(st["Es"])
    E.eval()
    tpl = st["tpl"].float().to(dev)
    tlen = torch.from_numpy(ctr.tlen).to(dev)
    band = float(st.get("band", 0.1))
    K = int(st.get("group", 4))
    bt = I.Batches(ctr, 777)
    own, single, nvo, gsz, kinds = [], [], [], [], []
    with torch.no_grad():
        for _ in range(n):
            b = {k: x.to(dev) for k, x in bt.sample_grouped(16, K).items()}
            m = b["ok"].clone()
            m[:, :T.WARM_SELF] = 0
            md = m * (b["cid"] >= 0).float()[:, None]
            sE = E.structure(P11.xin(b["ry"], b["rv"]), b["rmask"])
            mu, _ = E(P11.xin(b["y"], b["v"]), sE)
            _, _, tau = I.share_path(mu[:, :I.D_CT], m, b["cid"], tpl, tlen, b["off"], b["nall"], False, band=band, ret_tau=True)
            valid = (md > 0) & (tau >= 0)
            Lm = tpl.shape[1]
            key = (b["gid"][:, None] * Lm + tau.clamp(min=0))
            rk = torch.arange(len(key), device=dev)[:, None] * (int(b["gid"].max()) + 1) * Lm + key
            kv, rv = key[valid], rk[valid]
            _, ki, kc = torch.unique(kv, return_inverse=True, return_counts=True)
            _, ri, rc = torch.unique(rv, return_inverse=True, return_counts=True)
            sh = rc[ri].float() / kc[ki].float()
            own.append(sh.cpu().numpy())
            single.append((sh == 1.0).float().cpu().numpy())
            rows_at = torch.zeros(kc.shape[0], device=dev).index_add_(0, ki, 1.0 / rc[ri].float())
            nvo.append(rows_at[ki].cpu().numpy())
            gsz += torch.bincount(b["gid"]).tolist()
    own_, single_, nvo_ = np.concatenate(own), np.concatenate(single), np.concatenate(nvo)
    res = {"ckpt": str(ck.relative_to(ROOT)), "batches": n, "K_nominal": K, "band": band,
           "group_size_hist": {str(k): int(v) for k, v in zip(*np.unique(gsz, return_counts=True))},
           "own_share_mean": round(float(own_.mean()), 4), "own_share_median": round(float(np.median(own_)), 4),
           "frac_frames_own_only": round(float(single_.mean()), 4), "voices_at_key_mean": round(float(nvo_.mean()), 3),
           "nominal_1_over_K": round(1.0 / K, 4), "frames": int(len(own_))}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

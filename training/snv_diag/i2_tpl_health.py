"""段 I のやり直し(snv_i2_m3e)の型の健全さ(paper_snv §5 M3 rev3・学習なし): 保存した型 μ_w・Σ で、学習の声の発話を対応づけたとき、型は演奏をどれだけ説明し、型からのずれのどれだけが声に一貫するか。
内容の種類ごと(文 = E1 / 段 D の台本・音素の無い発声 = E1-NV の(種類・説明文の番号))に分けて、
R² = 型が事後の平均の分散を説明する割合・幅 = 型からのずれの二乗平均 / Σ(次元の平均)・代価 = Σ_d ずれ² / 2σ_d²(フレームあたり・KL の平均の項)・
声の一貫 = 発話ごとのずれの平均の声の間の分散 / ずれのフレームの分散(文の内容だけ・同じ内容を発した声の間)。演奏 = E(構造 = 同じ声の別の普通の発話 3 本 × 最大 10 s)・4 フレームの平均・冒頭 WARM_SELF を外す。
    cd training && uv run --no-sync python snv_diag/i2_tpl_health.py results/snv_i2_m3e/step10k.pt [n_txt=40] [n_nv=20] [out=results/snv_i2/tpl_health_10k.json]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_snv_i as I
import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = I.ROOT


def main() -> int:
    ck = [x for x in sys.argv[1:] if "=" not in x][0]
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    n_txt, n_nv = int(opt.get("n_txt", 40)), int(opt.get("n_nv", 20))
    torch.set_num_threads(8)
    st = torch.load(ROOT / ck, map_location="cpu", weights_only=False)
    E = P11.EncS().eval()
    E.load_state_dict(st["Es"])
    tpl, npos, logsig = st["tpl"].float(), st["npos"], st["logsig"].float()
    sig2 = (2 * logsig).exp()
    names = sorted(json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"], key=lambda v: v["voice"])
    tr, _ = T.split([v["voice"] for v in names])
    c = I.Corpus(tr)
    by: dict = {}
    for n in tr:
        for kind in ("norm", "nv"):
            for u in c.u[n][kind]:
                if u["cid"] >= 0:
                    by.setdefault(u["cid"], []).append((n, u))
    rng = random.Random(0)
    keys = list(c.cid)
    txt = sorted(k for k, v in by.items() if keys[k][0] == "txt" and len({n for n, _ in v}) >= 6)
    nv = sorted(k for k in by if keys[k][0] == "nv")
    rng.shuffle(txt), rng.shuffle(nv)

    def refs_of(n: str, skip: dict) -> tuple:
        rs = [u for u in c.u[n]["norm"] if u is not skip][:I.REF_N]
        ry = np.concatenate([u["y"][:, :I.REF_LEN] for u in rs], 1)
        rv = np.concatenate([u["v"][:I.REF_LEN] for u in rs])
        return torch.from_numpy(ry)[None], torch.from_numpy(rv)[None]

    @torch.no_grad()
    def resid(cid: int, n: str, u: dict) -> tuple | None:
        ry, rv = refs_of(n, u)
        s = E.structure(P11.xin(ry, rv), torch.ones(rv.shape, dtype=torch.bool))
        mu = E(P11.xin(torch.from_numpy(u["y"])[None], torch.from_numpy(u["v"])[None]), s)[0][0]
        L = int(c.tlen[cid])
        x = mu[:, T.WARM_SELF:]
        tn = x.shape[1] // I.POOL
        if tn < 2:
            return None
        mp = x[:, :tn * I.POOL].reshape(x.shape[0], tn, I.POOL).mean(-1)
        tp = tpl[cid, :L]
        lp = -0.5 * (((mp.T[:, None, :] - tp[None]) ** 2) / sig2).sum(-1)
        path = I.mas_free(lp[None], torch.tensor([tn]), torch.tensor([L]))[0]
        return mp.T.numpy(), tp[path].numpy()

    out: dict = {"ckpt": ck, "step": st.get("step"), "sigma_mean": float(logsig.exp().mean()), "sigma_min_max": [float(logsig.exp().min()), float(logsig.exp().max())], "kinds": {}}
    for kind, ks, cap in (("txt", txt[:n_txt], 8), ("nv", nv[:n_nv], 8)):
        U, P, per_voice, within = [], [], [], []
        for k in ks:
            items = by[k][:]
            rng.shuffle(items)
            seen, rows = set(), []
            for n, u in items:
                if n in seen:
                    continue
                seen.add(n)
                r = resid(k, n, u)
                if r is not None:
                    rows.append(r)
                if len(rows) >= cap:
                    break
            for m_, p_ in rows:
                U.append(m_), P.append(p_)
            if len(rows) >= 2:
                means = np.stack([(m_ - p_).mean(0) for m_, p_ in rows])
                per_voice.append(means.var(0).mean())
                within.append(np.mean([(m_ - p_).var(0).mean() for m_, p_ in rows]))
        U_, P_ = np.concatenate(U), np.concatenate(P)
        R = U_ - P_
        s2 = sig2.numpy()
        out["kinds"][kind] = {
            "contents": len(ks), "frames_pooled": int(len(U_)),
            "R2": round(float(1 - (R ** 2).sum() / ((U_ - U_.mean(0)) ** 2).sum()), 4),
            "resid_ms_over_sigma2": round(float(((R ** 2).mean(0) / s2).mean()), 4),
            "kl_mean_term_nat_per_frame": round(float(((R ** 2) / (2 * s2)).sum(1).mean()), 3),
            "voice_consistent_over_within": round(float(np.mean(per_voice) / max(np.mean(within), 1e-12)), 4) if per_voice else None,
            "u_var_mean": round(float(U_.var(0).mean()), 4),
        }
        print(kind, json.dumps(out["kinds"][kind]), flush=True)
    dst = ROOT / opt.get("out", "results/snv_i2/tpl_health_10k.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", dst, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

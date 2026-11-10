"""段 I の H_I_none(results/snv_i/STOP_10k.txt)の切り分け(paper_snv §6 段 I・学習なし): 演奏 u の中で声はどんな形で残り、内容の共有の事前分布にいくら払っているか。
A0F の 24 声: 演奏 = その声の E(構造 = その声の参照 j 0〜2)の平均 u。
  一次 = 声ごとの u の平均 m_v(参照 j 0〜2 の全フレーム)の声の間の散らばりを事前分布の幅 σ(学習の記録の平均)で測る・KL の代価 = Σ_d (m_v − m̄)² / 2σ²(フレームあたり)・
         発話 j 3 の u の平均から最も近い m_v の声の率(24 声)。
  二次 = 発話ごとに平均を引いたフレームで、声ごとの正規(参照 j 0〜2・縮小した共分散)の対数尤度の和の 1 位の率(QDA)。
  変換 = P3 の切り分けと同じ女声 → 別の女声(72 組・k = 0)で、u をそのまま / u − m_u + m_v(登録の定数のずらし)にして C → rvoc3hi → ECAPA + LDA の 24 声中の 1 位(M_FF・src_FF)。
    cd training && uv run --no-sync python snv_diag/i10_voice_code.py snv_i_m3@step10k [sigma=0.926] [out=results/snv_i/voice_code_10k.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nvoc as N
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV


def qda_fit(x: np.ndarray, shrink: float = 0.1) -> tuple[np.ndarray, np.ndarray, float]:
    mu = x.mean(0)
    c = np.cov(x.T)
    c = (1 - shrink) * c + shrink * np.trace(c) / c.shape[0] * np.eye(c.shape[0])
    ci = np.linalg.inv(c)
    return mu, ci, float(np.linalg.slogdet(c)[1])


def qda_ll(x: np.ndarray, p: tuple) -> float:
    mu, ci, ld = p
    d = x - mu
    return float(-0.5 * (np.einsum("ti,ij,tj->t", d, ci, d).sum() + len(x) * ld))


def main() -> int:
    tag = [x for x in sys.argv[1:] if "=" not in x][0]
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    sigma = float(opt.get("sigma", 0.926))
    from snv_a0_measure import load, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    gen = EV.load_gen()
    s = EV.System(tag, gen)
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (Tz["mu"], Tz["scalings"])

    def emb(w16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"]
    Et = {v: emb(np.concatenate([lw(EV.A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)])) for v in tgts}
    top = lambda e: max(tgts, key=lambda u: float(e @ Et[u]))
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    play = {v: EV.utt(f"a0/{v}_3") for v in tgts}
    tt = lambda u, k: torch.from_numpy(u[k])[None].to(DEV)

    @torch.no_grad()
    def u_of(utt: dict, stat) -> np.ndarray:
        return s.play(tt(utt, "y"), tt(utt, "v"), stat)[0].cpu().numpy().T

    stat = {v: s.op_stat(trefs[v]) for v in tgts}
    U_ref = {v: np.concatenate([u_of(r, stat[v]) for r in trefs[v]]) for v in tgts}
    U_play = {v: u_of(play[v], stat[v]) for v in tgts}
    m = {v: U_ref[v].mean(0) for v in tgts}
    M = np.stack([m[v] for v in tgts])
    mbar = M.mean(0)
    within = np.concatenate([U_ref[v] - m[v] for v in tgts])
    res: dict = {"tag": tag, "sigma_prior_mean": sigma, "dims": int(M.shape[1]), "voices": len(tgts)}
    res["first_order"] = {
        "between_voice_std_mean": round(float(M.std(0).mean()), 4),
        "within_voice_frame_std_mean": round(float(within.std(0).mean()), 4),
        "between_over_sigma": round(float(M.std(0).mean() / sigma), 4),
        "kl_cost_nat_per_frame": round(float((((M - mbar) ** 2) / (2 * sigma ** 2)).sum(1).mean()), 4),
        "utt_mean_nearest_voice": round(float(np.mean([min(tgts, key=lambda w: float(((U_play[v].mean(0) - m[w]) ** 2).sum())) == v for v in tgts])), 4),
    }
    q = {v: qda_fit(U_ref[v] - m[v]) for v in tgts}
    res["second_order"] = {"qda_centered_top1": round(float(np.mean([max(tgts, key=lambda w: qda_ll(U_play[v] - U_play[v].mean(0), q[w])) == v for v in tgts])), 4)}
    print(json.dumps(res, ensure_ascii=False), flush=True)

    @torch.no_grad()
    def render(src: dict, a: np.ndarray, S: dict) -> np.ndarray:
        y, v = tt(src, "y"), tt(src, "v")
        lf = tt(src, "lf")
        yo = s.C(torch.from_numpy(a.T.astype(np.float32))[None].to(DEV), lf, v, S)
        cond = torch.cat([yo[:, :25], lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(src["f0"])[None].to(DEV)
        return s.gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    pairs = [(tgts[(i + d) % len(tgts)], v) for i, v in enumerate(tgts) for d in (1, 8, 15)]
    rows = {}
    for mode in ("none", "shift"):
        hit, src_ = [], []
        for u, v in pairs:
            a = U_play[u] if mode == "none" else U_play[u] - m[u] + m[v]
            t1 = top(emb(trim(w16(render(play[u], a, s.structure(f"t|{v}", trefs[v]))))))
            hit.append(t1 == v), src_.append(t1 == u)
        rows[mode] = {"M_FF": round(float(np.mean(hit)), 4), "src_FF": round(float(np.mean(src_)), 4), "other": round(1 - float(np.mean(hit)) - float(np.mean(src_)), 4)}
        print(mode, json.dumps(rows[mode]), flush=True)
    res["ff"] = rows
    out = ROOT / opt.get("out", "results/snv_i/voice_code_10k.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

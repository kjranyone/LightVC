"""P1c の判定器 J2 の欺きやすさの探り(diag・学習なし・製品判定に引用しない): 写し + 平均のずらし MS の Y に、|δ| ≤ ε(Y の単位・全フレーム)の摂動を
勾配で 100 歩足して J2 の埋め込みを目標の登録へ寄せる。J2 の 30 声中の 1 位が上がるのに ECAPA(描画した音)が動かない ε が小さいほど、判定器の圧は欺きに落ちやすい。
組 = VAL(P1 の検証の女声 30)の 10 声 × VCTK の男声 6 × 1 本・k = 7。比べに J(平均を読む判定器)も同じ探りで。
移りの検査(J3 の関門): judges=J3:results/snv_p1c_judge3/judge.pt,J2:results/snv_p1c_judge/judge.pt sig=results/snv_p1c_judge3/judge.pt eps=0,0.1,0.25,0.5,1,2 out=diag_adv_transfer
sig= を与えると ε は次元ごとの時間の標準偏差 σ_d の単位(δ_d ≤ ε σ_d)。
    cd training && uv run --no-sync python snv_diag/p1c_adv_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import snv_p1_eval as EV
import train_snv_p1 as T
from train_snv_p1j import Judge

ROOT = Path(__file__).resolve().parent.parent.parent
DEV = "cuda"
EPS = (0.0, 0.01, 0.02, 0.05)
STEPS = 100


def main() -> int:
    from snv_a0_measure import load, to16
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    gen = EV.load_gen()
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    jl = [x.split(":", 1) for x in opt["judges"].split(",")] if "judges" in opt else [("J2", "results/snv_p1c_judge/judge.pt"), ("J", "results/snv_p1_judge/judge.pt")]
    eps_l = tuple(float(x) for x in opt["eps"].split(",")) if "eps" in opt else EPS
    unit = "sigma" if "sig" in opt else "raw"
    scale = torch.load(ROOT / opt["sig"], map_location="cpu", weights_only=False)["sig"].to(DEV) if unit == "sigma" else torch.ones(T.D_Y, device=DEV)
    judges = {}
    for name, p in jl:
        sj = torch.load(ROOT / p, map_location="cpu", weights_only=False)
        J = Judge(len(sj["classes"]), norm=sj.get("norm", "none")).to(DEV).eval().requires_grad_(False)
        J.load_state_dict(sj["J"])
        judges[name] = J

    def emb(w16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)

    @torch.no_grad()
    def render(y: torch.Tensor, f0: np.ndarray) -> np.ndarray:
        v = torch.from_numpy((f0 > 0).astype(np.float32))[None].to(DEV)
        lf = torch.from_numpy(np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32))[None].to(DEV)
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        return gen(cond, EV.TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(DEV), cond.shape[-1] * EV.N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[EV.N.DELAY:].astype(np.float32)

    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    tg_all = [v for v in split["val"] if v.startswith("ef")]
    tg = tg_all[::3]
    spk = sorted({p.stem.split("_")[0] for p in (T.FEAT / "vctk").glob("*.npz")})
    srcs, enr = [], {}
    for s_ in spk:
        fs = sorted((T.FEAT / "vctk").glob(f"{s_}_*.npz"))
        enr[s_] = EV.mu_of([T.load_utt(f) for f in fs[:2]])
        srcs.append((s_, T.load_utt(fs[2])))
    Et = {v: emb(np.concatenate([lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 4, 5)])) for v in tg_all}
    refY = {v: torch.from_numpy(np.concatenate([EV.utt(f"e1/{v}/t0{j}")["y"] for j in (3, 4, 5)], 1))[None].to(DEV) for v in tg_all}
    res: dict = {"prereg": "results/snv_p1c/prereg.yaml(diag・判定器の欺きやすさの探り)", "steps": STEPS, "n_pairs": len(srcs) * len(tg), "eps_unit": unit,
                 "y_time_std_mean": round(float(np.mean([u["y"][:25].std(1).mean() for _, u in srcs])), 4), "by": {}}
    for jn, J in judges.items():
        with torch.no_grad():
            Jt = {v: J.embed(refY[v])[0] for v in tg_all}
        Jm = torch.stack([Jt[v] for v in tg_all])
        for eps in eps_l:
            jh, eh, dl = [], [], []
            for s_, src in srcs:
                y = torch.from_numpy(src["y"])[None].to(DEV)
                f0 = src["f0"] * 2 ** (7 / 12)
                for v in tg:
                    env = T.y_env(y) - enr[s_][..., None] + EV.mu_of([EV.utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)])[..., None]
                    y0 = torch.cat([(env[:, :1] - T.PA.C0_SIL) / 10, env[:, 1:25] / 10, y[:, 25:29]], 1)
                    u = torch.zeros_like(y0, requires_grad=True)
                    eb = eps * scale[None, :, None]
                    if eps > 0:
                        adam = torch.optim.Adam([u], lr=0.1)
                        for _ in range(STEPS):
                            e = J.embed(y0 + u * eb)[0]
                            loss = -(e @ Jt[v]) + torch.logsumexp(Jm @ e * 30.0, 0) / 30.0
                            adam.zero_grad()
                            loss.backward()
                            adam.step()
                            with torch.no_grad():
                                u.clamp_(-1.0, 1.0)
                    with torch.no_grad():
                        d = u * eb
                        ya = y0 + d
                        e = J.embed(ya)[0]
                        jh.append(tg_all[int((Jm @ e).argmax())] == v)
                        we = emb(trim(w16(render(ya, f0))))
                        eh.append(max(tg_all, key=lambda u: float(we @ Et[u])) == v)
                        dl.append(float(d.abs().mean()))
            res["by"][f"{jn}_eps{eps}"] = {"judge_top1_30": round(float(np.mean(jh)), 4), "ecapa_top1_30": round(float(np.mean(eh)), 4), "mean_abs_delta": round(float(np.mean(dl)), 4)}
            print(jn, eps, json.dumps(res["by"][f"{jn}_eps{eps}"]), flush=True)
    out = ROOT / f"results/snv_p1c/{opt.get('out', 'diag_adv_probe')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

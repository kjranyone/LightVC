"""分布の損失の移りの探り(diag・学習なし・製品判定に引用しない): 写し + 平均のずらし MS の Y に δ を足し、
MMD²(変換の有声のフレーム, 目標の参照の有声のフレーム)+ λ · mean((δ / σ_d)²) を 200 歩で下げる(固定の核 = 学習した判定器の盲点が無い・λ = 内容の錨 = 最小の移動)。
フレームの特徴 = 次元ごとの σ_d で割った Y(29)と 1 フレーム差(29)。核 = 帯域 {0.5, 1, 2, 4, 8} × 次元数 の RBF の和。
読み = 描画した音の ECAPA の 30 声中の 1 位(VAL 女声 30・登録 t03〜t05)・J2 の 1 位・移動の大きさ。上限の参考 = 目標の写し(高さだけ運用)0.544・床 MS 0.036。
組 = VAL の 10 声 × VCTK の男声 6 × 1 本・k = 7。
ruler=test = test の物差し(本物の男声の 12 区間の 1 本おき 6 → A0F 24 声・ECAPA + LDA の 24 声中の 1 位・k = 7・上限 COPY_k7 0.653・床 MS 0.056)。
    cd training && uv run --no-sync python snv_diag/p1d_mmd_probe.py [ruler=test]
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
LAMS = (None, 1.0, 0.1, 0.01)
STEPS = 200
NX, NY = 384, 768
BW = (0.5, 1.0, 2.0, 4.0, 8.0)


def feats(y: torch.Tensor, sig: torch.Tensor, sigd: torch.Tensor) -> torch.Tensor:
    d = torch.cat([y[..., :1] * 0, y[..., 1:] - y[..., :-1]], -1)
    return torch.cat([y / sig[:, None], d / sigd[:, None]], 0).T


def mmd2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    D = x.shape[1]
    xx, yy, xy = torch.cdist(x, x).pow(2), torch.cdist(y, y).pow(2), torch.cdist(x, y).pow(2)
    k = lambda d: sum(torch.exp(-d / (b * D)) for b in BW)
    n, m = len(x), len(y)
    kxx = (k(xx).sum() - len(BW) * n) / (n * (n - 1))
    kyy = (k(yy).sum() - len(BW) * m) / (m * (m - 1))
    return kxx + kyy - 2 * k(xy).mean()


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    ruler = opt.get("ruler", "dev")
    from snv_a0b_measure import proj
    from snv_a0_measure import load, to16
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    gen = EV.load_gen()
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    sj = torch.load(ROOT / "results/snv_p1c_judge/judge.pt", map_location="cpu", weights_only=False)
    J2 = Judge(len(sj["classes"]), norm=sj["norm"]).to(DEV).eval().requires_grad_(False)
    J2.load_state_dict(sj["J"])

    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")

    def emb(w16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        e = torch.nn.functional.normalize(e, dim=0).cpu().numpy()
        return proj(e, (Tz["mu"], Tz["scalings"])) if ruler == "test" else e

    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)

    @torch.no_grad()
    def render(y: torch.Tensor, f0: np.ndarray) -> np.ndarray:
        v = torch.from_numpy((f0 > 0).astype(np.float32))[None].to(DEV)
        lf = torch.from_numpy(np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32))[None].to(DEV)
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        return gen(cond, EV.TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(DEV), cond.shape[-1] * EV.N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[EV.N.DELAY:].astype(np.float32)

    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    tr_us = [T.load_utt(p) for v in split["train"][::12] for p in sorted((T.FEAT / "e1" / v).glob("t0[0-4].npz"))]
    sig = torch.from_numpy(np.mean([u["y"].std(1) for u in tr_us], 0)).to(DEV)
    sigd = torch.from_numpy(np.mean([np.diff(u["y"], axis=1).std(1) for u in tr_us], 0)).to(DEV)
    srcs, enr = [], {}
    if ruler == "test":
        meta = json.loads((EV.A1 / "convert_meta.json").read_text())
        tg_all = meta["targets"]
        tg = tg_all
        for s_ in meta["sources"][::2]:
            sp = s_.rsplit("_", 1)[0]
            enr[s_] = EV.mu_of([EV.utt(f"real/{sp}_enroll")])
            srcs.append((s_, EV.utt(f"real/{s_}")))
        Et = {v: emb(np.concatenate([lw(EV.A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)])) for v in tg_all}
        jy = {v: [EV.utt(f"a0/{v}_{j}")["y"] for j in (3, 4, 5)] for v in tg_all}
        refs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tg}
    else:
        tg_all = [v for v in split["val"] if v.startswith("ef")]
        tg = tg_all[::3]
        spk = sorted({p.stem.split("_")[0] for p in (T.FEAT / "vctk").glob("*.npz")})
        for s_ in spk:
            fs = sorted((T.FEAT / "vctk").glob(f"{s_}_*.npz"))
            enr[s_] = EV.mu_of([T.load_utt(f) for f in fs[:2]])
            srcs.append((s_, T.load_utt(fs[2])))
        Et = {v: emb(np.concatenate([lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 4, 5)])) for v in tg_all}
        jy = {v: [EV.utt(f"e1/{v}/t0{j}")["y"] for j in (3, 4, 5)] for v in tg_all}
        refs = {v: [EV.utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)] for v in tg}
    with torch.no_grad():
        Jt = {v: J2.embed(torch.from_numpy(np.concatenate(jy[v], 1))[None].to(DEV))[0] for v in tg_all}
    Jm = torch.stack([Jt[v] for v in tg_all])
    rfeat = {}
    for v in tg:
        ry = torch.from_numpy(np.concatenate([r["y"] for r in refs[v]], 1)).to(DEV)
        rv = torch.from_numpy(np.concatenate([r["v"] for r in refs[v]])).to(DEV) > 0
        rfeat[v] = feats(ry, sig, sigd)[rv]
    g = torch.Generator(device=DEV).manual_seed(0)
    res: dict = {"prereg": "diag(分布の損失の移りの探り)", "ruler": ruler, "steps": STEPS, "n_pairs": len(srcs) * len(tg), "by": {}}
    for lam in LAMS:
        eh, jh, mv, m0, m1 = [], [], [], [], []
        for s_, src in srcs:
            y = torch.from_numpy(src["y"])[None].to(DEV)
            f0 = src["f0"] * 2 ** (7 / 12)
            sv = torch.from_numpy(src["v"]).to(DEV) > 0
            for v in tg:
                env = T.y_env(y) - enr[s_][..., None] + EV.mu_of(refs[v])[..., None]
                y0 = torch.cat([(env[:, :1] - T.PA.C0_SIL) / 10, env[:, 1:25] / 10, y[:, 25:29]], 1)
                d = torch.zeros_like(y0, requires_grad=True)
                rf = rfeat[v]
                if lam is not None:
                    adam = torch.optim.Adam([d], lr=0.02)
                    for it in range(STEPS):
                        fx = feats((y0 + d)[0], sig, sigd)[sv]
                        ix = torch.randint(len(fx), (min(NX, len(fx)),), generator=g, device=DEV)
                        iy = torch.randint(len(rf), (min(NY, len(rf)),), generator=g, device=DEV)
                        loss = mmd2(fx[ix], rf[iy]) + lam * ((d[0] / sig[:, None]) ** 2).mean()
                        adam.zero_grad()
                        loss.backward()
                        adam.step()
                with torch.no_grad():
                    ya = (y0 + d).detach()
                    fx = feats(ya[0], sig, sigd)[sv]
                    iy = torch.randint(len(rf), (min(NY, len(rf)),), generator=g, device=DEV)
                    m1.append(float(mmd2(fx[:NX * 2], rf[iy])))
                    m0.append(float(mmd2(feats(y0[0], sig, sigd)[sv][:NX * 2], rf[iy])))
                    we = emb(trim(w16(render(ya, f0))))
                    eh.append(max(tg_all, key=lambda u: float(we @ Et[u])) == v)
                    jh.append(tg_all[int((Jm @ J2.embed(ya)[0]).argmax())] == v)
                    mv.append(float((d[0] / sig[:, None]).abs().mean()))
        key = "MS" if lam is None else f"lam{lam}"
        res["by"][key] = {"ecapa_top1": round(float(np.mean(eh)), 4), "J2_top1": round(float(np.mean(jh)), 4), "move_sigma": round(float(np.mean(mv)), 4),
                          "mmd2_before": round(float(np.mean(m0)), 4), "mmd2_after": round(float(np.mean(m1)), 4)}
        print(key, json.dumps(res["by"][key]), flush=True)
    out = ROOT / ("results/snv_p1c/diag_mmd_probe.json" if ruler == "dev" else "results/snv_p1c/diag_mmd_probe_test.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

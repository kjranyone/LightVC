"""P1c の起動前の較正(results/snv_p1c/prereg.yaml): 判定器 J2 の関門と、運用の条件での物差しの床と天井。
judge: VAL(P1 の検証の女声 30)× VCTK の男声 6 × 2 本・k = 7。J2 の埋め込みでの 30 声中の 1 位(登録 = t03〜t05 の Y)を
ORI(目標の t06)・SRC(源の Y)・MS(写し + 平均のずらし)・n(P1 の対照 40k)・wj(P1 の 20k)で。n は ECAPA の開発の M も(次の腕の対照)。
ruler: test の物差し(本物の男声 12 区間 → A0F 24 声・ECAPA + LDA の 1 位・k ∈ {0, 5, 7, 12})の
床 MS(網なしの写し + 平均のずらし)と天井 COPY(目標の j 3〜5 の写しで高さの中央値だけ源 × 2^(k/12) へ = 構造の変換が完全なときの上限)。
    cd training && uv run --no-sync python snv_p1c_cal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import snv_p1_eval as EV
import train_snv_p1 as T
from train_snv_p1j import Judge

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda"


def med(f0: np.ndarray) -> float:
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) else 200.0


def lf_of(f0: np.ndarray) -> np.ndarray:
    return np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)


def main() -> int:
    from snv_a0_measure import load, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    gen = EV.load_gen()
    sysn, sysw = EV.System("snv_p1_n", gen), EV.System("snv_p1_wj@step20k", gen)
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (Tz["mu"], Tz["scalings"])
    sj = torch.load(ROOT / "results/snv_p1c_judge/judge.pt", map_location="cpu", weights_only=False)
    J = Judge(len(sj["classes"]), norm=sj["norm"]).to(DEV).eval().requires_grad_(False)
    J.load_state_dict(sj["J"])

    def emb(w16: np.ndarray, lda: bool) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        e = torch.nn.functional.normalize(e, dim=0).cpu().numpy()
        return proj(e, Tt) if lda else e

    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)
    jemb = lambda y: J.embed(torch.as_tensor(y, device=DEV).reshape(1, T.D_Y, -1))[0].cpu().numpy()

    @torch.no_grad()
    def render(y: torch.Tensor, f0: np.ndarray) -> np.ndarray:
        v = torch.from_numpy((f0 > 0).astype(np.float32))[None].to(DEV)
        lf = torch.from_numpy(lf_of(f0))[None].to(DEV)
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        return gen(cond, EV.TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(DEV), cond.shape[-1] * EV.N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[EV.N.DELAY:].astype(np.float32)

    def ms(src: dict, mu_op: torch.Tensor, refs: list[dict]) -> torch.Tensor:
        y = torch.from_numpy(src["y"])[None].to(DEV)
        env = T.y_env(y) - mu_op[..., None] + EV.mu_of(refs)[..., None]
        return torch.cat([(env[:, :1] - T.PA.C0_SIL) / 10, env[:, 1:25] / 10, y[:, 25:29]], 1)

    @torch.no_grad()
    def net(s: EV.System, src: dict, mu_op: torch.Tensor, key: str, refs: list[dict], f0: np.ndarray) -> torch.Tensor:
        y = torch.from_numpy(src["y"])[None].to(DEV)
        v = torch.from_numpy(src["v"])[None].to(DEV)
        return s.C(s.E(T.e_in(y, v, mu_op)), torch.from_numpy(lf_of(f0))[None].to(DEV), v, s.structure(key, refs))

    res: dict = {"prereg": "results/snv_p1c/prereg.yaml", "judge": sj["res"]}
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    tg = [v for v in split["val"] if v.startswith("ef")]
    spk = sorted({p.stem.split("_")[0] for p in (T.FEAT / "vctk").glob("*.npz")})
    srcs, enr = [], {}
    for s_ in spk:
        fs = sorted((T.FEAT / "vctk").glob(f"{s_}_*.npz"))
        enr[s_] = EV.mu_of([T.load_utt(f) for f in fs[:2]])
        srcs += [(s_, T.load_utt(fs[2])), (s_, T.load_utt(fs[3]))]
    refs = {v: [EV.utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)] for v in tg}
    Jt = {v: jemb(np.concatenate([EV.utt(f"e1/{v}/t0{j}")["y"] for j in (3, 4, 5)], 1)) for v in tg}
    Et = {v: emb(np.concatenate([lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 4, 5)]), False) for v in tg}
    top = lambda e, ref: max(tg, key=lambda u: float(e @ ref[u]))
    jg = {"ORI": float(np.mean([top(jemb(EV.utt(f"e1/{v}/t06")["y"]), Jt) == v for v in tg]))}
    rows = {k_: [] for k_ in ("SRC", "MS", "n", "wj")}
    n_ecapa = []
    for s_, src in srcs:
        f0 = src["f0"] * 2 ** (7 / 12)
        for v in tg:
            ys = {"SRC": torch.from_numpy(src["y"])[None].to(DEV), "MS": ms(src, enr[s_], refs[v]),
                  "n": net(sysn, src, enr[s_], f"VAL|{v}", refs[v], f0), "wj": net(sysw, src, enr[s_], f"VAL|{v}", refs[v], f0)}
            for k_, y in ys.items():
                rows[k_].append(top(jemb(y[0].cpu().numpy()), Jt) == v)
            n_ecapa.append(top(emb(trim(w16(render(ys["n"], f0))), False), Et) == v)
    jg.update({k_: round(float(np.mean(r)), 4) for k_, r in rows.items()})
    res["J2_top1_VAL_k7"] = jg
    res["dev_M_k7_n40k_ecapa"] = round(float(np.mean(n_ecapa)), 4)
    print("judge gate", json.dumps(jg), "n40k dev M_k7", res["dev_M_k7_n40k_ecapa"], flush=True)
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts, tsrc = meta["targets"], meta["sources"]
    EtL = {v: emb(np.concatenate([lw(EV.A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)]), True) for v in tgts}
    topL = lambda e: max(tgts, key=lambda u: float(e @ EtL[u]))
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    src_u = {s_: EV.utt(f"real/{s_}") for s_ in tsrc}
    mu_op = {sp: EV.mu_of([EV.utt(f"real/{sp}_enroll")]) for sp in ("uirou", "namikawa")}
    ruler: dict = {}
    for k in EV.KS:
        cp, mm = [], []
        for si, s_ in enumerate(tsrc):
            src = src_u[s_]
            for v in tgts:
                u = EV.utt(f"a0/{v}_{3 + si % 3}")
                wc = render(torch.from_numpy(u["y"])[None].to(DEV), u["f0"] * (med(src["f0"]) * 2 ** (k / 12) / med(u["f0"])))
                wm = render(ms(src, mu_op[s_.rsplit("_", 1)[0]], trefs[v]), src["f0"] * 2 ** (k / 12))
                cp.append(topL(emb(trim(w16(wc)), True)) == v)
                mm.append(topL(emb(trim(w16(wm)), True)) == v)
        ruler[f"k{k}"] = {"COPY": round(float(np.mean(cp)), 4), "MS": round(float(np.mean(mm)), 4)}
        print("ruler", k, json.dumps(ruler[f"k{k}"]), flush=True)
    ruler["mean_4k"] = {x: round(float(np.mean([ruler[f"k{k}"][x] for k in EV.KS])), 4) for x in ("COPY", "MS")}
    res["test_ruler"] = ruler
    print("ruler mean", json.dumps(ruler["mean_4k"]), flush=True)
    (ROOT / "results/snv_p1c").mkdir(exist_ok=True)
    (ROOT / "results/snv_p1c/cal.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

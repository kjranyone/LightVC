"""P1 の途中(20k)の診断 2(diag・製品判定に引用しない): 運用の高さ(源 + 固定の k 半音)が ECAPA の本人らしさの天井をどこに置くか。
組 = VAL(検証の女声 30)・源 = VCTK の男声 6 × 2 本。ECAPA の 30 声中の 1 位(登録 = t03〜t05 の元の音)・目標と他の cos の差・源の登録より目標に近い率。
条件: COPY_ORIP = 目標の t06 の写し(自分の高さ)・COPY_k = 目標の t06 の写しで高さの中央値を源 × 2^(k/12) に合わせる(包絡は目標そのもの = 構造の変換が完全なときの上限)・
MS / n / wj の k(運用)と TM(源の高さの形を目標の中央値へ・運用でない)。
    cd training && uv run --no-sync python snv_diag/p1_mid_diag2.py
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

ROOT = Path(__file__).resolve().parent.parent.parent
DEV = "cuda"


def med(f0: np.ndarray) -> float:
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) else 200.0


def main() -> int:
    from snv_a0_measure import load, to16
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    gen = EV.load_gen()
    systems = {t: EV.System(t, gen) for t in ("snv_p1_n@step20k", "snv_p1_wj@step20k")}
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

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
        f0t = torch.from_numpy(f0.astype(np.float32))[None].to(DEV)
        return gen(cond, EV.TR.excitation(f0t, cond.shape[-1] * EV.N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[EV.N.DELAY:].astype(np.float32)

    @torch.no_grad()
    def conv(name: str, src: dict, mu_op: torch.Tensor, key: str, refs: list[dict], f0: np.ndarray) -> torch.Tensor:
        y = torch.from_numpy(src["y"])[None].to(DEV)
        if name == "MS":
            env = T.y_env(y) - mu_op[..., None] + EV.mu_of(refs)[..., None]
            return torch.cat([(env[:, :1] - T.PA.C0_SIL) / 10, env[:, 1:25] / 10, y[:, 25:29]], 1)
        v = torch.from_numpy(src["v"])[None].to(DEV)
        lf = torch.from_numpy(np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32))[None].to(DEV)
        s = systems[name]
        return s.C(s.E(T.e_in(y, v, mu_op)), lf, v, s.structure(key, refs))

    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    tg = [v for v in split["val"] if v.startswith("ef")]
    spk = sorted({p.stem.split("_")[0] for p in (T.FEAT / "vctk").glob("*.npz")})
    srcs, enr = [], {}
    for s_ in spk:
        fs = sorted((T.FEAT / "vctk").glob(f"{s_}_*.npz"))
        enr[s_] = EV.mu_of([T.load_utt(f) for f in fs[:2]])
        srcs += [(s_, T.load_utt(fs[2])), (s_, T.load_utt(fs[3]))]
    Es = {s_: emb(np.concatenate([lw(f.with_suffix(".wav")) for f in sorted((T.FEAT / "vctk").glob(f"{s_}_*.npz"))[:2]])) for s_ in spk}
    refs = {v: [EV.utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)] for v in tg}
    Et = {v: emb(np.concatenate([lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 4, 5)])) for v in tg}
    t06 = {v: EV.utt(f"e1/{v}/t06") for v in tg}
    res: dict = {"prereg": "results/snv_p1/prereg.yaml(diag・途中の切り分け 2)", "set": "VAL", "src_median_f0": {s_: round(med(u["f0"]), 1) for s_, u in srcs},
                 "tgt_median_f0_mean": round(float(np.mean([med(t06[v]["f0"]) for v in tg])), 1)}

    def score(rows: list) -> dict:
        hit = [r[0] for r in rows]
        return {"ecapa_top1": round(float(np.mean(hit)), 4), "ecapa_margin": round(float(np.mean([r[1] for r in rows])), 4), "src_closer": round(float(np.mean([r[2] for r in rows])), 4), "n": len(rows)}

    def judge(w: np.ndarray, v: str, s_: str) -> tuple:
        e = emb(trim(w16(w)))
        sc = {u: float(e @ Et[u]) for u in tg}
        return max(sc, key=sc.get) == v, sc[v] - float(np.mean([sc[u] for u in tg if u != v])), float(e @ Es[s_]) > sc[v]

    rows = [judge(render(torch.from_numpy(t06[v]["y"])[None].to(DEV), t06[v]["f0"]), v, srcs[0][0]) for v in tg]
    res["COPY_ORIP"] = score(rows)
    print("COPY_ORIP", json.dumps(res["COPY_ORIP"]), flush=True)
    for k in (0, 7, 12):
        rows = []
        for s_, src in srcs:
            for v in tg:
                f0 = t06[v]["f0"] * (med(src["f0"]) * 2 ** (k / 12) / med(t06[v]["f0"]))
                rows.append(judge(render(torch.from_numpy(t06[v]["y"])[None].to(DEV), f0), v, s_))
        res[f"COPY_k{k}"] = score(rows)
        print(f"COPY_k{k}", json.dumps(res[f"COPY_k{k}"]), flush=True)
    for nm in ("MS", "snv_p1_n@step20k", "snv_p1_wj@step20k"):
        for pc in ("k12", "TM"):
            rows = []
            for s_, src in srcs:
                for v in tg:
                    f0 = src["f0"] * (2 ** (12 / 12) if pc == "k12" else med(t06[v]["f0"]) / med(src["f0"]))
                    rows.append(judge(render(conv(nm, src, enr[s_], f"VAL|{v}", refs[v], f0), f0), v, s_))
            res[f"{nm}_{pc}"] = score(rows)
            print(f"{nm}_{pc}", json.dumps(res[f"{nm}_{pc}"]), flush=True)
    out = ROOT / "results/snv_p1/diag_mid20k_pitch.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

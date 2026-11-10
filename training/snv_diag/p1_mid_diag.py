"""P1 の途中(20k)の診断(diag・製品判定に引用しない): 開発の物差しの較正と、判定器の圧が ECAPA に移らない理由の切り分け。
組 = VAL(検証の女声 30・未学習)と TRN(学習の女声 30・判定器の閉じた組)。源 = VCTK の男声 6 × 2 本(登録 = 別の 2 本)・k = 7。
系 = ORI(目標の t06 の元の音・物差しの上限)・SRC(源の Y をそのまま)・MS(源の包絡 − 源の登録の平均 + 目標の登録の平均 = 写し + 平均のずらし)・n / wj の 20k。
読み = ECAPA の 30 声中の 1 位・目標と他の声の cos の差・源の登録より目標に近い率・判定器 J の埋め込みでの 1 位(Y で)・TRN では J の分類の 1 位・描画した音を前段 v3 で読み直した J の 1 位(源 1 本)。
    cd training && uv run --no-sync python snv_diag/p1_mid_diag.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_p1 as T
from train_snv_p1j import Judge

ROOT = Path(__file__).resolve().parent.parent.parent
DEV = "cuda"
K = 7


def main() -> int:
    from snv_a0_measure import load, to16
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    gen = EV.load_gen()
    systems = {t: EV.System(t, gen) for t in ("snv_p1_n@step20k", "snv_p1_wj@step20k")}
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    sj = torch.load(ROOT / "results/snv_p1_judge/judge.pt", map_location="cpu", weights_only=False)
    J = Judge(len(sj["classes"])).to(DEV).eval().requires_grad_(False)
    J.load_state_dict(sj["J"])
    cls = {n: i for i, n in enumerate(sj["classes"])}

    def emb(w16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)
    jemb = lambda y: J.embed(torch.as_tensor(y, device=DEV).reshape(1, T.D_Y, -1))[0].cpu().numpy()

    @torch.no_grad()
    def render(y: torch.Tensor, lf: torch.Tensor, v: torch.Tensor, f0: np.ndarray, k: int) -> np.ndarray:
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0t = torch.from_numpy(f0 * 2 ** (k / 12))[None].to(DEV)
        return gen(cond, EV.TR.excitation(f0t, cond.shape[-1] * EV.N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[EV.N.DELAY:].astype(np.float32)

    @torch.no_grad()
    def convert_y(name: str, src: dict, mu_op: torch.Tensor, v_: str, refs: list[dict], k: int) -> torch.Tensor:
        y = torch.from_numpy(src["y"])[None].to(DEV)
        v = torch.from_numpy(src["v"])[None].to(DEV)
        lf = torch.from_numpy(src["lf"])[None].to(DEV) + k * T.LN2_12 * v
        if name == "SRC":
            return y
        if name == "MS":
            env = T.y_env(y) - mu_op[..., None] + EV.mu_of(refs)[..., None]
            return torch.cat([(env[:, :1] - T.PA.C0_SIL) / 10, env[:, 1:25] / 10, y[:, 25:29]], 1)
        s = systems[name]
        return s.C(s.E(T.e_in(y, v, mu_op)), lf, v, s.structure(f"{v_}", refs))

    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    val_f = [v for v in split["val"] if v.startswith("ef")]
    trn_all = [v for v in split["train"] if v.startswith("ef")]
    trn_f = trn_all[:: max(1, len(trn_all) // 30)][:30]
    spk = sorted({p.stem.split("_")[0] for p in (T.FEAT / "vctk").glob("*.npz")})
    srcs, enr = [], {}
    for s_ in spk:
        fs = sorted((T.FEAT / "vctk").glob(f"{s_}_*.npz"))
        enr[s_] = EV.mu_of([T.load_utt(f) for f in fs[:2]])
        srcs += [(s_, T.load_utt(fs[2]), fs[2].stem)]
        srcs += [(s_, T.load_utt(fs[3]), fs[3].stem)]
    Es = {s_: emb(np.concatenate([lw(f.with_suffix(".wav")) for f in sorted((T.FEAT / "vctk").glob(f"{s_}_*.npz"))[:2]])) for s_ in spk}
    names = ["SRC", "MS", "snv_p1_n@step20k", "snv_p1_wj@step20k"]
    res: dict = {"prereg": "results/snv_p1/prereg.yaml(diag・途中の切り分け)", "k": K, "n_src": len(srcs)}
    t0 = time.time()
    for set_name, tg in (("VAL", val_f), ("TRN", trn_f)):
        refs = {v: [EV.utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)] for v in tg}
        Et = {v: emb(np.concatenate([lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 4, 5)])) for v in tg}
        Jt = {v: jemb(np.concatenate([EV.utt(f"e1/{v}/t0{j}")["y"] for j in (3, 4, 5)], 1)) for v in tg}
        eo = {v: emb(trim(lw(ROOT / f"data/snv_e1/{v}/t06.wav"))) for v in tg}
        jo = {v: jemb(EV.utt(f"e1/{v}/t06")["y"]) for v in tg}
        ori = [max(tg, key=lambda u: float(eo[v] @ Et[u])) == v for v in tg]
        jori = [max(tg, key=lambda u: float(jo[v] @ Jt[u])) == v for v in tg]
        row: dict = {"ORI_ecapa_top1": round(float(np.mean(ori)), 4), "ORI_J_top1": round(float(np.mean(jori)), 4)}
        for nm in names:
            hit, marg, closer, jhit, jcls, rehit = [], [], [], [], [], []
            for si, (s_, src, stem) in enumerate(srcs):
                for v in tg:
                    yo = convert_y(nm, src, enr[s_], f"{set_name}|{v}", refs[v], K)
                    v_t = torch.from_numpy(src["v"])[None].to(DEV)
                    lf_t = torch.from_numpy(src["lf"])[None].to(DEV) + K * T.LN2_12 * v_t
                    w = render(yo, lf_t, v_t, src["f0"], K)
                    e = emb(trim(w16(w)))
                    sc = {u: float(e @ Et[u]) for u in tg}
                    hit.append(max(sc, key=sc.get) == v)
                    marg.append(sc[v] - float(np.mean([sc[u] for u in tg if u != v])))
                    closer.append(float(e @ Es[s_]) > sc[v])
                    je = jemb(yo[0].cpu().numpy())
                    jhit.append(max(tg, key=lambda u: float(je @ Jt[u])) == v)
                    if set_name == "TRN":
                        jcls.append(int(J.logits(yo.float())[0].argmax()) == cls[v])
                    if si == 0:
                        f = P.analyze(w.astype(np.float64))
                        yr = T.to_y(f["env"], f["per"], f["f0"])
                        rehit.append(max(tg, key=lambda u: float(jemb(yr) @ Jt[u])) == v)
            row[nm] = {"ecapa_top1": round(float(np.mean(hit)), 4), "ecapa_margin": round(float(np.mean(marg)), 4), "src_closer": round(float(np.mean(closer)), 4),
                       "J_top1_Y": round(float(np.mean(jhit)), 4), "J_top1_reanalysed_src0": round(float(np.mean(rehit)), 4)}
            if jcls:
                row[nm]["J_class_top1_564"] = round(float(np.mean(jcls)), 4)
            print(set_name, nm, json.dumps(row[nm]), f"{time.time() - t0:.0f}s", flush=True)
        res[set_name] = row
        print(set_name, "ORI", row["ORI_ecapa_top1"], row["ORI_J_top1"], flush=True)
    out = ROOT / "results/snv_p1/diag_mid20k.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

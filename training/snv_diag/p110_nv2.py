"""段 P110 の物差し v2(学習なし): 音素の無い発声の「声の当て」。同じ種類・同じ文の発声を、検証の女声 30 声が演じた音を候補にして、変換の出力がどの声に最も近いかを当てる(候補は声だけが違う)。
源 = 同じ文を演じた男声 6 声(em596 = 検証・em576〜em592 の 4 つおき = 学習の声)。目標 = 検証の女声 6 声。高さは目標の同じ発声の中央値にそろえる(半音を項目ごとに決める = 声色だけを見る)。
量: 話者の埋め込み(ECAPA)と包絡の平均(c1〜c24・有声)のそれぞれで、正しい目標の順位(1 位の率・逆順位の平均)。床 = 源そのまま(高さだけそろえた写し)を同じ候補で当てた値。
    cd training && SNV_ESM=1 SNV_ENVC=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p110_nv2.py <名前> [sys=<系>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import p107_edges as E
import snv_p1_eval as EV
from snv_a0_measure import to16

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ["em596", "em592", "em588", "em584", "em580", "em576"]
CATS = ["moan", "scream", "pant", "cry", "laugh", "sigh", "mixed"]


def main() -> None:
    name = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    man = {(x["voice"], x["cat"], x["k"]): x["text_id"] for x in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"]}
    pool = [v for v in json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"] if v.startswith("ef")]
    TGT = pool[:6]
    for c in CATS:
        assert len({man[(v, c, 0)] for v in pool + SRC}) == 1, c
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": EV.DEV})

    def emb(w: np.ndarray) -> np.ndarray:
        e = ec.encode_batch(torch.from_numpy(to16(np.asarray(w, np.float64), EV.SR).astype(np.float32))[None].to(EV.DEV))[0, 0].detach().cpu().numpy()
        return e / np.linalg.norm(e)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    mean = lambda y, v: y[1:25][:, v].mean(1)
    S = EV.Ens(opt.get("sys", E.SYS), EV.load_gen())
    p0 = S.parts[0]
    cand = {}
    for v in pool:
        for c in CATS:
            x = lw(ROOT / f"data/snv_e1nv/{v}/{c}_0.wav")
            u = PB.utt(x)
            f = u["f0"][u["f0"] > 0]
            cand[(v, c)] = (emb(x), mean(u["y"], u["f0"] > 0), float(np.median(f)) if len(f) else 0.0)
    st = {t: S.structure("t" + t, [PB.utt(lw(ROOT / f"data/snv_e1/{t}/t0{j}.wav")) for j in (0, 1, 2)]) for t in TGT}
    males = [f"em{i}" for i in range(480, 600, 4)]
    mc = {}
    for v in males:
        for c in CATS:
            x = lw(ROOT / f"data/snv_e1nv/{v}/{c}_0.wav")
            u = PB.utt(x)
            mc[(v, c)] = (emb(x), mean(u["y"], u["f0"] > 0))

    def gender(e: np.ndarray, m: np.ndarray, c: str, skip: tuple = ()) -> tuple[float, float]:
        fe = np.mean([float(e @ cand[(v, c)][0]) for v in pool if v not in skip])
        me = np.mean([float(e @ mc[(v, c)][0]) for v in males if v not in skip])
        fm = np.mean([float(np.linalg.norm(m - cand[(v, c)][1])) for v in pool if v not in skip])
        mm = np.mean([float(np.linalg.norm(m - mc[(v, c)][1])) for v in males if v not in skip])
        return fe - me, mm - fm
    gref = {"female": [gender(cand[(v, c)][0], cand[(v, c)][1], c, (v,)) for v in pool for c in CATS], "male": [gender(mc[(v, c)][0], mc[(v, c)][1], c, (v,)) for v in males for c in CATS]}
    rows = []
    for s in SRC:
        enr = [PB.utt(lw(ROOT / f"data/snv_e1/{s}/t0{j}.wav")) for j in (0, 1, 2)]
        for c in CATS:
            u = PB.utt(lw(ROOT / f"data/snv_e1nv/{s}/{c}_0.wav"))
            pu = p0.prep(u)
            vv = pu["f0"] > 0
            fs = float(np.median(pu["f0"][vv])) if vv.any() else 0.0
            for t in TGT:
                ft = cand[(t, c)][2]
                k = float(np.clip(12.0 * np.log2(ft / fs), -12.0, 24.0)) if fs > 0 and ft > 0 else 0.0
                w, yo = S.convert(u, enr, st[t], k, with_y=True)
                wsrc = p0.render(pu, torch.from_numpy(pu["y"][:29])[None].to(EV.DEV), k, None)
                y = yo[0].detach().cpu().numpy()
                q = {"out": (emb(np.asarray(w)), mean(y, vv)), "floor": (emb(np.asarray(wsrc)), mean(pu["y"], vv))}
                r = {"src": s, "cat": c, "tgt": t, "k": round(k, 1)}
                for nm, (e, m) in q.items():
                    r[f"{nm}_ge"], r[f"{nm}_gm"] = gender(e, m, c, (s, t))
                for nm, (e, m) in q.items():
                    se = sorted(pool, key=lambda v: -float(e @ cand[(v, c)][0]))
                    sm = sorted(pool, key=lambda v: float(np.linalg.norm(m - cand[(v, c)][1])))
                    r[f"{nm}_e_rank"], r[f"{nm}_m_rank"] = se.index(t) + 1, sm.index(t) + 1
                rows.append(r)

    def agg(rs: list) -> dict:
        o = {"n": len(rs)}
        for nm in ("out", "floor"):
            for kind in ("e", "m"):
                rk = np.array([r[f"{nm}_{kind}_rank"] for r in rs])
                o[f"{nm}_{kind}_top1"], o[f"{nm}_{kind}_mrr"] = round(float((rk == 1).mean()), 3), round(float((1.0 / rk).mean()), 3)
        return o
    gsum = {k: [round(float(np.mean([g[0] for g in v])), 3), round(float(np.mean([g[1] for g in v])), 3)] for k, v in gref.items()} | {nm: [round(float(np.mean([r[f"{nm}_ge"] for r in rows])), 3), round(float(np.mean([r[f"{nm}_gm"] for r in rows])), 3)] for nm in ("out", "floor")}
    gsum["out_female_side"] = [round(float(np.mean([r["out_ge"] > 0 for r in rows])), 3), round(float(np.mean([r["out_gm"] > 0 for r in rows])), 3)]
    gsum["by_cat_out_ge"] = {c: round(float(np.mean([r["out_ge"] for r in rows if r["cat"] == c])), 3) for c in CATS}
    res = {"name": name, "sys": opt.get("sys", E.SYS), "pool": len(pool), "gender": gsum, "chance_top1": round(1 / len(pool), 3), "all": agg(rows), "by_cat": {c: agg([r for r in rows if r["cat"] == c]) for c in CATS}, "rows": rows}
    (ROOT / f"results/snv_p110/nv2_{name}.json").write_text(json.dumps(res, indent=1))
    print("全体", json.dumps(res["all"]), "候補", len(pool), flush=True)
    print("女声らしさ(埋め込み / 包絡の平均・正 = 女声の側)", json.dumps(gsum, ensure_ascii=False), flush=True)
    for c in CATS:
        print(c, json.dumps(res["by_cat"][c]), flush=True)


if __name__ == "__main__":
    main()

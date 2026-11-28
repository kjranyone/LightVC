"""段 P110(results/snv_p110/prereg.yaml・学習なし): 音素の無い発声の変換の物差し。合成の男声(検証の 6 声)の 7 種の発声 + 通常の発話を、2 本ずつ在る女声(ef000 = 検証・ef001・ef002 = 学習の声)へ変換する。
量: 話者の埋め込み(ECAPA)と包絡の平均(c1〜c24・有声)で、出力が「源そのまま」(床)から「目標の同じ種類の別の発声」(天井)へどれだけ進んだか(進み = (出力 − 床) / (天井 − 床))・源への近さ・周期性の行の目標との差。
    cd training && SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p110_nv.py <名前> [sys=<系>] [k=0]
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
SRC = ["em594", "em595", "em596", "em597", "em598", "em599"]
TGT = ["ef000", "ef001", "ef002"]
CATS = ["moan", "scream", "pant", "cry", "laugh", "sigh", "mixed", "speech"]


def main() -> None:
    name = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    k = int(opt.get("k", 0))
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": EV.DEV})

    def emb(w: np.ndarray) -> np.ndarray:
        e = ec.encode_batch(torch.from_numpy(to16(np.asarray(w, np.float64), EV.SR).astype(np.float32))[None].to(EV.DEV))[0, 0].detach().cpu().numpy()
        return e / np.linalg.norm(e)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    wav = lambda v, c, j: lw(ROOT / (f"data/snv_e1/{v}/t0{5 + j}.wav" if c == "speech" else f"data/snv_e1nv/{v}/{c}_{j}.wav"))

    def stat(u: dict) -> tuple[np.ndarray, np.ndarray]:
        v = u["f0"] > 0
        return u["y"][1:25][:, v].mean(1), np.median(u["y"][25:27][:, v], 1)
    S = EV.Ens(opt.get("sys", E.SYS), EV.load_gen())
    rnv = opt.get("refs", "speech") == "nv"
    st = {t: S.structure("t" + t, [PB.utt(lw(ROOT / f"data/snv_e1/{t}/t0{j}.wav")) for j in (0, 1, 2)] + ([PB.utt(wav(t, c, 1)) for c in CATS if c != "speech"] if rnv else [])) for t in TGT}
    tg = {(t, c, j): (lambda x: (emb(x),) + stat(PB.utt(x)))(wav(t, c, j)) for t in TGT for c in CATS for j in (0, 1)}
    rows = []
    for s in SRC:
        enr = [PB.utt(lw(ROOT / f"data/snv_e1/{s}/t0{j}.wav")) for j in (0, 1, 2)]
        for c in CATS:
            x = wav(s, c, 0)
            u = PB.utt(x)
            es, ms, ps = (emb(x),) + stat(u)
            pu = S.parts[0].prep(u)
            vv = pu["f0"] > 0
            for t in TGT:
                w, yo = S.convert(u, enr, st[t], k, with_y=True)
                eo = emb(np.asarray(w))
                y = yo[0].detach().cpu().numpy()
                mo, po = y[1:25][:, vv].mean(1), np.median(y[25:27][:, vv], 1)
                e0, m0, p0 = tg[(t, c, 0)]
                e1, m1, _ = tg[(t, c, 1)]
                d = lambda a, b: float(np.linalg.norm(a - b))
                cc = float(np.corrcoef(y[0], pu["y"][0])[0, 1])
                rows.append({"src": s, "cat": c, "tgt": t, "c0corr": cc, "e_floor": float(es @ e0), "e_out": float(eo @ e0), "e_ceil": float(e1 @ e0), "e_src": float(eo @ es),
                             "m_floor": d(ms, m0), "m_out": d(mo, m0), "m_ceil": d(m1, m0), "m_src": d(mo, ms), "per_out": po.tolist(), "per_tgt": p0.tolist(), "per_src": ps.tolist()})
    def agg(rs: list) -> dict:
        g = lambda key: float(np.mean([r[key] for r in rs]))
        ef, eo, ecl, mf, mo, mc = g("e_floor"), g("e_out"), g("e_ceil"), g("m_floor"), g("m_out"), g("m_ceil")
        return {"n": len(rs), "e_floor": round(ef, 3), "e_out": round(eo, 3), "e_ceil": round(ecl, 3), "e_prog": round((eo - ef) / max(1e-6, ecl - ef), 2), "e_src": round(g("e_src"), 3),
                "m_floor": round(mf, 3), "m_out": round(mo, 3), "m_ceil": round(mc, 3), "m_prog": round((mf - mo) / max(1e-6, mf - mc), 2), "m_src": round(g("m_src"), 3), "c0corr": round(g("c0corr"), 3),
                "per01_out": np.round(np.mean([r["per_out"] for r in rs], 0), 2).tolist(), "per01_tgt": np.round(np.mean([r["per_tgt"] for r in rs], 0), 2).tolist(), "per01_src": np.round(np.mean([r["per_src"] for r in rs], 0), 2).tolist()}
    res = {"name": name, "k": k, "refs": opt.get("refs", "speech"), "sys": opt.get("sys", E.SYS), "by_cat": {c: agg([r for r in rows if r["cat"] == c]) for c in CATS}, "nv_all": agg([r for r in rows if r["cat"] != "speech"]),
           "nv_heldout_tgt": agg([r for r in rows if r["cat"] != "speech" and r["tgt"] == "ef000"]), "rows": rows}
    (ROOT / f"results/snv_p110/nv_{name}.json").write_text(json.dumps(res, indent=1))
    for c in CATS:
        print(c, json.dumps(res["by_cat"][c]), flush=True)
    print("NV 全体", json.dumps(res["nv_all"]))
    print("NV・検証の目標だけ", json.dumps(res["nv_heldout_tgt"]))


if __name__ == "__main__":
    main()

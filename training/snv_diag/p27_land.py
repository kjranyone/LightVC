"""段 P27 step a(results/snv_p27/prereg.yaml・学習なし・ASR なし): 着地(M = ECAPA + LDA が A0F の 24 声の中で目標を 1 位に当てる率)を、源の族ごとに床・天井と並べて測る。
源の組: valf = E1 の検証の女声 × t05・t06(加工なし)/ x16f = 同じ声と発話の擬似の操作者の版(data/snv_p_feat_x16 = M29 の族)/ valm = 検証の男声 × t05〜t08 /
    org_g0・org_w = data/snv_d5 の合成の男声とその生成した低い声 / cv = devA の先頭 n 人 × 2 発話。目標 = 割り当ての 2 声(A0F の先頭 4 声を回す)× k。
    self = A0F の 24 声の j 3〜5 を源に、同じ声の j 0〜2 を登録と構造に(自己の reamp・目標 = 同じ声・copy は同じ発話の本当の包絡 = 理想)。
系: copy = 写しの描画で高さだけ k 半音上げる(床)/ ceil = 目標の j 3〜5 の写しで、高さの中央値だけ源 × 2^(k/12) に(天井)/ snv_p1_eval.System の名前。
    名前の後ろに +cp = C の高さの入力だけを目標の声域へ(登録の高さの中央値 → 目標の参照の中央値・描画の高さは源 + k のまま)/ +w120 など = 源と登録の包絡を先に一律の伸縮 α で。
    +hg164 など = C の出力の c13〜c24 の、目標の参照の平均からのずれを g 倍(snv_p1_eval.System の +hgNNN・段 P27 step b0)。
    +ms100 など = C の出力の有声のフレームに、登録の平均の差(目標 − 源)の定数倍を足す(snv_p1_eval.System の +msNNN・段 P29 step b0)。
行: set・spk・src・valid・tgt・k・m(1 位が目標か)・top(1 位の声)・ct(素の埋め込みの目標への余弦)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p27_land.py copy ceil snv_p24_c snv_p24_c+cp [sets=valf,x16f,valm,org_g0,org_w,cv] [ks=0,7] [n=64] [out=results/snv_p27/land.json] [redo=1]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvoc as N
import probe_rig as G
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = G.ROOT
DEV = EV.DEV
SPLIT = "results/snv_p24_c/split.json"


def build(names: list[str], n: int) -> list[dict]:
    val = json.loads((ROOT / SPLIT).read_text())["val"]
    e1 = lambda v, t: T.load_utt(T.FEAT / f"e1/{v}/{t}.npz")
    x16 = lambda v, t: T.load_utt(ROOT / f"data/snv_p_feat_x16/e1/{v}/{t}.npz")
    ev = lambda nm, vs, ts, ld: [{"set": nm, "spk": v, "idx": i, "src": f"{v}|{t}", "u": ld(v, t), "enr": [ld(v, o) for o in ("t00", "t01", "t02")], "valid": True} for i, v in enumerate(vs) for t in ts]
    ef, em = [v for v in val if v.startswith("ef")], [v for v in val if v.startswith("em")]
    mk = {"valf": lambda: ev("valf", ef, ("t05", "t06"), e1), "x16f": lambda: ev("x16f", ef, ("t05", "t06"), x16), "valm": lambda: ev("valm", em, ("t05", "t06", "t07", "t08"), e1),
          "x16m": lambda: ev("x16m", em, ("t05", "t06", "t07", "t08"), x16), "cv": lambda: G.cv_items(n)} | {f"org_{k}": (lambda k=k: G.org_items(k)) for k in G.ORG}
    mk["self"] = lambda: [{"set": "self", "spk": v, "idx": i, "src": f"{v}|{j}", "u": EV.utt(f"a0/{v}_{j}"), "enr": [EV.utt(f"a0/{v}_{o}") for o in (0, 1, 2)], "valid": True} for i, v in enumerate(G.targets()[0]) for j in (3, 4, 5)]
    return [it for nm in names for it in mk[nm]()]


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for st in dict.fromkeys(r["set"] for r in rows):
        for k in dict.fromkeys(r["k"] for r in rows):
            R = [r for r in rows if r["set"] == st and r["k"] == k and r["valid"]]
            if R:
                out[f"{st}_k{k}"] = {"n": len(R), "M": round(float(np.mean([r["m"] for r in R])), 3), "ct": round(float(np.mean([r["ct"] for r in R])), 3)}
    return out


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a]
    n, sets = int(opt.get("n", 64)), opt.get("sets", "valf,x16f,valm,org_g0,org_w,cv").split(",")
    ks = tuple(int(x) for x in opt.get("ks", "0,7").split(","))
    out = ROOT / opt.get("out", "results/snv_p27/land.json")
    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p27/prereg.yaml", "n": n, "sets": sets, "ks": list(ks), "systems": {}, "rows": {}}
    tags = [t for t in tags if t not in res["systems"] or opt.get("redo") == "1"]
    if not tags:
        print(json.dumps(res["systems"], ensure_ascii=False, indent=1))
        return 0
    from snv_a0_measure import load, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(trim(y16)).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    lda = lambda e: proj(e, (Tz["mu"], Tz["scalings"]))
    gen = EV.load_gen()

    @torch.no_grad()
    def render(y: np.ndarray, f0: np.ndarray) -> np.ndarray:
        v = (f0 > 0).astype(np.float32)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
        cond = torch.from_numpy(np.concatenate([y[:25], lf[None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0))[None].to(DEV)
        ex = TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(DEV), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
        return gen(cond, ex)[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    all_t, tg4 = G.targets()
    Et = {v: emb(np.concatenate([to16(*load(EV.A0 / f"wav/{v}_{j}.wav")).astype(np.float32) for j in (0, 1, 2)])) for v in all_t}
    El = {v: lda(Et[v]) for v in all_t}
    tv = all_t if "self" in sets else tg4
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tv}
    tceil = {v: [EV.utt(f"a0/{v}_{j}") for j in (3, 4, 5)] for v in tv}
    med = lambda us: float(np.median(np.concatenate([x["f0"][x["f0"] > 0] for x in us])))
    med_t = {v: med(trefs[v]) for v in tv}
    items = build(sets, n)
    print("items", {st: sum(i["set"] == st for i in items) for st in sets}, "源の高さの中央値", {st: round(float(np.median([med([i["u"]]) for i in items if i["set"] == st]))) for st in sets}, flush=True)
    for tag in tags:
        base, _, mod = tag.partition("+")
        assert mod in ("", "cp") or mod[0] == "w" or mod[:2] in ("hg", "ml", "at", "ms"), mod
        al = float(mod[1:]) / 100 if mod[:1] == "w" else None
        s = None if tag in ("copy", "ceil") else EV.System(tag if mod[:2] in ("hg", "ml", "at", "ms") else base, gen)
        rows, wenr = [], {}
        for it in items:
            u, enr = it["u"], it["enr"]
            if al is not None:
                key = (it["set"], it["spk"])
                if key not in wenr:
                    wenr[key] = [EV.warp_utt(e, al) for e in enr]
                u, enr = EV.warp_utt(u, al), wenr[key]
            ms = med(enr)
            for k in ks:
                wc = render(u["y"], u["f0"] * 2 ** (k / 12)) if tag == "copy" else None
                for v in ((it["spk"],) if it["set"] == "self" else (tg4[it["idx"] % 4], tg4[(it["idx"] + 2) % 4])):
                    if tag == "copy":
                        w = wc
                    elif tag == "ceil":
                        ut = tceil[v][it["idx"] % 3]
                        w = render(ut["y"], ut["f0"] * (med([u]) * 2 ** (k / 12) / med([ut])))
                    else:
                        w = s.convert(u, enr, s.structure(f"t|{v}", trefs[v]), k, dlf_c=float(np.log(med_t[v] / ms)) if mod == "cp" else None)
                    e = emb(w16(w))
                    el = lda(e)
                    top = max(all_t, key=lambda t: float(el @ El[t]))
                    rows.append({"set": it["set"], "spk": it["spk"], "src": it["src"], "valid": bool(it["valid"]), "tgt": v, "k": k, "m": int(top == v), "top": top, "ct": round(float(e @ Et[v]), 4)})
        res["systems"][tag] = summarize(rows)
        res["rows"][tag] = rows
        print(tag, json.dumps(res["systems"][tag], ensure_ascii=False), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        del s
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

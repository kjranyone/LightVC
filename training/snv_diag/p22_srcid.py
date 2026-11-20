"""段 P22(results/snv_p22/prereg.yaml・学習なし・ASR なし): 変換の出力に源の声が残っているかを、源の話者を当てる率で直に測る。
出力の ECAPA(素の埋め込み・LDA なし)を、源の組の全話者の登録(元の音)と A0F の目標 24 声(j 0〜2 の元の音)を合わせた候補の中で 1 位に当て、4 つに分ける:
  tgt = 目標 / src = 源の話者 / other_tgt = 別の目標の声 / other_src = 源の組の別の話者。並べて cs − ct = 源の登録への余弦 − 目標への余弦。
行には top = 最も近い目標の声・m = 成績表と同じ M(ECAPA + LDA が目標 24 声の中で目標を 1 位に当てるか)も残す。
源の組: cv = devA の先頭 n 人 × s0・s1(登録 = e0〜e7 の元の音)/ org_g0 = data/snv_d5 の対照の声 24(源 t01・t02・登録 = 残りの 2 発話)。割り当ての目標 2 声 × k(既定 0・7・ks=12 などで替える)。
系: copy = 写しの描画で高さだけ k 半音上げる(包絡が源のままのときの上限)/ ceil = 目標の j 3〜5 の写しで、高さの中央値だけ源 × 2^(k/12) に合わせる(包絡が目標そのもの = 着地の天井・snv_p1c_cal の COPY と同じ)/ anchor = 一律のずらしの錨(snv_dev_eval と同じ)/ snv_p1_eval.System の名前。
系の名前の後ろに +cp = C の高さの入力だけを目標の声域へ(操作者の登録の高さの中央値 → 目標の参照の中央値・描画の高さは操作者 + k のまま・System.convert の dlf_c)/
+w115 = 源と操作者の登録の包絡を先に一律の伸縮 1.15 で(snv_p1_eval.warp_utt・高さは元のまま)。どちらも学習なしの探り。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p22_srcid.py copy anchor snv_p3_x snv_p8_wst snv_p16_x snv_p19_x snv_p20_q snv_p21_x [n=64] [sets=cv,org_g0] [ks=0,7] [out=results/snv_p22/srcid.json] [redo=1]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvoc as N
import probe_rig as G
import snv_dev_eval as DE
import snv_p0c_anchor as AN
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = G.ROOT
DEV = EV.DEV
KS = (0, 7)
CATS = ("tgt", "src", "other_tgt", "other_src")


def ci(per: dict) -> list[float]:
    sp = sorted(per)
    m = np.array([np.mean(per[s]) for s in sp])
    g = np.random.default_rng(0)
    bs = [float(np.mean(g.choice(m, len(m)))) for _ in range(2000)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for st in dict.fromkeys(r["set"] for r in rows):
        for k in dict.fromkeys(r["k"] for r in rows):
            R = [r for r in rows if r["set"] == st and r["k"] == k and r["valid"]]
            if not R:
                continue
            per: dict = {}
            mg: dict = {}
            for r in R:
                per.setdefault(r["spk"], []).append(float(r["cat"] == "src"))
                mg.setdefault(r["spk"], []).append(r["cs"] - r["ct"])
            out[f"{st}_k{k}"] = {"n": len(R)} | {c: round(float(np.mean([r["cat"] == c for r in R])), 3) for c in CATS} | {"src_ci": ci(per), "cs_minus_ct": round(float(np.mean([np.mean(v) for v in mg.values()])), 3), "margin_ci": ci(mg), "M": round(float(np.mean([r["m"] for r in R])), 3)}
    return out


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a]
    n, sets = int(opt.get("n", 64)), opt.get("sets", "cv,org_g0").split(",")
    ks = tuple(int(x) for x in opt.get("ks", ",".join(str(k) for k in KS)).split(","))
    out = ROOT / opt.get("out", "results/snv_p22/srcid.json")
    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p22/prereg.yaml", "n": n, "sets": sets, "systems": {}, "rows": {}}
    tags = [t for t in tags if t not in res["systems"] or opt.get("redo") == "1"]
    if not tags:
        print(json.dumps(res["systems"], ensure_ascii=False, indent=1))
        return 0
    from snv_a0_measure import load, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    lw = lambda p: w16(librosa.load(p, sr=EV.SR, mono=True)[0])
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
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tg4}
    mu_t = {v: EV.mu_of(trefs[v])[0].cpu().numpy() for v in tg4}
    tceil = {v: [EV.utt(f"a0/{v}_{j}") for j in (3, 4, 5)] for v in tg4}
    med_t = {v: float(np.median(np.concatenate([x["f0"][x["f0"] > 0] for x in trefs[v]]))) for v in tg4}
    items, Es = [], {}
    if "cv" in sets:
        dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())["ref"]
        for it in DE.cv_items("devA", n):
            items.append(it | {"set": "cv", "valid": dv[it["src"]]["text_vs_orig"] <= 0.1})
            if ("cv", it["spk"]) not in Es:
                Es[("cv", it["spk"])] = emb(np.concatenate([lw(e["wav_path"]) for e in it["enr"]]))
    if "org_g0" in sets:
        for it in G.org_items("g0"):
            items.append(it)
            if ("org_g0", it["spk"]) not in Es:
                t = it["src"].split("|")[1]
                Es[("org_g0", it["spk"])] = emb(np.concatenate([lw(ROOT / f"data/snv_d5/{it['spk']}/{o}.wav") for o in ("t00", "t01", "t02") if o != t]))
    pool = {st: [k for k in Es if k[0] == st] for st in sets}
    print("items", {st: sum(i["set"] == st for i in items) for st in sets}, "候補", {st: len(p) + len(all_t) for st, p in pool.items()}, flush=True)
    for tag in tags:
        base, _, mod = tag.partition("+")
        assert mod in ("", "cp", "w115"), mod
        s = None if tag in ("copy", "anchor", "ceil") else EV.System(base, gen)
        rows, wenr = [], {}
        for it in items:
            u, enr = it["u"], it["enr"]
            if mod == "w115":
                if (it["set"], it["spk"]) not in wenr:
                    wenr[(it["set"], it["spk"])] = [EV.warp_utt(e, 1.15) for e in enr]
                u, enr = EV.warp_utt(u, 1.15), wenr[(it["set"], it["spk"])]
            ms = float(np.median(np.concatenate([e["f0"][e["f0"] > 0] for e in enr])))
            for k in ks:
                wc = render(u["y"], u["f0"] * 2 ** (k / 12)) if tag == "copy" else None
                for v in (tg4[it["idx"] % 4], tg4[(it["idx"] + 2) % 4]):
                    if tag == "copy":
                        w = wc
                    elif tag == "ceil":
                        ut = tceil[v][it["idx"] % 3]
                        w = render(ut["y"], ut["f0"] * float(np.median(u["f0"][u["f0"] > 0]) * 2 ** (k / 12) / np.median(ut["f0"][ut["f0"] > 0])))
                    elif tag == "anchor":
                        env = np.concatenate([u["y"][:1] * 10 + T.PA.C0_SIL, u["y"][1:25] * 10], 0)
                        ew = AN.warp_env(env, AN.fit_alpha(EV.mu_of(enr)[0].cpu().numpy(), mu_t[v]))
                        w = render(np.concatenate([(ew[:1] - T.PA.C0_SIL) / 10, ew[1:25] / 10, u["y"][25:29]], 0).astype(np.float32), u["f0"] * 2 ** (k / 12))
                    else:
                        w = s.convert(u, enr, s.structure(f"t|{v}", trefs[v]), k, dlf_c=float(np.log(med_t[v] / ms)) if mod == "cp" else None)
                    e = emb(w16(w))
                    cs = {key: float(e @ x) for key in pool[it["set"]] for x in (Es[key],)}
                    ct = {t: float(e @ Et[t]) for t in all_t}
                    bs, bt = max(cs, key=cs.get), max(ct, key=ct.get)
                    if cs[bs] > ct[bt]:
                        cat = "src" if bs[1] == it["spk"] else "other_src"
                    else:
                        cat = "tgt" if bt == v else "other_tgt"
                    el = lda(e)
                    rows.append({"set": it["set"], "spk": it["spk"], "src": it["src"], "valid": bool(it["valid"]), "tgt": v, "k": k, "cat": cat, "cs": round(cs[(it["set"], it["spk"])], 4), "ct": round(ct[v], 4), "top": bt,
                                 "m": int(max(all_t, key=lambda t: float(el @ El[t])) == v)})
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

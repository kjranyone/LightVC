"""段 P29 step a の a3(results/snv_p29/prereg.yaml・学習なし・ASR なし): 同じ文を読んだ別の声を天井に置いて、声を跨ぐ出力の平均・動き・目標への余弦を測る。
組 = 段 D(data/snv_d・同じ文を 8 声が読む)の検証の声どうしで、同じ文を読んだ (源 u, 目標 v = 検証の女声)。源の登録 = u の E1 の t00〜t02・目標の参照 = v の E1 の t00〜t02・k 0。
天井の発話 = v が同じ文を読んだ発話。評価の天井にだけ使う(時刻は揃えない・発話の有声のフレームの統計だけを比べる・学習と対の回帰に使わない)。
量(帯 lo = c1〜c12・hi = c13〜c24):
  L = 平均の着地 <x̄ − s̄, V̂> / |V̂|²(x̄ = 発話の有声のフレームの平均・s̄ = 源の発話・V̂ = 目標の参照の平均 − 源の登録の平均)。x = 出力(L)と天井の発話(Lt)。
  sr = 時間方向の標準偏差の比(出力 / 天井の発話・係数ごとの比の平均)。sr_src = 源の発話 / 天井の発話。
  ct = 素の ECAPA の目標への余弦(目標の埋め込み = v の E1 の t00〜t02 の音): 出力・floor(源の写しの描画)・ceil(天井の発話の写しの描画・高さの中央値だけ源に合わせる)・self(天井の発話を v の構造で reamp)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p29_same.py snv_p24_c snv_p8_wst snv_i10_c_pr [out=results/snv_p29/same.json] [redo=1]
sp=1(a3c)= 自己(天井の発話を v の構造で reamp)を、高さの中央値だけ源に合わせて描いた余弦 ct_self_sp だけを測る(床と天井は same.json の ref を読む・出力 = results/snv_p29/same_sp.json)。
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
BANDS = {"lo": (1, 13), "hi": (13, 25)}
SPLIT = "results/snv_p24_c/split.json"


def pairs() -> list[dict]:
    val = set(json.loads((ROOT / SPLIT).read_text())["val"])
    by: dict = {}
    for v in json.loads((ROOT / "data/snv_d/manifest.json").read_text())["voices"]:
        if v["voice"] in val:
            for x in v["files"]:
                by.setdefault(x["content"], []).append((v["voice"], x["s"]))
    return [{"w": w, "u": u, "su": su, "v": v, "sv": sv} for w, ps in sorted(by.items()) for u, su in ps for v, sv in ps if u != v and v.startswith("ef")]


def vmean(us: list[dict], lo: int, hi: int) -> np.ndarray:
    return np.concatenate([u["y"][lo:hi, u["v"] > 0] for u in us], 1).mean(1)


def boot(num: dict, den: dict, n: int = 2000) -> list[float]:
    ks, g = sorted(num), np.random.default_rng(0)
    a, b = np.array([num[k] for k in ks]), np.array([den[k] for k in ks])
    bs = [float(a[i].sum() / b[i].sum()) for i in (g.integers(0, len(ks), len(ks)) for _ in range(n))]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def summarize(rows: list[dict], ref: list[dict]) -> dict:
    out: dict = {}
    for sx in ("ef", "em"):
        R = [(r, q) for r, q in zip(rows, ref) if r["u"].startswith(sx)]
        if not R:
            continue
        o: dict = {"n": len(R), "n_src": len({r["u"] for r, _ in R})}
        for b in BANDS:
            num, den = {}, {}
            for r, q in R:
                num[r["u"]] = num.get(r["u"], 0.0) + r[f"L_{b}"]
                den[r["u"]] = den.get(r["u"], 0.0) + q[f"Lt_{b}"]
            o[f"L_{b}(出力 / 天井の発話)"] = [round(float(np.mean([r[f"L_{b}"] for r, _ in R])), 3), round(float(np.mean([q[f"Lt_{b}"] for _, q in R])), 3)]
            o[f"L_{b} の比"] = [round(sum(num.values()) / sum(den.values()), 3), boot(num, den)]
            o[f"sr_{b}(出力 / 源)"] = [round(float(np.mean([r[f"sr_{b}"] for r, _ in R])), 3), round(float(np.mean([q[f"sr_src_{b}"] for _, q in R])), 3)]
        ct, fl, ce, sf = (float(np.mean(x)) for x in ([r["ct"] for r, _ in R], [q["ct_floor"] for _, q in R], [q["ct_ceil"] for _, q in R], [r["ct_self"] for r, _ in R]))
        num, den = {}, {}
        for r, q in R:
            num[r["u"]] = num.get(r["u"], 0.0) + r["ct"] - q["ct_floor"]
            den[r["u"]] = den.get(r["u"], 0.0) + q["ct_ceil"] - q["ct_floor"]
        o["ct(床 / 出力 / 自己 / 天井)"] = [round(fl, 3), round(ct, 3), round(sf, 3), round(ce, 3)]
        o["g_ct"] = [round((ct - fl) / (ce - fl), 3), boot(num, den)]
        o["g_ct(自己)"] = round((sf - fl) / (ce - fl), 3)
        out[sx] = o
    return out


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a]
    out = ROOT / ("results/snv_p29/same.json" if opt.get("sp") == "1" else opt.get("out", "results/snv_p29/same.json"))
    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p29/prereg.yaml(a3)", "systems": {}, "rows": {}}
    tags = [t for t in tags if t not in res["systems"] or opt.get("redo") == "1" or opt.get("sp") == "1"]
    if not tags:
        print(json.dumps(res["systems"], ensure_ascii=False, indent=1))
        return 0
    from snv_a0_measure import load, to16
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(trim(y16)).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    gen = EV.load_gen()

    @torch.no_grad()
    def render(y: np.ndarray, f0: np.ndarray) -> np.ndarray:
        v = (f0 > 0).astype(np.float32)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
        cond = torch.from_numpy(np.concatenate([y[:25], lf[None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0))[None].to(DEV)
        ex = TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(DEV), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
        return gen(cond, ex)[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    P = pairs()
    voices = sorted({p["u"] for p in P} | {p["v"] for p in P})
    enr = {v: [T.load_utt(T.FEAT / f"e1/{v}/{t}.npz") for t in ("t00", "t01", "t02")] for v in voices}
    du = {(v, s): T.load_utt(T.FEAT / f"d/{v}/{s}.npz") for p in P for v, s in ((p["u"], p["su"]), (p["v"], p["sv"]))}
    Et = {v: emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/{t}.wav")).astype(np.float32) for t in ("t00", "t01", "t02")])) for v in sorted({p["v"] for p in P})}
    med = lambda u: float(np.median(u["f0"][u["f0"] > 0]))
    sd = lambda y, m, lo, hi: y[lo:hi, m].std(1)
    print("pairs", len(P), "女声 → 女声", sum(p["u"].startswith("ef") for p in P), "男声 → 女声", sum(p["u"].startswith("em") for p in P), flush=True)
    if "ref" not in res:
        fl_cache, ce_cache, ref = {}, {}, []
        for p in P:
            s, t = du[(p["u"], p["su"])], du[(p["v"], p["sv"])]
            ks, kc = (p["u"], p["su"]), (p["u"], p["su"], p["v"], p["sv"])
            if ks not in fl_cache:
                fl_cache[ks] = emb(w16(render(s["y"], s["f0"])))
            if kc not in ce_cache:
                ce_cache[kc] = emb(w16(render(t["y"], t["f0"] * (med(s) / med(t)))))
            q = {"ct_floor": round(float(fl_cache[ks] @ Et[p["v"]]), 4), "ct_ceil": round(float(ce_cache[kc] @ Et[p["v"]]), 4)}
            for b, (lo, hi) in BANDS.items():
                V = vmean(enr[p["v"]], lo, hi) - vmean(enr[p["u"]], lo, hi)
                q[f"Lt_{b}"] = round(float((vmean([t], lo, hi) - vmean([s], lo, hi)) @ V / max(float(V @ V), 1e-12)), 4)
                q[f"sr_src_{b}"] = round(float(np.mean(sd(s["y"], s["v"] > 0, lo, hi) / np.maximum(sd(t["y"], t["v"] > 0, lo, hi), 1e-6))), 4)
            ref.append(q)
        res["ref"] = ref
    if opt.get("sp") == "1":
        sp = {"prereg": "results/snv_p29/prereg.yaml(a3c)", "systems": {}, "rows": {}}
        for tag in tags:
            s_ = EV.System(tag, gen)
            rows = []
            for p, q in zip(P, res["ref"]):
                s, t = du[(p["u"], p["su"])], du[(p["v"], p["sv"])]
                w = s_.convert(t, enr[p["v"]], s_.structure(f"t|{p['v']}", enr[p["v"]]), 12 * float(np.log2(med(s) / med(t))))
                rows.append({"u": p["u"], "v": p["v"], "ct_self_sp": round(float(emb(w16(w)) @ Et[p["v"]]), 4), "ct_floor": q["ct_floor"], "ct_ceil": q["ct_ceil"]})
            sp["rows"][tag] = rows
            sp["systems"][tag] = {sx: {"n": sum(r["u"].startswith(sx) for r in rows), "ct_self_sp": round(float(np.mean([r["ct_self_sp"] for r in rows if r["u"].startswith(sx)])), 4)} for sx in ("ef", "em")}
            print(tag, json.dumps(sp["systems"][tag], ensure_ascii=False), flush=True)
            del s_
        (ROOT / opt.get("out", "results/snv_p29/same_sp.json")).write_text(json.dumps(sp, ensure_ascii=False, indent=1))
        return 0
    for tag in tags:
        s_ = EV.System(tag, gen)
        rows, self_cache = [], {}
        for p in P:
            s, t = du[(p["u"], p["su"])], du[(p["v"], p["sv"])]
            S = s_.structure(f"t|{p['v']}", enr[p["v"]])
            w, yo = s_.convert(s, enr[p["u"]], S, 0, with_y=True)
            yo = yo[0].cpu().numpy()
            vm = s["v"][:yo.shape[1]] > 0
            kt = (p["v"], p["sv"])
            if kt not in self_cache:
                self_cache[kt] = round(float(emb(w16(s_.convert(t, enr[p["v"]], S, 0))) @ Et[p["v"]]), 4)
            r = {"w": p["w"], "u": p["u"], "v": p["v"], "ct": round(float(emb(w16(w)) @ Et[p["v"]]), 4), "ct_self": self_cache[kt]}
            for b, (lo, hi) in BANDS.items():
                V = vmean(enr[p["v"]], lo, hi) - vmean(enr[p["u"]], lo, hi)
                r[f"L_{b}"] = round(float((yo[lo:hi, vm].mean(1) - vmean([s], lo, hi)) @ V / max(float(V @ V), 1e-12)), 4)
                r[f"sr_{b}"] = round(float(np.mean(sd(yo, vm, lo, hi) / np.maximum(sd(t["y"], t["v"] > 0, lo, hi), 1e-6))), 4)
            rows.append(r)
        res["systems"][tag] = summarize(rows, res["ref"])
        res["rows"][tag] = rows
        print(tag, json.dumps(res["systems"][tag], ensure_ascii=False), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        del s_
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

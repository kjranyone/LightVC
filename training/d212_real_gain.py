"""D212(学習なし・diag_rr16c GAP の帰属): 未知の 56 人の同じ文・同じ記憶(クローンの手本)・同じ入力で、目標をクローン(xc)と A の実録音(xr)に替えて記憶の効きを測る。
gain = L1(差し替え 4 人の平均) − L1(正しい)・dcorr = corr(R − 表, 目標 − 表) の正しい − 差し替え(有声のフレーム・d193 と同じ)。表は A の手本のまま。
    uv run python d212_real_gain.py --tags diag_rr14c diag_rr16c diag_rr8c --base diag_rr14c --main diag_rr16c --out ../results/conv_p0/d212.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent
NSW = 4


def build_items(st, ds: RRm.Pairs, its: list) -> list:
    out = []
    for i, it in enumerate(its):
        A = it["target"]
        ref = lambda x: [st.f[("real_fem", x["target"], s)] for s in x["ref_stems"] if ("real_fem", x["target"], s) in st.f]
        rA = ref(it)
        sw = [ref(its[(i + k) % len(its)]) for k in range(1, NSW + 1)]
        b = st.f[("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC")]
        row = {"A": A}
        for kind, a in (("xc", st.f[("realx_clone", "clone", it["clone"])]), ("xr", st.f[("real_fem", A, it["stem"])])):
            row[kind] = [ds.cross_item(random.Random(0), a, b, A, it["spk"], m, kind, tab_refs=rA) for m in [rA] + sw]
        out.append(row)
    return out


@torch.no_grad()
def score(tag: str, rows: list, dev: str) -> dict:
    ck = torch.load(ROOT / "results" / tag / "last.pt", map_location="cpu", weights_only=False)
    RRm.CFG["align"] = ck["args"].get("align", "rr")
    net = RRm.build(ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["net"])
    res: dict = {"xc": [], "xr": []}
    for r in rows:
        for kind in ("xc", "xr"):
            vals = []
            for d in r[kind]:
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in RRm.collate([d]).items()}
                env, _, _ = net(b["q"], b["tab"], net.memory(b["rp"], b["rv"]), b["rm"])
                v = b["yv"][0] & b["tm"][0]
                e, y, t = env[0][:, v], b["y"][0][:, v], b["tab"][0][:, v]
                dc = float(torch.corrcoef(torch.stack([(e - t).flatten(), (y - t).flatten()]))[0, 1])
                vals.append((float((e - y).abs().mean()), float((t - y).abs().mean()), dc))
            l1r, l1t, dcr = vals[0]
            l1s = float(np.mean([x[0] for x in vals[1:]])); dcs = float(np.mean([x[2] for x in vals[1:]]))
            res[kind].append({"A": r["A"], "l1": l1r, "l1_tab": l1t, "gain": l1s - l1r, "dcorr": dcr - dcs, "dcorr_right": dcr})
    del net
    torch.cuda.empty_cache()
    return res


def boot(zs: list, rng: np.random.Generator, nb: int = 5000) -> list:
    n = len(zs[0])
    bs = [[z[ii].mean() for z in zs] for ii in (rng.integers(0, n, n) for _ in range(nb))]
    return np.percentile(np.array(bs), [2.5, 97.5], axis=0).T.round(4).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = RRm.Store()
    its = [it for it in json.loads((ROOT / "data/realx/items_clone.json").read_text()) if it["val"]
           and ("realx_clone", "clone", it["clone"]) in st.f and ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC") in st.f and ("real_fem", it["target"], it["stem"]) in st.f]
    assert len({it["target"] for it in its}) == len(its)
    ds = RRm.Pairs(st, [], [], 1, 0.0, 7, fixed=True)
    rows = build_items(st, ds, its)
    res = {t: score(t, rows, "cuda") for t in a.tags}
    rng = np.random.default_rng(0)
    rep: dict = {"n": len(its), "swap_partners": NSW}
    for t, r in res.items():
        rep[t] = {}
        for kind in ("xc", "xr"):
            g = np.array([x["gain"] for x in r[kind]]); d = np.array([x["dcorr"] for x in r[kind]])
            ci = boot([g, d], rng)
            rep[t][kind] = {"l1_rr": round(float(np.mean([x["l1"] for x in r[kind]])), 4), "l1_tab": round(float(np.mean([x["l1_tab"] for x in r[kind]])), 4),
                            "dcorr_right": round(float(np.mean([x["dcorr_right"] for x in r[kind]])), 4), "gain_mem": round(float(g.mean()), 4), "gain_mem_ci95": ci[0],
                            "dcorr_mem": round(float(d.mean()), 4), "dcorr_mem_ci95": ci[1]}
    for t in a.tags:
        if t == a.base:
            continue
        blk = {}
        for kind in ("xc", "xr"):
            dg = np.array([x["gain"] - y["gain"] for x, y in zip(res[t][kind], res[a.base][kind])])
            dd = np.array([x["dcorr"] - y["dcorr"] for x, y in zip(res[t][kind], res[a.base][kind])])
            dl = np.array([x["l1"] - y["l1"] for x, y in zip(res[t][kind], res[a.base][kind])])
            ci = boot([dg, dd, dl], rng)
            blk[kind] = {"gain_mem": round(float(dg.mean()), 4), "gain_mem_ci95": ci[0], "dcorr_mem": round(float(dd.mean()), 4), "dcorr_mem_ci95": ci[1],
                         "l1_rr": round(float(dl.mean()), 4), "l1_rr_ci95": ci[2]}
        gx = np.array([x["gain"] - y["gain"] for x, y in zip(res[t]["xc"], res[a.base]["xc"])])
        gr = np.array([x["gain"] - y["gain"] for x, y in zip(res[t]["xr"], res[a.base]["xr"])])
        rb = [gr[ii].mean() / gx[ii].mean() for ii in (rng.integers(0, len(gx), len(gx)) for _ in range(5000))]
        blk["rho_gain"] = round(float(gr.mean() / gx.mean()), 3)
        blk["rho_gain_ci95"] = [round(float(q), 3) for q in np.percentile(rb, [2.5, 97.5])]
        rep[f"{t}_minus_{a.base}"] = blk
    m = rep[f"{a.main}_minus_{a.base}"]
    rep["verdict"] = "ENV_TRANSFERS" if m["xr"]["gain_mem_ci95"][0] > 0 and m["rho_gain"] >= 0.5 else "ENV_GAP"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

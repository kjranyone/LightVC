"""D193(学習なし・diag_rr13c の判定): クローンの組(男声 TTS の読み → 目標 A のクローン・記憶 = 手本と同じ A の実録音)で、複数の R を同じ項目・同じ乱数で測り、対にして比べる。
群: unseen_all(items_clone_multi の未知 12 人 + items_clone_scale の未知 unseen2)・seen24_held(items_clone_multi の既知 24 人の保留文)・seen2_held(items_clone_scale の既知 seen2 の保留文)・--seen3 で seen3_held(items_clone_full の保留文)。
各項目: gain = L1(差し替えた記憶) − L1(正しい記憶)・dcorr = corr(R − 表, 目標 − 表) の正しい記憶 − 差し替え。対の差(--tags の各 − --base)は話者単位のブートストラップ 5000 回。
事前登録 = results/diag_rr13c/prereg.yaml(fork は unseen_all の diag_rr13c − diag_rr12c)。
    uv run python d193_scale_eval.py --tags diag_rr13c diag_rr9c --base diag_rr12c --out ../results/diag_rr13c/verdict_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def run(tag: str, st, groups: dict, mult: int, dev: str) -> dict:
    ck = torch.load(ROOT / "results" / tag / "last.pt", map_location="cpu", weights_only=False)
    RRm.CFG["align"] = ck["args"].get("align", "rr")
    net = RRm.build(ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["net"])
    out = {}
    for name, its in groups.items():
        cols = []
        for swap in (False, True):
            ds = RRm.Pairs(st, [], [], mult * len(its), 0.0, 7, fixed=True, clonex=its, p_clonex=1.0, swap_mem=swap)
            if ck["args"].get("pbase"):
                import pbase as PB
                ds.pbase = PB.PBase(ROOT / "data/realx/pbase")
            dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4, collate_fn=RRm.collate)
            res = []
            with torch.no_grad():
                for b in dl:
                    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                    env, _, _ = net(b["q"], b["tab"], net.memory(b["rp"], b["rv"]), b["rm"])
                    v = b["yv"][0] & b["tm"][0]
                    e, y, t = env[0][:, v], b["y"][0][:, v], b["tab"][0][:, v]
                    dc = float(torch.corrcoef(torch.stack([(e - t).flatten(), (y - t).flatten()]))[0, 1])
                    res.append((b["A"][0], float((e - y).abs().mean()), float((t - y).abs().mean()), dc))
            cols.append(res)
        spk = np.array([r[0] for r in cols[0]])
        X = np.array([[r[1], r[2], r[3], s_[1], s_[3]] for r, s_ in zip(cols[0], cols[1])], float)
        out[name] = {"spk": spk, "l1": X[:, 0], "l1_tab": X[:, 1], "gain": X[:, 3] - X[:, 0], "dcorr": X[:, 2] - X[:, 4], "dcorr_right": X[:, 2]}
    del net
    torch.cuda.empty_cache()
    return out


def boot(spk: np.ndarray, zs: list, rng: np.random.Generator, nb: int = 5000) -> list:
    us = sorted(set(spk)); idx = {u: np.flatnonzero(spk == u) for u in us}
    bs = []
    for _ in range(nb):
        ii = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
        bs.append([z[ii].mean() for z in zs])
    return np.percentile(np.array(bs), [2.5, 97.5], axis=0).T.round(4).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mult", type=int, default=2)
    ap.add_argument("--seen2_spk", type=int, default=150, help="seen2_held は先頭のこの人数の話者の保留文だけ(記述用)")
    ap.add_argument("--lite", action="store_true", help="評価の群に要る鍵(クローン・源・目標の参照)と male_ref だけを読む(学習と同時に走らせるときのメモリ節約・結果は同じ)")
    ap.add_argument("--seen3", type=int, default=0, help="> 0 で seen3_held(items_clone_full の既知 seen3 の保留文・先頭のこの人数の話者)を足す(diag_rr16c・記述)")
    ap.add_argument("--min_gain", type=float, default=0.025)
    ap.add_argument("--min_dcorr", type=float, default=0.020)
    a = ap.parse_args()
    dev = "cuda"
    if a.lite:
        need = set()
        for f in ("items_clone_multi.json", "items_clone_scale.json", "items_clone_full.json"):
            for it in json.loads((ROOT / "data/realx" / f).read_text()):
                need |= {("realx_clone", "clone", it["clone"]), ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC")} | {("real_fem", it["target"], x) for x in it["ref_stems"]}
        st = RRm.Store(keep=lambda c, s_, t_: c == "male_ref" or (c, s_, t_) in need)
    else:
        st = RRm.Store()
    ok = lambda it: ("realx_clone", "clone", it["clone"]) in st.f and ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC") in st.f
    cm = json.loads((ROOT / "data/realx/items_clone_multi.json").read_text())
    cs = json.loads((ROOT / "data/realx/items_clone_scale.json").read_text())
    groups = {"unseen_all": [it for it in cm + cs if it["group"] in ("unseen", "unseen2") and ok(it)],
              "seen24_held": [it for it in cm if it["group"] == "seen" and it["held"] and ok(it)],
              "seen2_held": [it for it in cs if it["group"] == "seen2" and it["held"] and ok(it) and int(it["clone"][1:5]) < a.seen2_spk]}
    if a.seen3 > 0:
        cf = json.loads((ROOT / "data/realx/items_clone_full.json").read_text())
        groups["seen3_held"] = [it for it in cf if it["group"] == "seen3" and it["held"] and ok(it) and int(it["clone"][1:5]) < 500 + a.seen3]
    rep: dict = {"base": a.base, "items": {k: len(v) for k, v in groups.items()}, "mult": a.mult}
    res = {t: run(t, st, groups, a.mult, dev) for t in [a.base] + a.tags}
    rng = np.random.default_rng(0)
    for t, r in res.items():
        rep[t] = {}
        for g, d in r.items():
            ci = boot(d["spk"], [d["gain"], d["dcorr"]], rng)
            rep[t][g] = {"speakers": len(set(d["spk"])), "l1_rr": round(float(d["l1"].mean()), 4), "l1_tab": round(float(d["l1_tab"].mean()), 4),
                         "dcorr_right": round(float(d["dcorr_right"].mean()), 4), "gain_mem": round(float(d["gain"].mean()), 4), "gain_mem_ci95": ci[0],
                         "dcorr_mem": round(float(d["dcorr"].mean()), 4), "dcorr_mem_ci95": ci[1]}
    for t in a.tags:
        rep[f"{t}_minus_{a.base}"] = {}
        for g in groups:
            b0, b1 = res[a.base][g], res[t][g]
            assert (b0["spk"] == b1["spk"]).all()
            dg, dd, dl = b1["gain"] - b0["gain"], b1["dcorr"] - b0["dcorr"], b1["l1"] - b0["l1"]
            ci = boot(b0["spk"], [dg, dd, dl], rng)
            rep[f"{t}_minus_{a.base}"][g] = {"gain_mem": round(float(dg.mean()), 4), "gain_mem_ci95": ci[0], "dcorr_mem": round(float(dd.mean()), 4), "dcorr_mem_ci95": ci[1],
                                              "l1_rr": round(float(dl.mean()), 4), "l1_rr_ci95": ci[2]}
    d = rep[f"{a.tags[0]}_minus_{a.base}"]["unseen_all"]
    up = d["gain_mem_ci95"][0] > 0 and d["dcorr_mem_ci95"][0] > 0 and d["gain_mem"] >= a.min_gain and d["dcorr_mem"] >= a.min_dcorr
    flat = d["gain_mem_ci95"][0] <= 0 and d["dcorr_mem_ci95"][0] <= 0
    rep["verdict"] = "SCALE_UP" if up else ("SCALE_FLAT" if flat else "MIXED")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

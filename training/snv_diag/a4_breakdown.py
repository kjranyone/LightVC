"""A4 の走行後の切り分け(事前登録の fork「id_ok + H_table_ext」の指示 = c の音素の無い発声の崩れの型を調べる・学習なし・記録だけ)。
TEX を群(帯の平均・標準偏差・差の標準偏差・変調・平坦度)と帯域に分け、音量の差を除いた形の差も読む。
組 = A0F(評価の女 24 声・音素の無い発声 336 本と普通の発話の自己の reamp 72 本)・E1V(学習に無い E1 の検証 30 声)・E1T(学習の E1 の女 30 声・学んだ Φ と登録で当てはめ直した Φ)。
当てはめの回数(0・25・100・400・1600)で A0F を描き直し、Φ の大きさを学習の Φ と比べる。出力 = results/diag_snv_a4/breakdown.json。
    cd training && uv run --no-sync python snv_diag/a4_breakdown.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT, load, to16
from snv_a2_eval import A1, vocoder
from snv_a4_eval import N0R, cluster_ci
from snv_n0_measure import hf_share, trim
from snv_n2c_tex import GROUPS, tex_stats
from train_snv_a2 import FEAT, G, playing
from train_snv_a4 import C, fit_phi

OUT = ROOT / "results/diag_snv_a4"
SYS = {"c2": "diag_snv_a4_c_gan", "x2": "diag_snv_a4_x_gan"}
STEPS = (0, 25, 100, 400, 1600)
ENROLL_T = ("t00", "t01", "t02")
HELD_T = ("t07", "t08", "t09")
N_E1 = 30
_CF = librosa.mel_frequencies(n_mels=34, fmin=50, fmax=8000)[1:-1]
REGIONS = {"lt500": _CF < 500, "500_1500": (_CF >= 500) & (_CF < 1500), "1500_3000": (_CF >= 1500) & (_CF < 3000), "3000_8000": _CF >= 3000}
DB = 10 / np.log(10)


def describe(o: np.ndarray, t: np.ndarray, scale: np.ndarray) -> dict:
    d = o - t
    r = {"tex": float(np.mean(np.abs(d) / scale))}
    for g, s in GROUPS.items():
        r[g] = float(np.mean(np.abs(d[s]) / scale[s]))
    m = d[GROUPS["mean"]]
    off = float(m.mean())
    r["gain_db"] = off * DB
    r["mean_shape"] = float(np.mean(np.abs(m - off) / scale[GROUPS["mean"]]))
    for k, b in REGIONS.items():
        r[f"mean_db_{k}"] = float((m[b] - off).mean() * DB)
        r[f"std_db_{k}"] = float(d[GROUPS["std"]][b].mean() * DB)
        r[f"dstd_db_{k}"] = float(d[GROUPS["dstd"]][b].mean() * DB)
    for i, f in enumerate(("flat_lo", "flat_mid", "flat_hi")):
        r[f] = float(d[GROUPS["flat"]][i])
    return r


def main() -> int:
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    voc = vocoder()

    def render(mel: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return voc(mel.to(DEV)).squeeze().cpu().numpy()

    n0r = json.loads((N0R / "items.json").read_text())
    tgts = json.loads((A1 / "convert_meta.json").read_text())["targets"]
    drows = json.loads((A1 / "diag/meta.json").read_text())
    names_tr = json.loads((ROOT / "results/diag_snv_a4_c_gan/train_log.json").read_text())["names_train"]
    man = {v["voice"]: v for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"]}
    allv = sorted(man)
    e1v = [v for v in allv if v not in names_tr][-N_E1:]
    e1t = [v for v in names_tr if v.startswith("ef")][:N_E1]
    nvman = [it for it in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"] if it["k"] == 0]

    sets: dict = {"A0F": {}, "E1V": {}, "E1T": {}}
    enroll: dict = {}
    for v in tgts:
        rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
        enroll[("A0F", v)] = (rmel, playing(FEAT / f"sparc/a1src/tref_{v}.npz", rmel.shape[1]))
    for it in n0r:
        if it["sex"] == "F":
            n = f"{it['voice']}_{it['cat']}_{it['k']}"
            sets["A0F"][("nv", n)] = {"voice": it["voice"], "cat": it["cat"], "mel": FEAT / f"mel/n0r/{n}.npy", "sp": FEAT / f"sparc/n0r/{n}.npz", "wav": FEAT / f"wav44/n0r__{n}.wav"}
    for r in drows:
        n = f"D0_{r['target']}_{r['g']}"
        sets["A0F"][("normal", n)] = {"voice": r["target"], "cat": "normal", "mel": FEAT / f"mel/a1diag/{n}.npy", "sp": FEAT / f"sparc/a1diag/{n}.npz", "wav": FEAT / f"wav44/a1diag__{n}.wav"}
    for sname, vs in (("E1V", e1v), ("E1T", e1t)):
        for v in vs:
            ms = [np.load(FEAT / f"mel/e1/{v}/{t}.npy").astype(np.float32) for t in ENROLL_T]
            ps = [playing(FEAT / f"sparc/e1/{v}/{t}.npz", m.shape[1]) for t, m in zip(ENROLL_T, ms)]
            enroll[(sname, v)] = (np.concatenate(ms, 1), np.concatenate(ps, 0))
            for t in HELD_T:
                sets[sname][("normal", f"{v}_{t}")] = {"voice": v, "cat": "normal", "mel": FEAT / f"mel/e1/{v}/{t}.npy", "sp": FEAT / f"sparc/e1/{v}/{t}.npz", "wav": ROOT / f"data/snv_e1/{v}/{t}.wav"}
        for it in nvman:
            if it["voice"] in vs:
                n = f"{it['voice']}_{it['cat']}_0"
                sets[sname][("nv", n)] = {"voice": it["voice"], "cat": it["cat"], "mel": FEAT / f"mel/e1nv/{it['voice']}/{it['cat']}_0.npy",
                                          "sp": FEAT / f"sparc/e1nv/{it['voice']}/{it['cat']}_0.npz", "wav": ROOT / f"data/snv_e1nv/{it['voice']}/{it['cat']}_0.wav"}
    print({k: len(v) for k, v in sets.items()}, "e1v", e1v[0], e1v[-1], "e1t", e1t[0], e1t[-1], flush=True)

    truth: dict = {}
    for sname, its in sets.items():
        for key, it in its.items():
            w, sr = load(it["wav"])
            t16 = to16(w, sr)
            vc = render(torch.from_numpy(np.load(it["mel"]).astype(np.float32))[None])
            truth[(sname, key)] = {"st": tex_stats(t16), "hf": hf_share(trim(w.astype(np.float64)), sr), "vcopy": tex_stats(to16(vc, 44100))}
    print("truth done", flush=True)

    rows: list = []
    phi_rms: dict = defaultdict(list)
    fit_l1: dict = defaultdict(list)
    for sysname, tag in SYS.items():
        d = ROOT / "results" / tag
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
        net = (C() if ck["mode"] == "c" else G("x")).to(DEV).eval()
        net.load_state_dict(ck["g"])
        nz = np.load(d / "norm.npz")
        mu, sd = nz["mu"], nz["sd"]
        learned = ck["codes"]["weight"].to(DEV) if ck["mode"] == "c" else None
        if learned is not None:
            phi_rms["learned_train"] = learned.pow(2).mean(1).sqrt().cpu().tolist()
        variants = []
        for sname in sets:
            if ck["mode"] == "x":
                variants.append((sname, "x2"))
            else:
                for s in (STEPS if sname == "A0F" else (100, 400)):
                    variants.append((sname, f"c2@{s}"))
                if sname == "E1T":
                    variants.append((sname, "c2@learned"))
        for sname, vname in variants:
            S = {}
            for (s_, v), (rm, rp) in enroll.items():
                if s_ != sname:
                    continue
                rmel = torch.from_numpy(rm)[None].to(DEV)
                rpt = torch.from_numpy((rp - mu) / sd)[None].to(DEV)
                mask = torch.ones(1, rm.shape[1], dtype=torch.bool, device=DEV)
                if ck["mode"] == "x":
                    with torch.no_grad():
                        S[v] = net.structure(rmel, rpt, mask)
                elif vname == "c2@learned":
                    S[v] = learned[names_tr.index(v)][None]
                else:
                    S[v] = fit_phi(net, rpt, rmel, mask, steps=int(vname.split("@")[1])) if vname != "c2@0" else torch.zeros(1, learned.shape[1], device=DEV)
                if learned is not None:
                    phi_rms[f"{sname}|{vname}"].append(float(S[v].pow(2).mean().sqrt()))
                with torch.no_grad():
                    fit_l1[f"{sname}|{vname}"].append(float((net(rpt, S[v])[..., 86:] - rmel[..., 86:]).abs().mean()))
            for key, it in sets[sname].items():
                mel_t = np.load(it["mel"]).astype(np.float32)
                p = torch.from_numpy((playing(it["sp"], mel_t.shape[1]) - mu) / sd)[None].to(DEV)
                with torch.no_grad():
                    y = net(p, S[it["voice"]])
                w = render(y)
                tr = truth[(sname, key)]
                r = {"set": sname, "sys": vname, "kind": key[0], "name": key[1], "voice": it["voice"], "cat": it["cat"],
                     "mel_l1": float(np.mean(np.abs(y[0].cpu().numpy() - mel_t))), "hf_ratio": hf_share(trim(w.astype(np.float64)), 44100) / max(tr["hf"], 1e-6)}
                r |= describe(tex_stats(to16(w, 44100)), tr["st"], scale)
                rows.append(r)
            print("done", sname, vname, flush=True)
        del net
        torch.cuda.empty_cache()
    for (sname, key), tr in truth.items():
        r = {"set": sname, "sys": "vcopy", "kind": key[0], "name": key[1], "voice": sets[sname][key]["voice"], "cat": sets[sname][key]["cat"], "mel_l1": 0.0, "hf_ratio": float("nan")}
        r |= describe(tr["vcopy"], tr["st"], scale)
        rows.append(r)

    fields = [k for k in rows[0] if k not in ("set", "sys", "kind", "name", "voice", "cat")]
    agg: dict = {}
    by = defaultdict(list)
    for r in rows:
        by[(r["set"], r["kind"], r["sys"])].append(r)
    for (s, k, v), rs in sorted(by.items()):
        agg[f"{s}|{k}|{v}"] = {"n": len(rs)} | {f: round(float(np.nanmedian([r[f] for r in rs])) if f == "hf_ratio" else float(np.mean([r[f] for r in rs])), 4) for f in fields}
    diffs: dict = {}
    for s in sets:
        for k in ("nv", "normal"):
            base = {r["name"]: r for r in by[(s, k, "x2")]}
            for v in sorted({r["sys"] for r in rows if r["set"] == s and r["sys"].startswith("c2")}):
                rs = [r for r in by[(s, k, v)] if r["name"] in base]
                grp = np.array([r["voice"] for r in rs])
                out = {}
                for f in ("tex", "mean", "mean_shape", "std", "dstd", "mod", "flat", "mel_l1"):
                    dd = np.array([r[f] - base[r["name"]][f] for r in rs])
                    out[f] = {"mean": round(float(dd.mean()), 4), "ci95_voice_cluster": cluster_ci(dd, grp, 2000)}
                diffs[f"{s}|{k}|{v}_minus_x2"] = out
    cats = defaultdict(dict)
    for s in sets:
        for v in ("x2", "c2@400", "vcopy"):
            for c in sorted({r["cat"] for r in by[(s, "nv", v)]}):
                rs = [r for r in by[(s, "nv", v)] if r["cat"] == c]
                cats[f"{s}|{v}"][c] = {f: round(float(np.mean([r[f] for r in rs])), 3) for f in ("tex", "mean_shape", "std", "dstd", "mod", "gain_db", "mean_db_3000_8000")}
    res = {"note": "事前登録の外・記録だけ(prereg の fork『id_ok + H_table_ext』の指示する崩れの型の切り分け)",
           "sets": {"E1V": [e1v[0], e1v[-1]], "E1T": [e1t[0], e1t[-1]], "enroll_E1": ENROLL_T, "held_E1": HELD_T},
           "phi_rms": {k: {"median": round(float(np.median(v)), 4), "p10": round(float(np.percentile(v, 10)), 4), "p90": round(float(np.percentile(v, 90)), 4)} for k, v in phi_rms.items()},
           "fit_enroll_l1": {k: round(float(np.mean(v)), 4) for k, v in fit_l1.items()},
           "agg": agg, "diff": diffs, "by_cat": cats}
    (OUT / "breakdown.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (OUT / "breakdown_rows.json").write_text(json.dumps(rows, ensure_ascii=False))
    for k, v in agg.items():
        print(k, {f: v[f] for f in ("n", "tex", "mean_shape", "gain_db", "std", "dstd", "mod", "flat", "mel_l1", "hf_ratio")}, flush=True)
    print(json.dumps({"phi_rms": res["phi_rms"], "fit_enroll_l1": res["fit_enroll_l1"]}, ensure_ascii=False), flush=True)
    for k, v in diffs.items():
        print(k, {f: v[f]["mean"] for f in v}, "tex ci", v["tex"]["ci95_voice_cluster"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

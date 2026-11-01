"""D225(学習なし・音声なし・results/conv_p0/d225_prereg.yaml): S 形の R̄(候補外 8 人の読みの平均)を安い形に替えたとき、製品で足す差分がどれだけ変わるか。
入力は D215 --pin と同じ(土台 = RUSTB − 較正 B・操作者の参照は _cal の書き出し)。候補外の記憶は全長で符号化してから列を選ぶ。
    uv run python d225_rbar_trunc.py --root <dev の親> --tag diag_rr18p --out ../results/conv_p0/d225.json
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
from d208_rr_natural import anchors

ROOT = Path(__file__).resolve().parent.parent
SETS = ("dev", "dev2", "dev3")
CHEAP = ("K1", "K2", "F1000", "S1000", "P8", "P4")
MAC_LIM = 10.0e6


def sl(mem, idx: torch.Tensor):
    if isinstance(mem, tuple):
        return mem[0][..., idx], mem[1][..., idx]
    return [(k[..., idx], v[..., idx]) for k, v in mem]


def cat(mems: list):
    if isinstance(mems[0], tuple):
        return torch.cat([m[0] for m in mems], -1), torch.cat([m[1] for m in mems], -1)
    return [(torch.cat([m[l][0] for m in mems], -1), torch.cat([m[l][1] for m in mems], -1)) for l in range(len(mems[0]))]


def mlen(mem) -> int:
    return int((mem if isinstance(mem, tuple) else mem[0])[0].shape[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tag", default="diag_rr18p")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_boot", type=int, default=5000)
    a = ap.parse_args()
    import conv_c0 as C0
    dev = a.device
    RT = Path(a.root)
    ck = torch.load(ROOT / "results" / a.tag / "last.pt", map_location="cpu", weights_only=False)
    net = RRm.build(ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["net"])
    nl = int(ck["cfg"].get("layers", 1))
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()

    def mem_in(p: np.ndarray, env: np.ndarray) -> tuple:
        n = min(env.shape[1], 5000)
        rc0 = (env[0, :n] - RRm.C0_SIL) / 10
        return p[:, :n].astype(np.float32), np.concatenate([env[1:25, :n] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)

    FEAT = ROOT / "data/rr_feat"
    man = {it["stem"]: it for it in json.loads((FEAT / "real_fem/manifest.json").read_text())["items"]}
    am = []
    with torch.no_grad():
        for A in anchors(8):
            ds = [torch.load(FEAT / man[s]["pt"], weights_only=False) for s in A["ref_stems"] if s in man]
            p = np.concatenate([RRm.dense(d["pk"], d["pv"]) for d in ds], 1)
            env = np.concatenate([d["env"].float().numpy() for d in ds], 1)
            rp, rv = mem_in(p, env)
            am.append(net.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev)))
    ML = [mlen(m) for m in am]
    print("anchor M", ML, flush=True)

    def ix(m: int, n: int, first: bool) -> torch.Tensor:
        return torch.arange(min(n, m), device=dev) if first else torch.from_numpy(np.unique(np.linspace(0, m - 1, min(n, m)).round().astype(np.int64))).to(dev)

    forms: dict = {"ALL": [[am[i]] for i in range(8)],
                   "K1": [[am[0]]], "K2": [[am[0]], [am[4]]], "K4a": [[am[i]] for i in (0, 2, 4, 6)], "K4b": [[am[i]] for i in (1, 3, 5, 7)],
                   "F1000": [[sl(am[i], ix(ML[i], 1000, True))] for i in range(8)], "S1000": [[sl(am[i], ix(ML[i], 1000, False))] for i in range(8)],
                   "P8": [[sl(am[i], ix(ML[i], ML[i] // 8, False)) for i in range(8)]], "P4": [[sl(am[i], ix(ML[i], ML[i] // 4, False)) for i in range(8)]]}
    reads = {f: [cat(g) if len(g) > 1 else g[0] for g in gs] for f, gs in forms.items()}
    cost_cols = {f: sum(mlen(m) for m in rs) for f, rs in reads.items()}
    ndec = {f: len(rs) for f, rs in reads.items()}

    def ld(FR: Path, corpus: str, stem: str) -> dict:
        mf = {it["stem"]: it for it in json.loads((FR / corpus / "manifest.json").read_text())["items"]}
        d = torch.load(FR / mf[stem]["pt"], weights_only=False)
        return {"p": RRm.dense(d["pk"], d["pv"]), "e3": d["e3"].float().numpy(), "c0f": d["c0f"].numpy(), "f1": d["f1"].numpy(), "env": d["env"].float().numpy(), "u": d["pk"][:, 0].long().numpy()}

    def fit(x: np.ndarray, m: int) -> np.ndarray:
        assert abs(x.shape[1] - m) <= 2, (x.shape, m)
        return np.pad(x, ((0, 0), (0, max(0, m - x.shape[1]))), mode="edge")[:, :m]

    @torch.no_grad()
    def outs(src: dict, sM: dict, tab: np.ndarray, tmem) -> dict:
        vM = src["f1"] > 0
        q = np.concatenate([src["p"], (src["e3"] - sM["e3"][:, None]) / 10, (src["c0f"] - sM["c0"])[None],
                            np.where(vM, np.log(np.maximum(src["f1"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float32)], 0).astype(np.float32)
        qt, tt = torch.from_numpy(q)[None].to(dev), torch.from_numpy(tab.astype(np.float32))[None].to(dev)

        def run(mem) -> tuple:
            e, d0, _ = net(qt, tt, mem, torch.ones(1, mlen(mem), dtype=torch.bool, device=dev))
            return e[0].cpu().numpy().astype(np.float64), d0[0].cpu().numpy().astype(np.float64)
        eA, dA = run(tmem)
        r = {"eA": eA, "dA": dA}
        for f, rs in reads.items():
            o = [run(m) for m in rs]
            r[f] = (np.mean([x[0] for x in o], 0), np.mean([x[1] for x in o], 0))
        return r

    rows = []
    mts = []
    for D in SETS:
        S = RT / D
        FR = S / "rr_real"
        cal = S / "d114/_cal_G03S10Z"
        tgt: dict = {}
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            k = f.stem[:-4]
            if ti not in tgt:
                r = ld(FR, "real_ref", f"ref_{ti:03d}")
                rp, rv = mem_in(r["p"], r["env"])
                with torch.no_grad():
                    tgt[ti] = net.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev))
            mts.append(mlen(tgt[ti]))
            b = ld(FR, "d56_src", f.stem); o = ld(FR, "d56_op", f"op_{pi:04d}")
            vo = o["f1"] > 0
            sM = {"e3": o["e3"][:, vo].mean(1), "c0": float(o["c0f"][vo].mean()), "lf": float(np.median(np.log(o["f1"][vo])))}
            C = np.fromfile(S / "d214" / f"{k}_RUSTB.f32", "<f4").reshape(-1, 31).astype(np.float64)
            bv = np.fromfile(cal / f"bias_{pi:04d}_t{ti:03d}.f32", "<f4").astype(np.float64)
            Co = np.fromfile(cal / f"op{pi:04d}_t{ti:03d}.f32", "<f4").reshape(-1, 31).astype(np.float64)
            ro = outs(o, sM, fit(Co[:, 1:25].T * 10, o["p"].shape[1]), tgt[ti])
            rs = outs(b, sM, fit((C[:, 1:25] - bv[None] * (C[:, 26:27] > 0.5)).T * 10, b["p"].shape[1]), tgt[ti])
            n = min(len(C), rs["dA"].shape[0])
            vC = C[:n, 26] > 0.5
            vo2 = o["f1"] > 0
            dS = rs["eA"] - rs["ALL"][0]
            me = (ro["eA"] - ro["ALL"][0])[:, vo2].mean(1)
            ref_e = (dS[:, :n][:, vC] - me[:, None])
            dc = rs["dA"] - rs["ALL"][1]
            ref_c = dc[:n] - float((ro["dA"] - ro["ALL"][1]).mean())
            rec = {"set": D, "key": k, "t": f"{D}_{ti}"}
            for fm in reads:
                if fm == "ALL":
                    continue
                ee = (rs[fm][0] - rs["ALL"][0])[:, :n][:, vC] - (ro[fm][0] - ro["ALL"][0])[:, vo2].mean(1)[:, None]
                ec = (rs[fm][1] - rs["ALL"][1])[:n] - float((ro[fm][1] - ro["ALL"][1]).mean())
                rec[fm] = (float(np.linalg.norm(ee) / (np.linalg.norm(ref_e) + 1e-12)), float(np.linalg.norm(ec) / (np.linalg.norm(ref_c) + 1e-12)))
            rec["K4ab"] = (float(np.linalg.norm((rs["K4a"][0] - rs["K4b"][0])[:, :n][:, vC] - (ro["K4a"][0] - ro["K4b"][0])[:, vo2].mean(1)[:, None]) / (np.linalg.norm(ref_e) + 1e-12)),
                           float(np.linalg.norm((rs["K4a"][1] - rs["K4b"][1])[:n] - float((ro["K4a"][1] - ro["K4b"][1]).mean())) / (np.linalg.norm(ref_c) + 1e-12)))
            rows.append(rec)
        print(D, len(rows), flush=True)
    Mt = float(np.median(mts))
    rd = 256 * nl
    mac = {f: 0.95e6 + (rd * Mt + 0.51e6) + rd * cost_cols[f] + ndec[f] * 0.51e6 for f in reads}
    tks = sorted({r["t"] for r in rows})
    by = {t: [i for i, r in enumerate(rows) if r["t"] == t] for t in tks}
    rng = np.random.default_rng(0)
    bidx = [np.concatenate([by[tks[j]] for j in rng.integers(0, len(tks), len(tks))]) for _ in range(a.n_boot)]
    rep: dict = {"tag": a.tag, "n_pairs": len(rows), "n_targets": len(tks), "M_target_median": Mt, "anchor_M": ML, "forms": {}}
    for fm in [f for f in reads if f != "ALL"] + ["K4ab"]:
        st = {}
        for j, nm in ((0, "env"), (1, "c0")):
            v = np.array([r[fm][j] for r in rows])
            bs = np.array([np.median(v[ii]) for ii in bidx])
            st[nm] = {"median": round(float(np.median(v)), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        if fm in reads:
            st["read_cols"] = cost_cols[fm]; st["n_decoder"] = ndec[fm]; st["MAC_per_frame"] = round(mac[fm] / 1e6, 2)
        rep["forms"][fm] = st
    rep["MAC_ALL"] = round(mac["ALL"] / 1e6, 2)
    cheap = [f for f in CHEAP if mac[f] <= MAC_LIM]
    up = {f: max(rep["forms"][f]["env"]["ci95"][1], rep["forms"][f]["c0"]["ci95"][1]) for f in cheap}
    ok = sorted([f for f in cheap if up[f] <= 0.15], key=lambda f: mac[f])
    mid = sorted([f for f in cheap if up[f] <= 0.35], key=lambda f: up[f])
    rep["cheap"] = cheap
    rep["cheap_upper"] = {f: round(v, 4) for f, v in up.items()}
    rep["verdict"] = "TRUNC_OK" if ok else "RENDER" if mid else "DISTILL"
    rep["choice"] = ok[0] if ok else mid[0] if mid else None
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

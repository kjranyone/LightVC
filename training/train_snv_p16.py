"""段 P16(paper_snv rev44 §5 M29): P8 の型(凍結した CPC の文脈 → PCA 32 × 遅れ 4 → 登録の統計で白くする → 線形 64 → 表の型の C)を、
束の一部の源を擬似の操作者の版(data/snv_p_feat_x16・snv_p16_feat.py = WORLD で高さ r 倍・声道 α 倍 → 録音の鎖 v2)に替えた自己の reamp で学ぶ。
源の演奏 = 版の白くした演奏(統計 = 同じ版の参照 = 操作者の登録)。構造 = 加工しない参照。C の高さ = 版の高さ − ln r(出力を描く高さ = 運用の「源の高さ + k」)・有声 = 版の判定。
当てる先 = 加工しない Y(有声の判定が食い違うフレームも含める = 雑音を有声と読んだフレームは周期性 0 の無声の包絡を描く)。rend の項・網・データ(E1 + E1-NV の全量)・step・学習率は P8(train_snv_p3 --play wst)と同じ。
事前登録 = results/snv_p16/prereg.yaml。保存の形は P3 / P8 と同じ(snv_p1_eval.System がそのまま読む)。
    cd training && uv run --no-sync python train_snv_p16.py --tag snv_p16_x
--vsel f(段 P70 step b・paper_snv 系 2x・診断)= 学習の声から男声(em*)を除く(検証の声はそのまま)。
--ndir <dir> --p_n p(段 P57 step b・paper_snv rev134 §5 M42): 源の側を確率 p で「前段を通った入力」(snv_p57_feat.py の特徴・k0 = Irodori の声・k1 = 擬似の操作者の版・源が版なら k1)に替える。
発話の前後の背景だけの時間も源に含め、その時間の当てる先は無音(Y = 0・有声 0)。操作者の登録の統計は、確率 --p_nref で同じ種類の足した版の参照から、残りは足さない参照から。構造(目標)は加工しない参照のまま・当てる先は加工しない Y。
--n_plain = 足した版が Irodori の声そのままの 2 版だけの置き場(data/snv_p_feat_rec2・擬似の操作者の版が無い)で使う: 源が版でないときだけ足した版に替え、k0 / k1 を半々で選ぶ(診断用)。
    cd training && uv run --no-sync python train_snv_p16.py --tag snv_p57_c --cpc results/snv_p30_dcpc/last.pt ... --ndir data/snv_p_feat_fr57 --p_n 0.5
--w_bt w(段 P78 step b・paper_snv 系 2x (xiv)): 源が擬似の操作者の版の発話で、今の回路が束の隣の声の構造へ描いた出力(勾配なし・高さは隣の声の平均へ)を源の側に置き、元の声の構造で描き直して加工しない Y に当てる項。操作者の統計は隣の声の参照。
--init <ckpt> = 線形と回路の初めの値(診断の短い学習用)。
--no_rend(段 P91)= rend の項を外す。環境変数 SNV_ENVC=1(段 P91)= 符号の入力(源・操作者の登録・目標の構造の鍵)を、固定の窓の包絡(data/snv_p_feat_st・data/snv_p_feat_st_x16 = snv_p91_stenv.py)に替える。当てる先と構造の値は今の包絡のまま。rend の項は符号の入力の型と合わないので --no_rend と組む。
--sdir <dir> --dly d(段 P102・results/snv_p102/prereg.yaml): 源が元の声の発話のとき、源と操作者の登録を --sdir の特徴(速い前段で読んだ同じ発話)から取る(当てる先・構造は T.FEAT のまま = 目標の側は元の前段)。--dly d = 源の側(包絡・高さ・有声)だけを d フレーム遅らせる(当てる先は加工しない Y のまま)。推論は遅らせない = 回路は d フレーム先の Y を描くよう学ぶ。rend の項の入力も同じだけ遅らせる。
--cn σ(段 P83・paper_snv 系 2x (xvii))= 学習のときだけ、源の白くした符号(PlayW.feat・線形の前)に、フレームと次元ごとに独立な正規の雑音(標準偏差 σ)を足す。当てる先・構造・検証は変えない(読み違いに対する回路の頑健さを学ぶ)。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

import train_snv_p1 as T
import train_snv_p3 as P3

ROOT = Path(__file__).resolve().parent.parent


def load_x(corpus: dict, xdir: Path) -> dict:
    out = {}
    for name in corpus:
        norm = [T.load_utt(xdir / "e1" / name / p.name) for p in sorted((T.FEAT / "e1" / name).glob("t*.npz"))]
        nv = [T.load_utt(xdir / "e1nv" / name / p.name) for p in sorted((T.FEAT / "e1nv" / name).glob("*.npz"))] if corpus[name]["nv"] else []
        assert len(norm) == len(corpus[name]["norm"]) and len(nv) == len(corpus[name]["nv"]), name
        for a, b in zip(norm + nv, corpus[name]["norm"] + corpus[name]["nv"]):
            assert a["y"].shape == b["y"].shape, f"{name}: 版と元のフレーム数が違う"
        out[name] = {"norm": norm, "nv": nv}
    return out


def load_n(corpus: dict, ndir: Path) -> dict:
    def part(d, sfx: str) -> dict:
        if f"f0{sfx}" not in d.files:
            return {"y": np.zeros((T.D_Y, 0), np.float32), "lf": np.zeros(0, np.float32), "v": np.zeros(0, np.float32)}
        f0, env, per = d[f"f0{sfx}"], d[f"env{sfx}"], d[f"per{sfx}"]
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
        return {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32)}

    def item(q: Path) -> list:
        out = []
        for k in (0, 1):
            d = np.load(q.with_name(f"{q.stem}_k{k}.npz"))
            out.append({"main": part(d, ""), "pre": part(d, "_pre"), "post": part(d, "_post")})
        return out
    out = {}
    for name in corpus:
        norm = [item(ndir / "e1" / name / p.name) for p in sorted((T.FEAT / "e1" / name).glob("t*.npz"))]
        nv = [item(ndir / "e1nv" / name / p.name) for p in sorted((T.FEAT / "e1nv" / name).glob("*.npz"))] if corpus[name]["nv"] else []
        assert len(norm) == len(corpus[name]["norm"]) and len(nv) == len(corpus[name]["nv"]), name
        for a, b in zip(norm + nv, corpus[name]["norm"] + corpus[name]["nv"]):
            assert all(abs(k["main"]["y"].shape[1] - b["y"].shape[1]) <= 2 for k in a), f"{name}: 足した版と元のフレーム数が違う"
        out[name] = {"norm": norm, "nv": nv}
    return out


def with_pads(tgt: dict, nk: dict) -> tuple[dict, dict]:
    n = min(tgt["y"].shape[1], nk["main"]["y"].shape[1])
    a, b = nk["pre"]["v"].shape[0], nk["post"]["v"].shape[0]
    src = {k: np.concatenate([nk["pre"][k], nk["main"][k][..., :n], nk["post"][k]], -1) for k in ("y", "lf", "v")}
    pad = lambda z: np.concatenate([np.zeros(z.shape[:-1] + (a,), z.dtype), z[..., :n], np.zeros(z.shape[:-1] + (b,), z.dtype)], -1)
    return {k: pad(tgt[k]) for k in ("y", "lf", "v")}, src


class XBatches:
    def __init__(self, corpus: dict, xcorpus: dict, dlf: dict, names: list[str], seed: int, p_x: float, p_nv: float = T.P_NV, ncorpus: dict | None = None, p_n: float = 0.0, p_nref: float = 0.5, n_plain: bool = False, scorpus: dict | None = None, dly: int = 0, rnv: float = 0.0) -> None:
        self.c, self.x, self.dlf, self.names, self.p_x, self.p_nv = corpus, xcorpus, dlf, names, p_x, p_nv
        self.s, self.dly = scorpus, dly
        self.rnv = rnv
        self.n, self.p_n, self.p_nref, self.n_plain = ncorpus, p_n, p_nref, n_plain
        self.rng = random.Random(seed)

    def _crop(self, us: tuple, L: int, dly: tuple = ()) -> tuple:
        n_all = us[0]["y"].shape[1]
        s = self.rng.randint(0, max(0, n_all - L))
        n = min(L, n_all - s)
        out = []
        for i, u in enumerate(us):
            y, lf, v = np.zeros((T.D_Y, L), np.float32), np.zeros(L, np.float32), np.zeros(L, np.float32)
            j = np.maximum(np.arange(s, s + n) - (dly[i] if dly else 0), 0)
            y[:, :n], lf[:n], v[:n] = u["y"][:, j], u["lf"][j], u["v"][j]
            out.append((y, lf, v))
        ok = np.zeros(L, np.float32)
        ok[:n] = 1.0
        return out, ok, n

    def sample(self, bs: int) -> dict:
        K = ("y", "lf", "v", "ok", "xy", "xlf", "xv", "dlf", "isx")
        B: dict = {k: [] for k in K}
        RY, RV, XRY, XRV, RYC = [], [], [], [], []
        for _ in range(bs):
            name = self.rng.choice(self.names)
            d, x = self.c[name], self.x[name]
            idx = list(range(len(d["norm"])))
            self.rng.shuffle(idx)
            use_nv = bool(d["nv"]) and self.rng.random() < self.p_nv
            isx = self.rng.random() < self.p_x
            j = self.rng.randrange(len(d["nv"])) if use_nv else idx[0]
            tgt, xt = (d["nv"][j], x["nv"][j]) if use_nv else (d["norm"][j], x["norm"][j])
            ri = idx[:T.REF_UTT] if use_nv else idx[1:1 + T.REF_UTT]
            isn = self.n is not None and self.rng.random() < self.p_n and not (self.n_plain and isx)
            nref = isn and self.rng.random() < self.p_nref
            kn = self.rng.randrange(2) if self.n_plain else int(isx)
            sd = self.s[name] if self.s is not None else d
            src = xt if isx else ((sd["nv"][j] if use_nv else sd["norm"][j]) if self.s is not None else tgt)
            if isn:
                tgt, src = with_pads(tgt, (self.n[name]["nv"] if use_nv else self.n[name]["norm"])[j][kn])
            (a, b), ok, _ = self._crop((tgt, T.code_y(src)), T.CROP, (0, self.dly))
            for k, z in zip(K, (a[0], a[1], a[2], ok, b[0], b[1], b[2], np.float32(self.dlf[name] if isx else 0.0), np.float32(isx))):
                B[k].append(z)
            ry, rv, xry, xrv, ryc = [], [], [], [], []
            for i in ri:
                rs = x["norm"][i] if isx else sd["norm"][i]
                if nref:
                    nm = self.n[name]["norm"][i][kn]["main"]
                    nn_ = min(nm["y"].shape[1], rs["y"].shape[1])
                    rc, rs = {k: d["norm"][i][k][..., :nn_] for k in ("y", "lf", "v")}, {k: nm[k][..., :nn_] for k in ("y", "lf", "v")}
                else:
                    rc = d["norm"][i]
                (a, b, c), _, n = self._crop((rc, T.code_y(rs), T.code_y(rc)), T.REF_MAX)
                ry.append(a[0][:, :n]), rv.append(a[2][:n]), xry.append(b[0][:, :n]), xrv.append(b[2][:n]), ryc.append(c[0][:, :n])
            if self.rnv > 0 and d["nv"] and self.rng.random() < self.rnv:
                ks = [q for q in range(len(d["nv"])) if not (use_nv and q == j)]
                if ks:
                    q = self.rng.choice(ks)
                    rc, rs = d["nv"][q], (x["nv"][q] if isx else sd["nv"][q])
                    (a, b, c), _, n = self._crop((rc, T.code_y(rs), T.code_y(rc)), T.REF_MAX)
                    ry.append(a[0][:, :n]), rv.append(a[2][:n]), xry.append(b[0][:, :n]), xrv.append(np.zeros(n, np.float32)), ryc.append(c[0][:, :n])
            RY.append(np.concatenate(ry, 1)), RV.append(np.concatenate(rv)), XRY.append(np.concatenate(xry, 1)), XRV.append(np.concatenate(xrv)), RYC.append(np.concatenate(ryc, 1))
        L = max(r.shape[1] for r in RY)
        pad_y = lambda zs: np.stack([np.pad(z, ((0, 0), (0, L - z.shape[1]))) for z in zs])
        pad_v = lambda zs: np.stack([np.pad(z, (0, L - len(z))) for z in zs])
        rm = np.stack([np.arange(L) < r.shape[1] for r in RY])
        t = lambda z: torch.from_numpy(np.stack(z) if isinstance(z, list) else z)
        return {k: t(v) for k, v in B.items()} | {"ry": t(pad_y(RY)), "rv": t(pad_v(RV)), "xry": t(pad_y(XRY)), "xrv": t(pad_v(XRV)), "ryc": t(pad_y(RYC)), "rmask": t(rm)}


def data_check(corpus: dict, xcorpus: dict, dlf: dict, names: list[str]) -> dict:
    rng = random.Random(0)
    df, ag, vr = [], [], []
    for name in rng.sample(names, min(200, len(names))):
        j = rng.randrange(len(corpus[name]["norm"]))
        u, x = corpus[name]["norm"][j], xcorpus[name]["norm"][j]
        both = (u["v"] > 0) & (x["v"] > 0)
        ag.append(float(np.mean(u["v"] == x["v"])))
        vr.append(float(x["v"].mean() - u["v"].mean()))
        if both.sum() > 20:
            df.append(float(np.median(np.abs(x["lf"][both] + dlf[name] - u["lf"][both]))))
    out = {"lf_abs_diff_median": round(float(np.median(df)), 4), "lf_abs_diff_p90": round(float(np.percentile(df, 90)), 4), "voicing_agree": round(float(np.mean(ag)), 4), "voiced_rate_shift": round(float(np.mean(vr)), 4)}
    print("data check(版の高さ − ln r 対 元の高さ・有声の一致)", json.dumps(out), flush=True)
    assert out["lf_abs_diff_median"] < 0.06 and out["voicing_agree"] > 0.7, "擬似の操作者の版が元と合わない"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cpc", default="results/snv_p2b_cpc/last.pt")
    ap.add_argument("--pca", default="results/snv_p8b/pca_c_P2b.npy")
    ap.add_argument("--xdir", default="data/snv_p_feat_x16")
    ap.add_argument("--lags", default="0,8,16,24")
    ap.add_argument("--circ", choices=["x", "q", "b"], default="x")
    ap.add_argument("--bneck", type=int, default=8)
    ap.add_argument("--cb", default="")
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--p_x", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ent_w", type=float, default=0.0)
    ap.add_argument("--ent_lo", type=float, default=0.01)
    ap.add_argument("--ent_hi", type=float, default=0.04)
    ap.add_argument("--marg_lo", type=float, default=0.0)
    ap.add_argument("--l1_10k", type=float, default=0.0)
    ap.add_argument("--xc0", action="store_true")
    ap.add_argument("--ndir", default="")
    ap.add_argument("--p_n", type=float, default=0.0)
    ap.add_argument("--p_nref", type=float, default=0.5)
    ap.add_argument("--n_plain", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default="results/snv_p16/prereg.yaml")
    ap.add_argument("--ship", default="results/snv_p16/ship.json")
    ap.add_argument("--vsel", default="", choices=("", "f"))
    ap.add_argument("--w_bt", type=float, default=0.0)
    ap.add_argument("--cn", type=float, default=0.0)
    ap.add_argument("--no_rend", action="store_true")
    ap.add_argument("--init", default="")
    ap.add_argument("--sdir", default="")
    ap.add_argument("--dly", type=int, default=0)
    ap.add_argument("--rnv", type=float, default=0.0)
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not ((ROOT / a.prereg).exists() and (ROOT / a.ship).exists()):
        print(f"{a.prereg} か {a.ship} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, "cpc", a.cpc, "xdir", a.xdir, "p_x", a.p_x, "ndir", a.ndir, "p_n", a.p_n, "p_nref", a.p_nref, flush=True)
    if (ROOT / a.ship).exists():
        sh = json.loads((ROOT / a.ship).read_text())
        fi = sh["future_invariance"]
        print("ship", a.ship, json.dumps({"future_invariance_ms": fi["ms"], "inconclusive": fi["inconclusive"], "unbounded": fi["unbounded"], "ledger_rows_ms": sh["ledger_rows_ms"], "ledger_pass": sh["ledger_pass"]}, ensure_ascii=False), flush=True)
        if not (sh["ledger_pass"] and fi["ms"] == 0 and not fi["inconclusive"] and not fi["unbounded"]):
            print("出荷の関門が PASS でない: 起動しない", flush=True)
            return 1
    torch.manual_seed(a.seed)
    P3.unit_tests(ROOT / a.cpc)
    lags = tuple(int(x) for x in a.lags.split(","))
    P3.playw_tests(ROOT / a.cpc, ROOT / a.pca, lags, a.shrink)
    if a.circ == "q":
        P3.circq_tests()
    if a.cb:
        P3.cmn_tests(ROOT / a.cpc, ROOT / a.pca, lags, a.shrink, ROOT / a.cb)
    if a.xc0:
        P3.xc0_tests(ROOT / a.cpc, ROOT / a.pca, lags, a.shrink, ROOT / a.cb if a.cb else None)
    t0 = time.time()
    corpus = T.load_corpus()
    xcorpus = load_x(corpus, ROOT / a.xdir)
    ncorpus = load_n(corpus, ROOT / a.ndir) if a.ndir else None
    scorpus = load_x(corpus, ROOT / a.sdir) if a.sdir else None
    print("rnv", a.rnv, "sdir", a.sdir, "dly", a.dly, flush=True)
    man = json.loads((ROOT / a.xdir / "manifest.json").read_text())["voices"]
    dlf = {v: -math.log(man[v]["r"]) for v in corpus}
    tr, va = T.split(sorted(corpus))
    if a.vsel == "f":
        tr = [v for v in tr if not v.startswith("em")]
    print(f"voices {len(corpus)} train {len(tr)} val {len(va)} load {time.time() - t0:.0f}s", flush=True)
    (out / "split.json").write_text(json.dumps({"train": tr, "val": va}))
    chk = data_check(corpus, xcorpus, dlf, tr)
    dev = "cuda"
    Pl, C = P3.PlayW(ROOT / a.cpc, ROOT / a.pca, lags=lags, shrink=a.shrink, cb=ROOT / a.cb if a.cb else None, xc0=a.xc0).to(dev), (T.CircQ() if a.circ == "q" else T.CircB(d=a.bneck) if a.circ == "b" else T.Circ()).to(dev)
    if a.init:
        ck0 = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)
        Pl.proj.load_state_dict(ck0["proj"]), C.load_state_dict(ck0["C"])
        print("init", a.init, "step", ck0.get("step"), flush=True)
    params = list(Pl.proj.parameters()) + list(C.parameters())
    cap: list = []
    if a.ent_w > 0:
        assert a.circ == "x" and 0 < a.ent_lo < a.ent_hi < 1
        C.blocks[3].register_forward_hook(lambda _m, _i, o: cap.append(o))

    def read_spread(h: torch.Tensor, s: dict, m: torch.Tensor) -> tuple:
        B, _, L = h.shape
        q = C.q(h.transpose(1, 2)).view(B, L, C.heads, C.dk).transpose(1, 2)
        logit = (q @ s["k"].transpose(-1, -2)) / math.sqrt(C.dk)
        pr = logit.masked_fill(~s["rmask"][:, None, None, :], -1e9).softmax(-1)
        r = -(pr * pr.clamp(min=1e-12).log()).sum(-1) - s["rmask"].float().sum(-1).clamp(min=1.0).log()[:, None, None]
        w = m[:, None]
        pen = ((torch.relu(r - math.log(a.ent_hi)) + torch.relu(math.log(a.ent_lo) - r)) * w).sum() / (w.sum() * C.heads).clamp(min=1.0)
        pm = (pr * w[..., None]).sum(2) / w.sum(-1, keepdim=True).clamp(min=1.0)
        rm = -(pm * pm.clamp(min=1e-12).log()).sum(-1) - s["rmask"].float().sum(-1).clamp(min=1.0).log()[:, None]
        if a.marg_lo > 0:
            pen = pen + torch.relu(math.log(a.marg_lo) - rm).mean()
        frac = float(((r * w).sum() / (w.sum() * C.heads).clamp(min=1.0)))
        return pen, frac, float(rm.mean())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = 200 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    trb = XBatches(corpus, xcorpus, dlf, tr, a.seed, a.p_x, ncorpus=ncorpus, p_n=a.p_n, p_nref=a.p_nref, n_plain=a.n_plain, scorpus=scorpus, dly=a.dly, rnv=a.rnv)
    vc, vx = XBatches(corpus, xcorpus, dlf, va, 12345, 0.0, p_nv=0.0, scorpus=scorpus, dly=a.dly), XBatches(corpus, xcorpus, dlf, va, 12345, 1.0, p_nv=0.0, scorpus=scorpus, dly=a.dly)
    vbs = {"c": [vc.sample(a.bs) for _ in range(8)], "x": [vx.sample(a.bs) for _ in range(8)]}
    if ncorpus is not None:
        vn, vxn = XBatches(corpus, xcorpus, dlf, va, 12345, 0.0, p_nv=0.0, ncorpus=ncorpus, p_n=1.0, p_nref=a.p_nref, n_plain=a.n_plain), XBatches(corpus, xcorpus, dlf, va, 12345, 1.0, p_nv=0.0, ncorpus=ncorpus, p_n=1.0, p_nref=a.p_nref)
        vbs |= {"n": [vn.sample(a.bs) for _ in range(8)]} | ({} if a.n_plain else {"xn": [vxn.sample(a.bs) for _ in range(8)]})
    to_dev = lambda b: {k: x.to(dev, non_blocking=True) for k, x in b.items()}

    def forward(b: dict, noisy: bool = False) -> tuple:
        st_ref = Pl.mean_of(b["ryc"], b["rv"], b["rmask"])
        st_src = Pl.mean_of(b["xry"], b["xrv"], b["rmask"])
        s = C.structure(b["ry"], Pl(b["ryc"], b["rv"], st_ref), b["rmask"])
        lf = (b["xlf"] + b["dlf"][:, None]) * b["xv"]
        if noisy and a.cn > 0:
            f = Pl.feat(b["xy"], b["xv"], st_src)
            a_src = Pl.proj((f + a.cn * torch.randn_like(f)).transpose(1, 2)).transpose(1, 2)
        else:
            a_src = Pl(b["xy"], b["xv"], st_src)
        return st_ref, s, lf, a_src, b["ok"].clone()

    def back(b: dict, st_ref: tuple, s: dict, lf: torch.Tensor, a_src: torch.Tensor) -> torch.Tensor:
        roll = lambda z: z.roll(1, 0)
        vm = lambda l, v: (l * v).sum(-1) / v.sum(-1).clamp(min=1.0)
        vb = roll(b["v"])
        d = torch.where(vb.sum(-1) > 10, vm(roll(b["lf"]), vb) - vm(lf, b["xv"]), torch.zeros_like(vb.sum(-1)))[:, None]
        with torch.no_grad():
            yB = C(a_src, (lf + d) * b["xv"], b["xv"], {k: roll(z) for k, z in s.items()})
        return C(Pl(yB, b["xv"], tuple(roll(z) for z in st_ref)), lf, b["xv"], s)

    @torch.no_grad()
    def evaluate() -> dict:
        Pl.eval(), C.eval()
        rec: dict = {}
        for name, bs_ in vbs.items():
            ls, lt, en, mx, ef, lb = [], [], [], [], [], []
            for vb in bs_:
                b = to_dev(vb)
                st_ref, s, lf, a_src, m = forward(b)
                m[:, :T.WARM_SELF] = 0
                ls.append(float(T.masked_l1(C(a_src, lf, b["xv"], s), b["y"], m)))
                mb = m.clone()
                mb[:, :T.WARM_CYC] = 0
                lb.append(float(T.masked_l1(back(b, st_ref, s, lf, a_src), b["y"], mb)))
                rmean = (b["ry"] * b["rmask"][:, None]).sum(-1, keepdim=True) / b["rmask"].sum(-1)[:, None, None].clamp(min=1)
                lt.append(float(T.masked_l1(rmean.expand_as(b["y"]), b["y"], m)))
                e_, x_, f_ = P3.attn_stats(C, a_src, lf, b["xv"], s)
                en.append(e_), mx.append(x_), ef.append(f_)
            rec |= {f"val_{name}_l1": float(np.mean(ls)), f"trivial_{name}": float(np.mean(lt)), f"attn_max_{name}": float(np.mean(mx)), f"attn_eff_{name}": float(np.mean(ef)), f"val_{name}_bt": float(np.mean(lb))}
        Pl.train(), C.train()
        return rec

    r0 = evaluate()
    print("floors", json.dumps(r0), flush=True)
    log = {"floors": r0, "data_check": chk, "evals": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        b = to_dev(trb.sample(a.bs))
        st_ref, s, lf, a_src, m = forward(b, noisy=True)
        ms, mc = m.clone(), m.clone()
        ms[:, :T.WARM_SELF] = 0
        mc[:, :T.WARM_CYC] = 0
        cap.clear()
        yA = C(a_src, lf, b["xv"], s)
        l_self = T.masked_l1(yA, b["y"], ms)
        yR = yA.detach() if a.dly == 0 else torch.cat([yA.detach()[..., :1].expand(-1, -1, a.dly), yA.detach()[..., :-a.dly]], -1)
        l_rend = torch.zeros((), device=dev) if a.no_rend else T.masked_l1(C(Pl(yR, b["xv"], st_ref), lf, b["xv"], s), b["y"], mc)
        loss = l_self + l_rend
        l_bt = T.masked_l1(back(b, st_ref, s, lf, a_src), b["y"], mc * b["isx"][:, None]) if a.w_bt > 0 else torch.zeros((), device=dev)
        loss = loss + a.w_bt * l_bt
        l_ent, frac, marg = (read_spread(cap[0], s, ms) if a.ent_w > 0 else (torch.zeros((), device=dev), 0.0, 0.0))
        loss = loss + a.ent_w * l_ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} self {float(l_self):.4f} rend {float(l_rend):.4f} gn {float(gn):.2f} {(time.time() - t0) / step:.3f}s/step" + (f" bt {float(l_bt):.4f}" if a.w_bt > 0 else "") + (f" ent {float(l_ent):.4f} frac {math.exp(frac):.4f} marg {math.exp(marg):.3f}" if a.ent_w > 0 else ""), flush=True)
        if step % 2000 == 0 or step == steps:
            cap.clear()
            rec = {"step": step, "train_self": float(l_self), "train_rend": float(l_rend), "sec": time.time() - t0} | ({"train_ent": float(l_ent), "read_frac": math.exp(frac), "read_marg": math.exp(marg)} if a.ent_w > 0 else {}) | evaluate()
            cap.clear()
            log["evals"].append(rec)
            print("eval", json.dumps(rec), flush=True)
            ck = {"proj": Pl.proj.state_dict(), "C": C.state_dict(), "cpc": a.cpc, "step": step, "circ": a.circ, "play": "wst", "pca": a.pca, "xdir": a.xdir, "p_x": a.p_x, "ndir": a.ndir, "p_n": a.p_n, "p_nref": a.p_nref, "n_plain": a.n_plain, "lags": list(lags), "shrink": a.shrink, "cb": a.cb, "xc0": a.xc0, "ent": [a.ent_lo, a.ent_hi, a.ent_w, a.marg_lo], "w_bt": a.w_bt, "init": a.init, "cn": a.cn, "no_rend": a.no_rend, "envc": T.ENVC, "sdir": a.sdir, "dly": a.dly, "rnv": a.rnv, "bneck": a.bneck}
            torch.save(ck, out / "last.pt")
            if step == steps // 2:
                torch.save(ck, out / f"step{step // 1000}k.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
            if step == 10000 and rec["val_c_l1"] > 0.8 * rec["trivial_c"]:
                print(f"TRIPWIRE: val L1 {rec['val_c_l1']:.4f} > 0.8 × trivial {rec['trivial_c']:.4f} at 10k → stop", flush=True)
                return 2
            if step == 10000 and a.ent_w > 0 and rec["read_frac"] > 2 * a.ent_hi:
                print(f"TRIPWIRE: 読みの広がり {rec['read_frac']:.4f} > 2 × {a.ent_hi} at 10k(上限が効いていない)→ stop", flush=True)
                return 2
            if step == 10000 and a.l1_10k > 0 and rec["val_c_l1"] > a.l1_10k:
                print(f"TRIPWIRE: val L1 {rec['val_c_l1']:.4f} > {a.l1_10k} at 10k → stop", flush=True)
                return 2
            if step == 2000 and a.marg_lo > 0 and rec["read_marg"] < 0.5 * a.marg_lo:
                print(f"TRIPWIRE: 問いで平均した広がり {rec['read_marg']:.4f} < 0.5 × {a.marg_lo} at 2k(下限が効いていない)→ stop", flush=True)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

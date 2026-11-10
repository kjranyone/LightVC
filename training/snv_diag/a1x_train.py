"""実データの A1 の測定 diag_snv_a1x(results/diag_snv_a1x/prereg.yaml・paper_snv rev9): 段 I と同じ描画 C(train_snv_p1.Circ)・尤度(ラプラス・尺度を学ぶ)・参照(3 本 × 最大 10 s)・切り出し 4 s で、
演奏だけを SPARC の EMA(物差しとしてだけ・12 次元・50 Hz → 200 fps の線形補間・全体で標準化・64 次元へ 0 で埋める)にする。C の鍵 = 参照の EMA。学習 = E1 + E1-NV(段 D は SPARC の特徴が無い)。
ema_of(rel) = data/snv_e1_feat/sparc/<組>/... を前段のフレームへ揃えた [64, n](評価 a1x_eval.py と共用)。
    cd training && uv run --no-sync python snv_diag/a1x_train.py --tag diag_snv_a1x [--steps 20000] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_snv_i as I
import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = I.ROOT
SP = ROOT / "data/snv_e1_feat/sparc"
NORM = ROOT / "results/diag_snv_a1x/ema_norm.npz"


def sparc_path(rel: str) -> Path:
    head, _, rest = rel.partition("/")
    if head == "real":
        return SP / f"a1src/{rest}.npz"
    return SP / f"{head}/{rest}.npz"


def ema_raw(rel: str, n: int) -> np.ndarray:
    e = np.load(sparc_path(rel))["ema"].astype(np.float64)
    t = np.arange(n) * 0.005
    k = np.arange(len(e)) * 0.02
    return np.stack([np.interp(t, k, e[:, d]) for d in range(e.shape[1])], 0)


def ema_of(rel: str, n: int, norm: tuple) -> np.ndarray:
    x = (ema_raw(rel, n) - norm[0][:, None]) / norm[1][:, None]
    return np.concatenate([x, np.zeros((T.D_A - x.shape[0], n))], 0).astype(np.float32)


class Data:
    def __init__(self, names: list[str], norm: tuple | None) -> None:
        e1 = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
        nv = json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())
        self.names = names
        self.u: dict = {n: {"norm": [], "nv": []} for n in names}
        for v in e1["voices"]:
            if v["voice"] in self.u:
                for f in v["files"]:
                    self.u[v["voice"]]["norm"].append(f"e1/{v['voice']}/{f['t']}")
        for it in nv["items"]:
            rel = f"e1nv/{it['voice']}/{it['cat']}_{it['k']}"
            if it["voice"] in self.u and (T.FEAT / f"{rel}.npz").exists() and sparc_path(rel).exists():
                self.u[it["voice"]]["nv"].append(rel)
        if norm is None:
            xs = [ema_raw(r, T.load_utt(T.FEAT / f"{r}.npz")["y"].shape[1]) for n in names[:60] for r in self.u[n]["norm"][:3]]
            z = np.concatenate(xs, 1)
            norm = (z.mean(1), z.std(1) + 1e-6)
        self.norm = norm
        self.cache: dict = {}

    def get(self, rel: str) -> dict:
        if rel not in self.cache:
            u = T.load_utt(T.FEAT / f"{rel}.npz")
            u["a"] = ema_of(rel, u["y"].shape[1], self.norm)
            self.cache[rel] = u
        return self.cache[rel]


class Batches:
    def __init__(self, d: Data, seed: int, p_nv: float = T.P_NV) -> None:
        self.d, self.rng, self.p_nv = d, random.Random(seed), p_nv

    def _crop(self, u: dict, L: int) -> tuple:
        n_all = u["y"].shape[1]
        s = self.rng.randint(0, max(0, n_all - L))
        n = min(L, n_all - s)
        out = [np.zeros((T.D_Y, L), np.float32), np.zeros(L, np.float32), np.zeros(L, np.float32), np.zeros(L, np.float32), np.zeros((T.D_A, L), np.float32)]
        out[0][:, :n], out[1][:n], out[2][:n], out[3][:n], out[4][:, :n] = u["y"][:, s:s + n], u["lf"][s:s + n], u["v"][s:s + n], 1.0, u["a"][:, s:s + n]
        return out

    def sample(self, bs: int) -> dict:
        B: dict = {k: [] for k in ("y", "lf", "v", "ok", "a", "ry", "rv", "ra")}
        for _ in range(bs):
            v = self.d.u[self.rng.choice(self.d.names)]
            idx = list(range(len(v["norm"])))
            self.rng.shuffle(idx)
            use_nv = bool(v["nv"]) and self.rng.random() < self.p_nv
            tgt = self.d.get(self.rng.choice(v["nv"]) if use_nv else v["norm"][idx[0]])
            for k, x in zip(("y", "lf", "v", "ok", "a"), self._crop(tgt, T.CROP)):
                B[k].append(x)
            refs = [v["norm"][i] for i in (idx[:I.REF_N] if use_nv else idx[1:1 + I.REF_N])]
            ry, rv, ra = [], [], []
            for r in refs:
                yy, _, vv, okk, aa = self._crop(self.d.get(r), I.REF_LEN)
                n = int(okk.sum())
                ry.append(yy[:, :n]), rv.append(vv[:n]), ra.append(aa[:, :n])
            B["ry"].append(np.concatenate(ry, 1)), B["rv"].append(np.concatenate(rv)), B["ra"].append(np.concatenate(ra, 1))
        Lr = max(r.shape[1] for r in B["ry"])
        ry, rv, ra, rm = np.zeros((bs, T.D_Y, Lr), np.float32), np.zeros((bs, Lr), np.float32), np.zeros((bs, T.D_A, Lr), np.float32), np.zeros((bs, Lr), bool)
        for i in range(bs):
            n = B["ry"][i].shape[1]
            ry[i, :, :n], rv[i, :n], ra[i, :, :n], rm[i, :n] = B["ry"][i], B["rv"][i], B["ra"][i], True
        t = torch.from_numpy
        return {"y": t(np.stack(B["y"])), "lf": t(np.stack(B["lf"])), "v": t(np.stack(B["v"])), "ok": t(np.stack(B["ok"])), "a": t(np.stack(B["a"])), "ry": t(ry), "rv": t(rv), "ra": t(ra), "rmask": t(rm)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="diag_snv_a1x")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print("prereg", ROOT / "results/diag_snv_a1x/prereg.yaml", "paper current/paper_snv.md", flush=True)
    torch.manual_seed(0)
    names = sorted(v["voice"] for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"])
    tr, va = T.split(names)
    if a.smoke:
        tr = tr[:40]
    dtr = Data(tr, None)
    if not a.smoke:
        NORM.parent.mkdir(parents=True, exist_ok=True)
        np.savez(NORM, mean=dtr.norm[0], std=dtr.norm[1])
    dva = Data(va, dtr.norm)
    (out / "split.json").write_text(json.dumps({"train": tr, "val": va}))
    dev = "cuda"
    C = T.Circ().to(dev)
    logb = nn.Parameter(torch.full((T.D_Y,), math.log(P11.B_INIT), device=dev))
    params = list(C.parameters()) + [logb]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = 200 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    trb, vab = Batches(dtr, 0), Batches(dva, 12345, p_nv=0.0)
    vb = [vab.sample(a.bs) for _ in range(8)]
    to_dev = lambda b: {k: x.to(dev, non_blocking=True) for k, x in b.items()}

    @torch.no_grad()
    def evaluate() -> dict:
        C.eval()
        r: dict = {"val_l1": [], "fwd": []}
        for b in map(to_dev, vb):
            m = b["ok"].clone()
            m[:, :T.WARM_SELF] = 0
            S = C.structure(b["ry"], b["ra"], b["rmask"])
            r["val_l1"].append(float(T.masked_l1(C(b["a"], b["lf"], b["v"], S), b["y"], m)))
            perm = torch.roll(torch.arange(b["y"].shape[0], device=dev), 1)
            r["fwd"].append(float(T.masked_l1(C(b["a"], b["lf"], b["v"], {k: x[perm] for k, x in S.items()}), b["y"], m)))
        C.train()
        return {k: float(np.mean(x)) for k, x in r.items()}

    log = {"evals": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        b = to_dev(trb.sample(a.bs))
        m = b["ok"].clone()
        m[:, :T.WARM_SELF] = 0
        S = C.structure(b["ry"], b["ra"], b["rmask"])
        yh = C(b["a"], b["lf"], b["v"], S)
        nll = (P11.nll_of(yh, b["y"], logb) * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True)
        nll.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} nll {float(nll):.2f} l1 {float(T.masked_l1(yh, b['y'], m)):.4f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 2000 == 0 or step == steps:
            ev = evaluate()
            rec = {"step": step, **ev, "train_nll": float(nll), "sec": time.time() - t0}
            log["evals"].append(rec)
            print("eval", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            torch.save({"C": C.state_dict(), "logb": logb.detach().cpu(), "step": step, "play": "ema", "ema_norm": [dtr.norm[0], dtr.norm[1]]}, out / "last.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

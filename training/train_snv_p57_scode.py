"""段 P57 step b(paper_snv rev136 §5 M42 (a′)): 足した入力から、きれいな入力の符号を当てる符号器(読み手の入口を足した入力の族で学ぶ)。
先生 = 凍結した研究の符号(results/snv_p30_dcpc・CPC の文脈 c)を、加工しない入力(Irodori の声 or その擬似の操作者の版・足さない)に掛けた出力。
生徒 = 同じ網(先生の重みから始める)。入力 = 同じ発話の足した版(--ndir = snv_p57_feat.py の特徴・k0 = Irodori の声・k1 = 擬似の操作者の版・前後の背景だけの時間を含む)か、足さない版(恒等を保つ)。
損失 = 有効なフレームの ‖(c_生徒(入力) − c_先生(きれいな入力)) / σ‖² の平均(σ = 先生の c の次元ごとの標準偏差・定数)。前後の背景だけの時間の当てる先 = 無音の入力に対する先生の c。
符号の座標は先生と同じ = PCA・符号帳・回路 C(snv_p30_c)をそのまま使える。当てる先は加工しない Irodori の音から(入力側だけの加工)・本物の音声は使わない・外の模型を損失に置かない(先生は自前の符号)。
網の形は先生と同じ = 推論の経路・演算量・遅延は変わらない(出荷の関門の台帳 results/snv_p30/ship_c/ship.json)。
    cd training && uv run --no-sync python train_snv_p57_scode.py --tag snv_p57_sv --ndir data/snv_p_feat_v357 --prereg results/snv_p57/prereg.yaml
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
import train_snv_p2_cpc as CP

ROOT = Path(__file__).resolve().parent.parent
XFEAT = ROOT / "data/snv_p_feat_x16"
CROP = 512
SKIP = 16
KINDS = (("c0", 0.15), ("c1", 0.15), ("n0", 0.35), ("n1", 0.35))


def arr(f0: np.ndarray, env: np.ndarray, per: np.ndarray) -> np.ndarray:
    return np.concatenate([T.to_y(env, per, f0), (f0 > 0).astype(np.float32)[None]], 0)


def load_sets(names: list[str], ndir: Path) -> dict:
    out: dict = {"c0": [], "c1": [], "n0": [], "n1": []}
    for v in names:
        for p in sorted((T.FEAT / "e1" / v).glob("t*.npz")) + sorted((T.FEAT / "e1nv" / v).glob("*.npz")):
            kind = p.parent.parent.name
            for k, base in ((0, T.FEAT), (1, XFEAT)):
                d = np.load(base / kind / v / p.name)
                out[f"c{k}"].append(arr(d["f0"], d["env"], d["per"]))
                n = np.load(ndir / kind / v / f"{p.stem}_k{k}.npz")
                out[f"n{k}"].append(tuple(arr(n[f"f0{s}"], n[f"env{s}"], n[f"per{s}"]) for s in ("_pre", "", "_post")))
    return out


def pair(S: dict, j: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    ref = S[f"c{kind[1]}"][j]
    if kind[0] == "c":
        return ref, ref
    pre, main, post = S[kind][j]
    n = min(ref.shape[1], main.shape[1])
    z = lambda m: np.zeros((ref.shape[0], m), np.float32)
    return np.concatenate([pre, main[:, :n], post], 1), np.concatenate([z(pre.shape[1]), ref[:, :n], z(post.shape[1])], 1)


def batch(S: dict, bs: int, rng: random.Random, kind: str | None = None) -> tuple:
    X, R, M = np.zeros((bs, T.D_Y + 1, CROP), np.float32), np.zeros((bs, T.D_Y + 1, CROP), np.float32), np.zeros((bs, CROP), np.float32)
    for i in range(bs):
        kd = kind or rng.choices([k for k, _ in KINDS], [p for _, p in KINDS])[0]
        x, r = pair(S, rng.randrange(len(S["c0"])), kd)
        s = rng.randint(0, max(0, x.shape[1] - CROP))
        n = min(CROP, x.shape[1] - s)
        X[i, :, :n], R[i, :, :n], M[i, SKIP:n] = x[:, s:s + n], r[:, s:s + n], 1.0
    return torch.from_numpy(X), torch.from_numpy(R), torch.from_numpy(M)


def dist(cs: torch.Tensor, ct: torch.Tensor, m: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    d = ((cs - ct) / sd[None, :, None]).pow(2).mean(1)
    return (d * m).sum() / m.sum().clamp(min=1.0)


def unit_tests(teacher: Path) -> None:
    st = torch.load(teacher, map_location="cpu", weights_only=False)
    a, b = CP.CPC().eval(), CP.CPC().eval()
    a.load_state_dict(st["cpc"]), b.load_state_dict(st["cpc"])
    sig = torch.from_numpy(np.asarray(st["sig"], np.float32))
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, T.D_Y + 1, 300, generator=g) * 0.3
    with torch.no_grad():
        ca, cb = a.encode(CP.x_torch(x, sig))[1], b.encode(CP.x_torch(x, sig))[1]
        x2 = x.clone()
        x2[..., 200:] = torch.randn(2, T.D_Y + 1, 100, generator=g)
        c2 = b.encode(CP.x_torch(x2, sig))[1]
    assert float(dist(ca, cb, torch.ones(2, 300), torch.ones(ca.shape[1]))) == 0.0, "始めの生徒が先生と違う"
    assert torch.allclose(cb[..., :200], c2[..., :200], atol=1e-5) and not torch.allclose(cb[..., 200:], c2[..., 200:]), "符号が因果でない"
    S = {"c0": [np.ones((30, 40), np.float32)], "c1": [np.ones((30, 40), np.float32) * 2], "n0": [(np.zeros((30, 5), np.float32) + 7, np.ones((30, 41), np.float32) * 3, np.zeros((30, 0), np.float32))], "n1": [(np.zeros((30, 0), np.float32), np.ones((30, 40), np.float32), np.zeros((30, 2), np.float32))]}
    x_, r_ = pair(S, 0, "n0")
    assert x_.shape == r_.shape == (30, 45) and float(r_[:, :5].sum()) == 0.0 and float(r_[:, 5:].min()) == 1.0 and float(x_[:, :5].min()) == 7.0, "前の背景だけの時間の当てる先が無音でない"
    x_, r_ = pair(S, 0, "n1")
    assert x_.shape == (30, 42) and float(r_[:, :40].min()) == 2.0 and float(r_[:, 40:].sum()) == 0.0, "版の当てる先がきれいな版でない"
    print("unit tests ok(始めの生徒 = 先生・因果性・前後の時間の当てる先は無音・版の当てる先はきれいな版)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--teacher", default="results/snv_p30_dcpc/last.pt")
    ap.add_argument("--ndir", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default="results/snv_p57/prereg.yaml")
    ap.add_argument("--ship", default="results/snv_p30/ship_c/ship.json")
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not ((ROOT / a.prereg).exists() and (ROOT / a.ship).exists()):
        print(f"{a.prereg} か {a.ship} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, "teacher", a.teacher, "ndir", a.ndir, flush=True)
    sh = json.loads((ROOT / a.ship).read_text())
    fi = sh["future_invariance"]
    print("ship", a.ship, json.dumps({"future_invariance_ms": fi["ms"], "inconclusive": fi["inconclusive"], "unbounded": fi["unbounded"], "ledger_rows_ms": sh["ledger_rows_ms"], "ledger_pass": sh["ledger_pass"]}, ensure_ascii=False), flush=True)
    if not (sh["ledger_pass"] and fi["ms"] == 0 and not fi["inconclusive"] and not fi["unbounded"]):
        print("出荷の関門が PASS でない: 起動しない", flush=True)
        return 1
    torch.manual_seed(a.seed)
    unit_tests(ROOT / a.teacher)
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    t0 = time.time()
    tr = load_sets(split["train"][:40] if a.smoke else split["train"], ROOT / a.ndir)
    va = load_sets(split["val"][:8] if a.smoke else split["val"], ROOT / a.ndir)
    print(f"utts train {len(tr['c0'])} val {len(va['c0'])} load {time.time() - t0:.0f}s", flush=True)
    dev = "cuda"
    st = torch.load(ROOT / a.teacher, map_location="cpu", weights_only=False)
    sig = np.asarray(st["sig"], np.float32)
    sig_t = torch.from_numpy(sig).to(dev)
    Tm, Sm = CP.CPC().to(dev).eval().requires_grad_(False), CP.CPC().to(dev)
    Tm.load_state_dict(st["cpc"]), Sm.load_state_dict(st["cpc"])
    rng = random.Random(a.seed)
    with torch.no_grad():
        cs = [Tm.encode(CP.x_torch(batch(tr, a.bs, rng, "c0")[0].to(dev), sig_t))[1] for _ in range(8)]
        sd = torch.cat(cs, 0).transpose(0, 1).reshape(cs[0].shape[1], -1).std(1).clamp(min=1e-4)
    vr = random.Random(12345)
    vbs = {k: [batch(va, a.bs, vr, k) for _ in range(6)] for k, _ in KINDS}
    params = [p for n, p in Sm.named_parameters() if not n.startswith("W.")]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = 200 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    @torch.no_grad()
    def evaluate() -> dict:
        Sm.eval()
        rec = {}
        for k, bs_ in vbs.items():
            ds, dt = [], []
            for X, R, M in bs_:
                X, R, M = X.to(dev), R.to(dev), M.to(dev)
                ct = Tm.encode(CP.x_torch(R, sig_t))[1]
                ds.append(float(dist(Sm.encode(CP.x_torch(X, sig_t))[1], ct, M, sd)))
                dt.append(float(dist(Tm.encode(CP.x_torch(X, sig_t))[1], ct, M, sd)))
            rec[f"val_{k}"] = round(float(np.mean(ds)), 4)
            rec[f"teacher_{k}"] = round(float(np.mean(dt)), 4)
        Sm.train()
        return rec

    r0 = evaluate()
    print("floors", json.dumps(r0), flush=True)
    log = {"floors": r0, "evals": []}
    t0 = time.time()
    Sm.train()
    for step in range(1, steps + 1):
        X, R, M = (q.to(dev) for q in batch(tr, a.bs, rng))
        with torch.no_grad():
            ct = Tm.encode(CP.x_torch(R, sig_t))[1]
        loss = dist(Sm.encode(CP.x_torch(X, sig_t))[1], ct, M, sd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or a.smoke and step % 50 == 0:
            print(f"step {step} loss {float(loss):.4f} gn {float(gn):.2f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 2000 == 0 or step == steps:
            rec = {"step": step, "train": float(loss), "sec": time.time() - t0} | evaluate()
            log["evals"].append(rec)
            print("eval", json.dumps(rec), flush=True)
            torch.save({"cpc": Sm.state_dict(), "sig": sig, "step": step, "teacher": a.teacher, "ndir": a.ndir}, out / "last.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

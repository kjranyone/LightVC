"""A5(synth_nam_vc §5・diag): 学習でも推論と同じ得方で回路の値 Φ を作る連続の回路。A4 の c と同じ G(因果の膨張畳み込み 12 層・層ごとの FiLM)で、
声ごとの自由な値を持たず、Φ = 当てはめ(同じ声の普通の発話の参照だけから・学んだ初期値 Φ0 から層ごとに正規化した勾配を K 歩・歩幅は固定の減衰列)。
外の損失(同じ声の別の普通の発話 1 本 + 音素の無い発声 1 本の L1)は当てはめた Φ の上で G へ流し、Φ0 へは素通しで流す(一階・K 歩の中は逆伝播しない)。推論の登録も同じ K 歩(fit)。
二階(歩幅も学ぶ)の走行は 2,500〜5,000 step で発散した(results/diag_snv_a5_s1/train_so_diverged.log)ので一階に替えた。
段 1 = L1。段 2(--gan)= 段 1 から A4 と同じ分布の損失(PatchGAN 2 段の LSGAN + 特徴の一致・重み 1 / 2 / 45)。
事前登録 = results/diag_snv_a5/prereg.yaml。演奏と描画の代役が非因果なので diag(製品の判断に使わない)。
    cd training && uv run --no-sync python train_snv_a5.py --tag diag_snv_a5_s1
    cd training && uv run --no-sync python train_snv_a5.py --tag diag_snv_a5_s2 --init diag_snv_a5_s1 --gan --steps 20000
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import random
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_snv_a2 import CROP, N_MEL, P_DIM, REF_UTT, ROOT, VAL_VOICES, WARM, l1
from train_snv_a4 import CD, CH, N_LAYER, PHI, C, MultiD, load_corpus, norm_stats

K_FIT = 8
A0 = 0.08
A_DECAY = 0.75
EPS = 1e-14
REF_CROP = 450
VPB = 8


class Fit(nn.Module):
    def __init__(self, k: int = K_FIT) -> None:
        super().__init__()
        self.phi0 = nn.Parameter(torch.zeros(PHI))
        self.register_buffer("a", torch.tensor([A0 * A_DECAY ** i for i in range(k)]))

    def forward(self, net: C, rp: torch.Tensor, rmel: torch.Tensor, rmask: torch.Tensor) -> torch.Tensor:
        w = rmask[:, None].float().clone()
        w[..., :WARM] = 0.0
        den = w.sum((1, 2)).clamp(min=1.0) * N_MEL
        base = self.phi0.detach().expand(rp.shape[0], PHI)
        phi = base
        with torch.enable_grad():
            for a in self.a.tolist():
                phi = phi.detach().requires_grad_(True)
                per = ((net(rp, phi) - rmel).abs() * w).sum((1, 2)) / den
                g, = torch.autograd.grad(per.sum(), phi)
                g = g.view(-1, N_LAYER, 2 * CH)
                phi = phi - a * (g / (g.pow(2).mean(-1, keepdim=True) + EPS).sqrt()).view(-1, PHI)
        return self.phi0 + (phi.detach() - base)


class VBatches:
    def __init__(self, voices: dict, names: list[str], mu: np.ndarray, sd: np.ndarray, seed: int) -> None:
        self.v, self.names, self.mu, self.sd = voices, names, mu, sd
        self.rng = random.Random(seed)

    def target(self, mel: np.ndarray, p: np.ndarray) -> tuple:
        T = mel.shape[1]
        s = self.rng.randint(0, max(0, T - CROP))
        m = np.zeros((N_MEL, CROP), np.float32)
        q = np.zeros((CROP, P_DIM), np.float32)
        k = np.zeros(CROP, np.float32)
        n = min(CROP, T - s)
        m[:, :n] = mel[:, s:s + n]
        q[:n] = (p[s:s + n] - self.mu) / self.sd
        k[WARM:n] = 1.0
        return m, q, k

    def sample(self, nv: int = VPB) -> dict:
        tm, tp, mask, isnv, rm, rp = [], [], [], [], [], []
        for vi in self.rng.sample(range(len(self.names)), nv):
            d = self.v[self.names[vi]]
            order = list(range(len(d["n"])))
            self.rng.shuffle(order)
            for mel, p in (d["n"][order[0]], d["v"][self.rng.randrange(len(d["v"]))]):
                m, q, k = self.target(mel, p)
                tm.append(m), tp.append(q), mask.append(k)
            isnv += [False, True]
            ms, ps = [], []
            for i in order[1:1 + REF_UTT]:
                mel, p = d["n"][i]
                s = self.rng.randint(0, max(0, mel.shape[1] - REF_CROP))
                ms.append(mel[:, s:s + REF_CROP]), ps.append(p[s:s + REF_CROP])
            rm.append(np.concatenate(ms, 1))
            rp.append((np.concatenate(ps, 0) - self.mu) / self.sd)
        L = max(r.shape[1] for r in rm)
        RM = np.zeros((nv, N_MEL, L), np.float32)
        RP = np.zeros((nv, L, P_DIM), np.float32)
        RK = np.zeros((nv, L), bool)
        for i, (a, b) in enumerate(zip(rm, rp)):
            RM[i, :, :a.shape[1]], RP[i, :a.shape[1]], RK[i, :a.shape[1]] = a, b, True
        return {"mel": torch.from_numpy(np.stack(tm)), "p": torch.from_numpy(np.stack(tp)), "mask": torch.from_numpy(np.stack(mask)),
                "nv": torch.tensor(isnv), "rmel": torch.from_numpy(RM), "rp": torch.from_numpy(RP), "rmask": torch.from_numpy(RK)}


def prefetch(b: VBatches, n: int = 8) -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=n)

    def run() -> None:
        while True:
            q.put({k: v.pin_memory() for k, v in b.sample().items()})

    threading.Thread(target=run, daemon=True).start()
    return q


def unit_tests() -> None:
    torch.manual_seed(0)
    net = C().double()
    fit = Fit().double()
    p = torch.randn(2, 200, P_DIM, dtype=torch.float64)
    phi = torch.randn(2, PHI, dtype=torch.float64) * 0.1
    y1 = net(p, phi)
    p2 = p.clone()
    p2[:, 120:] = torch.randn(2, 80, P_DIM, dtype=torch.float64)
    assert torch.allclose(y1[..., :120], net(p2, phi)[..., :120], atol=1e-9) and not torch.allclose(y1[..., 120:], net(p2, phi)[..., 120:]), "C が因果でない"
    with torch.no_grad():
        fit.phi0.normal_(0, 0.1)
    rp = torch.randn(2, 300, P_DIM, dtype=torch.float64)
    rmel = torch.randn(2, N_MEL, 300, dtype=torch.float64)
    rk = torch.ones(2, 300, dtype=torch.bool)
    rk[1, 220:] = False
    pb = fit(net, rp, rmel, rk)
    with torch.no_grad():
        pa = fit(net, rp, rmel, rk)
    assert torch.allclose(pa, pb.detach(), atol=1e-12) and pb.requires_grad and not pa.requires_grad, "学習と推論で当てはめの値が違う"
    p1 = fit(net, rp[1:, :220], rmel[1:, :, :220], rk[1:, :220])
    assert torch.allclose(pa[1:], p1, atol=1e-9), "詰め物の有無で当てはめの値が違う(推論の 1 声と学習の束が一致しない)"
    rp3 = rp.clone()
    rp3[0, 250:] += 5.0
    assert torch.allclose(fit(net, rp3, rmel, rk)[1], pa[1], atol=1e-12), "束の中の別の声が当てはめに漏れる"
    tgt = torch.randn(2, N_MEL, 200, dtype=torch.float64)
    (net(p, pb) - tgt).abs().mean().backward()
    gd = [q.grad.clone() for q in net.parameters()]
    net.zero_grad()
    (net(p, pb.detach()) - tgt).abs().mean().backward()
    assert fit.phi0.grad.abs().sum() > 0 and all(torch.equal(x, q.grad) for x, q in zip(gd, net.parameters())), "外の損失が Φ0 へ流れないか G へ当てはめの中を通って流れる(一階でない)"
    with torch.no_grad():
        fit.a.zero_()
    assert torch.allclose(fit(net, rp, rmel, rk), fit.phi0.expand(2, PHI), atol=1e-12), "歩幅 0 で Φ = Φ0 にならない"
    mk = lambda i: (np.full((N_MEL, 500), i, np.float32), np.zeros((500, P_DIM), np.float32))
    b = VBatches({f"v{j}": {"n": [mk(10 * j + i) for i in range(4)], "v": [mk(10 * j + 9)]} for j in range(3)}, ["v0", "v1", "v2"],
                 np.zeros(P_DIM, np.float32), np.ones(P_DIM, np.float32), 0)
    for _ in range(20):
        s = b.sample(3)
        assert s["nv"].tolist() == [False, True] * 3
        for i in range(3):
            tn, tv = float(s["mel"][2 * i, 0, 0]), float(s["mel"][2 * i + 1, 0, 0])
            refs = set(np.unique(s["rmel"][i][:, s["rmask"][i]].numpy()).tolist())
            j = int(tn // 10)
            assert tv == 10 * j + 9 and tn % 10 != 9, "対象の組み方が違う"
            assert len(refs) == REF_UTT and tn not in refs and tv not in refs and all(int(r // 10) == j for r in refs), "参照に対象・音素の無い発声・別の声が入る"
            assert s["rmask"][i].sum() == REF_UTT * REF_CROP, "参照の切り出しの長さが違う"
    print("unit tests ok(C の因果性・学習と推論・詰め物の有無で当てはめの値が同じ・束の中で声が漏れない・外の損失は Φ0 へ素通しで G へは一階・歩幅 0 で Φ0・参照の組み方)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init", default="")
    ap.add_argument("--gan", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print("prereg", ROOT / "results/diag_snv_a5/prereg.yaml", flush=True)
    print("shipping_gate: 対象外(diag・演奏と描画の代役が非因果)。G は因果(単体検査)", flush=True)
    torch.manual_seed(a.seed)
    unit_tests()
    voices = load_corpus()
    names = sorted(voices)
    val, tr = names[-VAL_VOICES:], names[:-VAL_VOICES]
    if a.init:
        nz = np.load(ROOT / "results" / a.init / "norm.npz")
        mu, sd = nz["mu"], nz["sd"]
    else:
        mu, sd = norm_stats({n: voices[n] for n in tr})
    np.savez(out / "norm.npz", mu=mu, sd=sd)
    print("voices", len(voices), "train", len(tr), "val", len(val), "normal", sum(len(voices[n]["n"]) for n in tr), "nv", sum(len(voices[n]["v"]) for n in tr), flush=True)
    dev = "cuda"
    net, fit = C().to(dev), Fit().to(dev)
    if a.init:
        ck = torch.load(ROOT / "results" / a.init / "last.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(ck["g"])
        fit.load_state_dict(ck["fit"])
    print("params G", sum(p.numel() for p in net.parameters()) / 1e6, "phi0", fit.phi0.numel(), "K", K_FIT, "steps", [round(x, 4) for x in fit.a.tolist()], flush=True)
    lr = a.lr * (0.5 if a.gan else 1.0)
    betas = (0.8, 0.99) if a.gan else (0.9, 0.999)
    opt = torch.optim.AdamW([{"params": net.parameters()}, {"params": fit.parameters()}], lr=lr, betas=betas, weight_decay=0.0)
    warm = 0 if a.gan else 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (warm + 1)) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    D = MultiD().to(dev) if a.gan else None
    dopt = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.8, 0.99)) if a.gan else None
    trq, vab = prefetch(VBatches(voices, tr, mu, sd, a.seed)), VBatches(voices, val, mu, sd, 12345)
    vbs = [vab.sample() for _ in range(16)]
    mel_mean = torch.from_numpy(np.concatenate([m for n in tr for m, _ in voices[n]["n"]], 1).mean(1)).to(dev)

    def evaluate() -> dict:
        net.eval()
        res = {"all": [], "nv": [], "n": [], "triv": [], "phi_rms": [], "enroll_l1": []}
        for b in vbs:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.no_grad():
                phi = fit(net, b["rp"], b["rmel"], b["rmask"])
                y = net(b["p"], phi.repeat_interleave(2, 0))
                w = b["mask"][:, None]
                per = ((y - b["mel"]).abs() * w).sum((1, 2)) / (w.sum((1, 2)) * N_MEL).clamp(min=1.0)
                res["all"].append(float(per.mean()))
                res["nv"] += per[b["nv"]].tolist()
                res["n"] += per[~b["nv"]].tolist()
                res["triv"].append(float(l1(mel_mean[None, :, None].expand_as(b["mel"]), b["mel"], b["mask"])))
                res["phi_rms"] += phi.pow(2).mean(1).sqrt().tolist()
                rw = b["rmask"][:, None].float().clone()
                rw[..., :WARM] = 0.0
                res["enroll_l1"].append(float(((net(b["rp"], phi) - b["rmel"]).abs() * rw).sum() / (rw.sum() * N_MEL)))
        net.train()
        r = {k: float(np.mean(v)) for k, v in res.items()}
        r["phi0_rms"] = float(fit.phi0.detach().pow(2).mean().sqrt())
        return r

    v0 = evaluate()
    print(f"L-4 floors: untrained val L1 {v0['all']:.4f}(n {v0['n']:.4f} / nv {v0['nv']:.4f})/ trivial mean-mel L1 {v0['triv']:.4f} / Φ rms {v0['phi_rms']:.4f}", flush=True)
    log, base = [], v0["all"]
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = {k: v.to(dev, non_blocking=True) for k, v in trq.get().items()}
        phi = fit(net, b["rp"], b["rmel"], b["rmask"])
        y = net(b["p"], phi.repeat_interleave(2, 0))
        rec_l1 = l1(y, b["mel"], b["mask"])
        loss = rec_l1
        info = {}
        if a.gan:
            ok = b["mask"][:, WARM:WARM + CD].bool().all(1)
            if ok.sum() >= 2:
                real, fake = b["mel"][ok][..., WARM:WARM + CD], y[ok][..., WARM:WARM + CD]
                dl = sum(((dr - 1) ** 2).mean() + (df ** 2).mean() for (dr, _), (df, _) in zip(D(real), D(fake.detach())))
                dopt.zero_grad(set_to_none=True)
                dl.backward()
                dopt.step()
                outs_r, outs_f = D(real), D(fake)
                adv = sum(((df - 1) ** 2).mean() for df, _ in outs_f)
                fm = sum(F.l1_loss(f, r.detach()) for (_, fr), (_, ff) in zip(outs_r, outs_f) for r, f in zip(fr, ff))
                loss = 45.0 * rec_l1 + adv + 2.0 * fm
                info = {"d": float(dl.detach()), "adv": float(adv.detach()), "fm": float(fm.detach())}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0 if not a.gan else 100.0)
        gf = torch.nn.utils.clip_grad_norm_(fit.parameters(), 1.0 if not a.gan else 100.0)
        if not (torch.isfinite(gn) and torch.isfinite(gf)):
            print("non-finite grad, skip", step, flush=True)
            continue
        opt.step()
        sched.step()
        if step == 200:
            print(f"speed {(time.time() - t0) / 200:.3f} s/step・peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
        if step % 2500 == 0:
            ev = evaluate()
            r = {"step": step, "train_l1": float(rec_l1.detach()), "val_l1": ev["all"], "val_l1_n": ev["n"], "val_l1_nv": ev["nv"], "val_phi_rms": ev["phi_rms"],
                 "val_enroll_l1": ev["enroll_l1"], "phi0_rms": ev["phi0_rms"], "gn_g": float(gn), "gn_phi0": float(gf), "sec": round(time.time() - t0, 1)} | info
            log.append(r)
            print(json.dumps(r), flush=True)
            if not a.gan and ev["all"] > 1.3 * min(x["val_l1"] for x in log):
                print(f"TRIPWIRE: 検証 L1 {ev['all']:.4f} がそれまでの最小 × 1.3 を超える(発散)→ 中止", flush=True)
                (out / "train_log.json").write_text(json.dumps({"floors": v0, "log": log, "aborted": "diverge"}, indent=1))
                return 2
            if not a.gan and step == 10000 and ev["all"] >= 0.8 * ev["triv"]:
                print(f"TRIPWIRE: step 10k の検証 L1 {ev['all']:.4f} が自明な予測 × 0.8 を下回らない → 中止", flush=True)
                (out / "train_log.json").write_text(json.dumps({"floors": v0, "log": log, "aborted": "tripwire"}, indent=1))
                return 2
            if a.gan and ev["all"] > 1.15 * base:
                print(f"TRIPWIRE: 分布の損失で検証 L1 {ev['all']:.4f} が段 1 の {base:.4f} × 1.15 を超える → 中止", flush=True)
                (out / "train_log.json").write_text(json.dumps({"floors": v0, "log": log, "aborted": "tripwire"}, indent=1))
                return 2
        if step % 10000 == 0 or step == a.steps:
            ck = {"g": net.state_dict(), "fit": fit.state_dict(), "mode": "a5", "step": step}
            if D is not None:
                ck["d"] = D.state_dict()
            torch.save(ck, out / "last.pt")
    (out / "train_log.json").write_text(json.dumps({"floors": v0, "log": log, "names_train": tr, "names_val": val}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

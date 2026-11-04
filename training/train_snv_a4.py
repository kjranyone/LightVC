"""A4(synth_nam_vc §5・diag): 表を使わない連続の回路 c と、表の参考値 x を、普通の発話 + 音素の無い発声(E1 + E1-NV)で学ぶ。
c = A2 の G と同じ因果の膨張畳み込み 12 層で、構造 = 声ごとの層ごとの回路の値 Φ(12 × 512 の FiLM)を G と一緒に学ぶ自動の復号器(符号器なし)。
新しい声は登録(普通の発話だけ)で Φ を当てはめる(A4a の H_film)。Φ は普通の発話の損失からだけ学ぶ(音素の無い発声の損失は Φ へ流さない = 登録と一致)。
x = A2 の x(参照 = 同じ声の普通の発話 2 本・音素の無い発声は参照に入れない)。
段 1 = L1。段 2(--gan)= 段 1 から log-mel の分布の損失(PatchGAN 2 段の LSGAN + 特徴の一致・HiFi-GAN の重み 1 / 2 / 45)を足す質感の微調整。
事前登録 = results/diag_snv_a4/prereg.yaml。演奏と描画の代役が非因果なので diag(製品の判断に使わない)。
    cd training && uv run --no-sync python train_snv_a4.py --mode c --tag diag_snv_a4_c
    cd training && uv run --no-sync python train_snv_a4.py --mode c --tag diag_snv_a4_c_gan --init diag_snv_a4_c --gan --steps 20000
A6 = --hold scream,moan,mixed(その種類の音素の無い発声を学習から抜く・検証の声からも抜く)--prereg results/diag_snv_a6/prereg.yaml。
A7 = --glot(演奏 = SPARC 15 + 声門 9 = PG 24 次元・snv_glot.py・data/snv_e1_feat/glot)--prereg results/diag_snv_a7/prereg.yaml。
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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from snv_glot import G_DIM
from train_snv_a2 import CROP, FEAT, N_MEL, P_DIM, REF_UTT, ROOT, VAL_VOICES, WARM, G, l1, playing

N_LAYER = 12
CH = 256
PHI = N_LAYER * 2 * CH
P_NV = 0.5
PRIOR = 1e-4
CD = 128
FIT_STEPS = 400
FIT_LR = 1e-2


def play(sp: Path, n: int, glot: bool = False) -> np.ndarray:
    p = playing(sp, n)
    if not glot:
        return p
    g = np.load(sp.parent.as_posix().replace("/sparc/", "/glot/", 1) + f"/{sp.stem}.npy")
    assert g.shape == (n, G_DIM), (sp, g.shape, n)
    return np.concatenate([p, g], 1)


def load_corpus(hold: frozenset = frozenset(), glot: bool = False) -> dict:
    man = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
    nv = json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"]
    voices = {}
    for v in man["voices"]:
        n = []
        for f in v["files"]:
            mp, sp = FEAT / f"mel/e1/{v['voice']}/{f['t']}.npy", FEAT / f"sparc/e1/{v['voice']}/{f['t']}.npz"
            if mp.exists() and sp.exists():
                mel = np.load(mp).astype(np.float32)
                n.append((mel, play(sp, mel.shape[1], glot)))
        voices[v["voice"]] = {"n": n, "v": []}
    for it in [it for it in nv if it["k"] == 0 and it["cat"] not in hold]:
        mp, sp = FEAT / f"mel/e1nv/{it['voice']}/{it['cat']}_{it['k']}.npy", FEAT / f"sparc/e1nv/{it['voice']}/{it['cat']}_{it['k']}.npz"
        if it["voice"] in voices and mp.exists() and sp.exists():
            mel = np.load(mp).astype(np.float32)
            voices[it["voice"]]["v"].append((mel, play(sp, mel.shape[1], glot)))
    return {k: d for k, d in voices.items() if len(d["n"]) >= REF_UTT + 1 and d["v"]}


def norm_stats(voices: dict) -> tuple[np.ndarray, np.ndarray]:
    P = np.concatenate([p for d in voices.values() for _, p in d["n"] + d["v"]], 0)
    return P.mean(0).astype(np.float32), (P.std(0) + 1e-4).astype(np.float32)


class Batches:
    def __init__(self, voices: dict, names: list[str], mu: np.ndarray, sd: np.ndarray, seed: int) -> None:
        self.v, self.names, self.mu, self.sd = voices, names, mu, sd
        self.rng = random.Random(seed)

    def sample(self, bs: int) -> dict:
        tm, tp, mask, idx, isnv, rm, rp, rl = [], [], [], [], [], [], [], []
        for _ in range(bs):
            vi = self.rng.randrange(len(self.names))
            d = self.v[self.names[vi]]
            order = list(range(len(d["n"])))
            self.rng.shuffle(order)
            nv = self.rng.random() < P_NV
            if nv:
                mel, p = d["v"][self.rng.randrange(len(d["v"]))]
                refs = [d["n"][i] for i in order[:REF_UTT]]
            else:
                mel, p = d["n"][order[0]]
                refs = [d["n"][i] for i in order[1:1 + REF_UTT]]
            T = mel.shape[1]
            s = self.rng.randint(0, max(0, T - CROP))
            m = np.zeros((N_MEL, CROP), np.float32)
            q = np.zeros((CROP, len(self.mu)), np.float32)
            k = np.zeros(CROP, np.float32)
            n = min(CROP, T - s)
            m[:, :n] = mel[:, s:s + n]
            q[:n] = (p[s:s + n] - self.mu) / self.sd
            k[WARM:n] = 1.0
            tm.append(m), tp.append(q), mask.append(k), idx.append(vi), isnv.append(nv)
            rm.append(np.concatenate([r[0] for r in refs], 1))
            rp.append((np.concatenate([r[1] for r in refs], 0) - self.mu) / self.sd)
            rl.append(rm[-1].shape[1])
        L = max(rl)
        RM = np.zeros((bs, N_MEL, L), np.float32)
        RP = np.zeros((bs, L, len(self.mu)), np.float32)
        RK = np.zeros((bs, L), bool)
        for i, (a, b, l) in enumerate(zip(rm, rp, rl)):
            RM[i, :, :l], RP[i, :l], RK[i, :l] = a, b, True
        return {"mel": torch.from_numpy(np.stack(tm)), "p": torch.from_numpy(np.stack(tp)), "mask": torch.from_numpy(np.stack(mask)),
                "idx": torch.tensor(idx), "nv": torch.tensor(isnv), "rmel": torch.from_numpy(RM), "rp": torch.from_numpy(RP), "rmask": torch.from_numpy(RK)}


class CBlock(nn.Module):
    def __init__(self, ch: int, dil: int) -> None:
        super().__init__()
        self.dil = dil
        self.conv = nn.Conv1d(ch, 2 * ch, 3, dilation=dil)
        self.out = nn.Conv1d(ch, ch, 1)

    def forward(self, h: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
        a, b = self.conv(F.pad(h, (2 * self.dil, 0))).chunk(2, 1)
        g, s = gs.chunk(2, -1)
        return h + self.out(torch.tanh(a * (1 + g[..., None]) + s[..., None]) * torch.sigmoid(b))


class C(nn.Module):
    def __init__(self, ch: int = CH, pdim: int = P_DIM) -> None:
        super().__init__()
        self.inp = nn.Conv1d(pdim, ch, 1)
        self.blocks = nn.ModuleList([CBlock(ch, d) for d in [1, 2, 4, 8, 16, 32] * 2])
        self.head = nn.Conv1d(ch, N_MEL, 1)

    def forward(self, p: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        h = self.inp(p.transpose(1, 2))
        phi = phi.view(phi.shape[0], N_LAYER, 2 * CH)
        for i, blk in enumerate(self.blocks):
            h = blk(h, phi[:, i])
        return self.head(h)


def fit_phi(net: C, rp: torch.Tensor, rmel: torch.Tensor, rmask: torch.Tensor, steps: int = FIT_STEPS) -> torch.Tensor:
    w = rmask[:, None].float().clone()
    w[..., :WARM] = 0.0
    phi = torch.zeros(rp.shape[0], PHI, device=rp.device, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=FIT_LR)
    for _ in range(steps):
        loss = ((net(rp, phi) - rmel).abs() * w).sum() / (w.sum() * N_MEL) + PRIOR * phi.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.zero_grad(set_to_none=True)
    return phi.detach()


class PatchD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sn = spectral_norm
        self.convs = nn.ModuleList([sn(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))), sn(nn.Conv2d(32, 64, (3, 9), stride=2, padding=(1, 4))),
                                    sn(nn.Conv2d(64, 128, (3, 9), stride=2, padding=(1, 4))), sn(nn.Conv2d(128, 256, 3, stride=2, padding=1))])
        self.post = sn(nn.Conv2d(256, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        fm = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fm.append(x)
        return self.post(x), fm


class MultiD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ds = nn.ModuleList([PatchD(), PatchD()])

    def forward(self, mel: torch.Tensor) -> list:
        x = mel[:, None]
        return [self.ds[0](x), self.ds[1](F.avg_pool2d(x, 2))]


def unit_tests() -> None:
    net = C()
    p = torch.randn(2, 200, P_DIM)
    phi = torch.randn(2, PHI) * 0.1
    y1 = net(p, phi)
    p2 = p.clone()
    p2[:, 120:] = torch.randn(2, 80, P_DIM)
    y2 = net(p2, phi)
    assert torch.allclose(y1[..., :120], y2[..., :120], atol=1e-5) and not torch.allclose(y1[..., 120:], y2[..., 120:]), "C が因果でない"
    codes = nn.Embedding(3, PHI, sparse=True)
    nn.init.normal_(codes.weight, std=0.1)
    idx, nvf = torch.tensor([0, 1]), torch.tensor([False, True])
    ph = codes(idx)
    ph = torch.where(nvf[:, None], ph.detach(), ph)
    net(p, ph).abs().mean().backward()
    gr = codes.weight.grad.to_dense()
    assert gr[0].abs().sum() > 0 and gr[1].abs().sum() == 0 and gr[2].abs().sum() == 0, "音素の無い発声の損失が Φ へ流れる"
    b = Batches({"a": {"n": [(np.full((N_MEL, 400), i, np.float32), np.zeros((400, P_DIM), np.float32)) for i in range(4)],
                       "v": [(np.full((N_MEL, 400), 9, np.float32), np.zeros((400, P_DIM), np.float32))]}}, ["a"], np.zeros(P_DIM, np.float32), np.ones(P_DIM, np.float32), 0)
    for _ in range(20):
        s = b.sample(4)
        for i in range(4):
            tv = float(s["mel"][i, 0, 0])
            refs = set(np.unique(s["rmel"][i][:, s["rmask"][i]].numpy()).tolist())
            assert 9.0 not in refs and tv not in refs and bool(s["nv"][i]) == (tv == 9.0), "参照に音素の無い発声か対象の発話が入る"
    print("unit tests ok(C の因果性・Φ の止め・参照の組み方)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["c", "x"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--code_lr", type=float, default=2e-3)
    ap.add_argument("--init", default="")
    ap.add_argument("--gan", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", default="")
    ap.add_argument("--prereg", default="results/diag_snv_a4/prereg.yaml")
    ap.add_argument("--glot", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print("prereg", ROOT / a.prereg, flush=True)
    torch.manual_seed(a.seed)
    unit_tests()
    hold = frozenset(c for c in a.hold.split(",") if c)
    nvm = json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"]
    assert hold <= {it["cat"] for it in nvm}, f"抜く種類の名前が違う: {hold}"
    kept = sorted({it["cat"] for it in nvm} - hold)
    print("hold", sorted(hold), "nv cats in training", kept, flush=True)
    pdim = P_DIM + (G_DIM if a.glot else 0)
    print("playing", "PG = SPARC 15 + glot 9" if a.glot else "P = SPARC 15", "pdim", pdim, flush=True)
    voices = load_corpus(hold, a.glot)
    assert all(p.shape[1] == pdim for d in voices.values() for _, p in d["n"] + d["v"]), "演奏の次元が違う"
    assert sum(len(d["v"]) for d in voices.values()) == sum(1 for it in nvm if it["k"] == 0 and it["cat"] in kept and it["voice"] in voices), "抜いた種類が学習に残る"
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
    net = (C(pdim=pdim) if a.mode == "c" else G("x", pdim=pdim)).to(dev)
    codes = nn.Embedding(len(tr), PHI, sparse=True).to(dev) if a.mode == "c" else None
    if codes is not None:
        nn.init.zeros_(codes.weight)
    if a.init:
        ck = torch.load(ROOT / "results" / a.init / "last.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(ck["g"])
        if codes is not None:
            codes.load_state_dict(ck["codes"])
    print("params", sum(p.numel() for p in net.parameters()) / 1e6, flush=True)
    lr = a.lr * (0.5 if a.gan else 1.0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.8, 0.99) if a.gan else (0.9, 0.999), weight_decay=0.0)
    copt = torch.optim.SparseAdam(codes.parameters(), lr=a.code_lr) if codes is not None else None
    warm = 0 if a.gan else 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (warm + 1)) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    D = MultiD().to(dev) if a.gan else None
    dopt = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.8, 0.99)) if a.gan else None
    trb, vab = Batches(voices, tr, mu, sd, a.seed), Batches(voices, val, mu, sd, 12345)
    vbs = [vab.sample(a.bs) for _ in range(8)]
    mel_mean = torch.from_numpy(np.concatenate([m for n in tr for m, _ in voices[n]["n"]], 1).mean(1)).to(dev)

    def predict(b: dict, train_codes: bool) -> torch.Tensor:
        if a.mode == "x":
            return net(b["p"], net.structure(b["rmel"], b["rp"], b["rmask"]))
        if train_codes:
            ph = codes(b["idx"])
            return net(b["p"], torch.where(b["nv"][:, None], ph.detach(), ph)), ph
        return net(b["p"], fit_phi(net, b["rp"], b["rmel"], b["rmask"], steps=100))

    def evaluate() -> dict:
        net.eval()
        res = {"all": [], "nv": [], "n": [], "triv": []}
        for b in vbs:
            b = {k: v.to(dev) for k, v in b.items()}
            if a.mode == "c":
                y = predict(b, False)
            else:
                with torch.no_grad():
                    y = predict(b, False)
            with torch.no_grad():
                w = b["mask"][:, None]
                per = ((y - b["mel"]).abs() * w).sum((1, 2)) / (w.sum((1, 2)) * N_MEL).clamp(min=1.0)
                res["all"].append(float(per.mean()))
                res["nv"] += per[b["nv"]].tolist()
                res["n"] += per[~b["nv"]].tolist()
                res["triv"].append(float(l1(mel_mean[None, :, None].expand_as(b["mel"]), b["mel"], b["mask"])))
        net.train()
        return {k: float(np.mean(v)) for k, v in res.items()}

    v0 = evaluate()
    print(f"L-4 floors: untrained val L1 {v0['all']:.4f}(n {v0['n']:.4f} / nv {v0['nv']:.4f})/ trivial mean-mel L1 {v0['triv']:.4f}", flush=True)
    log, base = [], v0["all"]
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = {k: v.to(dev, non_blocking=True) for k, v in trb.sample(a.bs).items()}
        if a.mode == "c":
            y, ph = predict(b, True)
            prior = PRIOR * ph.pow(2).mean()
        else:
            y, prior = predict(b, True), torch.zeros((), device=dev)
        rec_l1 = l1(y, b["mel"], b["mask"])
        loss = rec_l1 + prior
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
                loss = 45.0 * rec_l1 + prior + adv + 2.0 * fm
                info = {"d": float(dl), "adv": float(adv), "fm": float(fm)}
        opt.zero_grad(set_to_none=True)
        if copt is not None:
            copt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0 if not a.gan else 100.0)
        if not torch.isfinite(gn):
            print("non-finite grad, skip", step, flush=True)
            continue
        opt.step()
        if copt is not None:
            copt.step()
        sched.step()
        if step % 2500 == 0:
            ev = evaluate()
            r = {"step": step, "train_l1": float(rec_l1), "val_l1": ev["all"], "val_l1_n": ev["n"], "val_l1_nv": ev["nv"], "sec": round(time.time() - t0, 1)} | info
            log.append(r)
            print(json.dumps(r), flush=True)
            if not a.gan and step == 10000 and ev["all"] >= 0.8 * ev["triv"]:
                print(f"TRIPWIRE: step 10k の検証 L1 {ev['all']:.4f} が自明な予測 × 0.8 を下回らない → 中止", flush=True)
                (out / "train_log.json").write_text(json.dumps({"floors": v0, "log": log, "aborted": "tripwire"}, indent=1))
                return 2
            if a.gan and ev["all"] > 1.15 * base:
                print(f"TRIPWIRE: 分布の損失で検証 L1 {ev['all']:.4f} が段 1 の {base:.4f} × 1.15 を超える → 中止", flush=True)
                (out / "train_log.json").write_text(json.dumps({"floors": v0, "log": log, "aborted": "tripwire"}, indent=1))
                return 2
        if step % 10000 == 0 or step == a.steps:
            ck = {"g": net.state_dict(), "mode": a.mode, "step": step, "pdim": pdim}
            if codes is not None:
                ck["codes"] = codes.state_dict()
            if D is not None:
                ck["d"] = D.state_dict()
            torch.save(ck, out / "last.pt")
    (out / "train_log.json").write_text(json.dumps({"floors": v0, "log": log, "names_train": tr, "names_val": val}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

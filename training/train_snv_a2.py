"""A2(synth_nam_vc §5・diag): 構造の表現の上限。自己の reamp で 演奏(SPARC の特徴の代役)+ 構造(参照)→ BigVGAN の log-mel を回帰する因果の描画器 G。
構造 g = 参照から作る 1 本の静的なベクトル(FiLM)。x = g + 参照を調音の状態で引く注意(鍵 = 参照の演奏・値 = 参照の mel の細い符号 16 次元 × 4 頭)。
xs = x の問いを内容によらない学習の定数に替えた対照(同じ経路・同じ値で、読みが内容で変わらない = 静的な注意の要約)。
事前登録 = results/diag_snv_a2_<mode>/prereg.yaml(L/I/C)。演奏と描画の代役が非因果なので diag(製品の判断に使わない)。
    cd training && uv run --no-sync python train_snv_a2.py --mode g --tag diag_snv_a2_g
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

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/snv_e1_feat"
MEL_HZ = 44100 / 512
SPARC_HZ = 50.0
P_DIM = 15
N_MEL = 128
CROP = 344
WARM = 86
REF_UTT = 2
VAL_VOICES = 30


def playing(npz: Path, n_mel: int) -> np.ndarray:
    d = np.load(npz)
    raw = np.concatenate([d["ema"], np.log(np.maximum(d["pitch"], 1.0))[:, None], d["loudness"][:, None], d["periodicity"][:, None]], 1)
    t_src = np.arange(len(raw)) / SPARC_HZ
    t_dst = np.arange(n_mel) / MEL_HZ
    return np.stack([np.interp(t_dst, t_src, raw[:, k]) for k in range(P_DIM)], 1).astype(np.float32)


def load_corpus() -> dict:
    man = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
    voices = {}
    for v in man["voices"]:
        utts = []
        for f in v["files"]:
            mp = FEAT / f"mel/e1/{v['voice']}/{f['t']}.npy"
            sp = FEAT / f"sparc/e1/{v['voice']}/{f['t']}.npz"
            if mp.exists() and sp.exists():
                mel = np.load(mp).astype(np.float32)
                utts.append((mel, playing(sp, mel.shape[1])))
        if len(utts) >= REF_UTT + 1:
            voices[v["voice"]] = utts
    return voices


def norm_stats(voices: dict) -> tuple[np.ndarray, np.ndarray]:
    P = np.concatenate([p for us in voices.values() for _, p in us], 0)
    return P.mean(0).astype(np.float32), (P.std(0) + 1e-4).astype(np.float32)


class Batches:
    def __init__(self, voices: dict, names: list[str], mu: np.ndarray, sd: np.ndarray, seed: int) -> None:
        self.v, self.names, self.mu, self.sd = voices, names, mu, sd
        self.rng = random.Random(seed)

    def sample(self, bs: int) -> dict:
        tm, tp, mask, rm, rp, rl = [], [], [], [], [], []
        for _ in range(bs):
            name = self.rng.choice(self.names)
            us = self.v[name]
            idx = list(range(len(us)))
            self.rng.shuffle(idx)
            mel, p = us[idx[0]]
            T = mel.shape[1]
            s = self.rng.randint(0, max(0, T - CROP))
            m = np.zeros((N_MEL, CROP), np.float32)
            q = np.zeros((CROP, P_DIM), np.float32)
            k = np.zeros(CROP, np.float32)
            n = min(CROP, T - s)
            m[:, :n] = mel[:, s:s + n]
            q[:n] = (p[s:s + n] - self.mu) / self.sd
            k[WARM:n] = 1.0
            tm.append(m), tp.append(q), mask.append(k)
            refs = [us[i] for i in idx[1:1 + REF_UTT]]
            rm.append(np.concatenate([r[0] for r in refs], 1))
            rp.append((np.concatenate([r[1] for r in refs], 0) - self.mu) / self.sd)
            rl.append(rm[-1].shape[1])
        L = max(rl)
        RM = np.zeros((bs, N_MEL, L), np.float32)
        RP = np.zeros((bs, L, P_DIM), np.float32)
        RK = np.zeros((bs, L), bool)
        for i, (a, b, l) in enumerate(zip(rm, rp, rl)):
            RM[i, :, :l], RP[i, :l], RK[i, :l] = a, b, True
        return {"mel": torch.from_numpy(np.stack(tm)), "p": torch.from_numpy(np.stack(tp)), "mask": torch.from_numpy(np.stack(mask)),
                "rmel": torch.from_numpy(RM), "rp": torch.from_numpy(RP), "rmask": torch.from_numpy(RK)}


class Block(nn.Module):
    def __init__(self, ch: int, dil: int, zdim: int) -> None:
        super().__init__()
        self.dil = dil
        self.conv = nn.Conv1d(ch, 2 * ch, 3, dilation=dil)
        self.film = nn.Linear(zdim, 2 * ch)
        self.out = nn.Conv1d(ch, ch, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        a, b = self.conv(F.pad(h, (2 * self.dil, 0))).chunk(2, 1)
        g, s = self.film(z).chunk(2, -1)
        return h + self.out(torch.tanh(a * (1 + g[..., None]) + s[..., None]) * torch.sigmoid(b))


class G(nn.Module):
    def __init__(self, mode: str, ch: int = 256, zdim: int = 256, heads: int = 4, dk: int = 64, dv: int = 16) -> None:
        super().__init__()
        self.mode, self.heads, self.dk, self.dv = mode, heads, dk, dv
        self.inp = nn.Conv1d(P_DIM, ch, 1)
        dils = [1, 2, 4, 8, 16, 32] * 2
        self.blocks = nn.ModuleList([Block(ch, d, zdim) for d in dils])
        self.head = nn.Conv1d(ch, N_MEL, 1)
        self.genc = nn.Sequential(nn.Conv1d(N_MEL, ch, 5, padding=2), nn.ReLU(), nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(), nn.Conv1d(ch, ch, 5, padding=2))
        self.gz = nn.Linear(ch, zdim)
        if mode in ("x", "xs"):
            self.q = nn.Linear(ch, heads * dk) if mode == "x" else nn.Parameter(torch.randn(heads * dk) / math.sqrt(dk))
            self.k = nn.Sequential(nn.Linear(P_DIM, ch), nn.ReLU(), nn.Linear(ch, heads * dk))
            self.venc = nn.Sequential(nn.Conv1d(N_MEL, ch, 3, padding=1), nn.ReLU(), nn.Conv1d(ch, heads * dv, 3, padding=1))
            self.o = nn.Linear(heads * dv, ch)

    def structure(self, rmel: torch.Tensor, rp: torch.Tensor, rmask: torch.Tensor) -> dict:
        e = self.genc(rmel)
        w = rmask[:, None].float()
        z = self.gz((e * w).sum(-1) / w.sum(-1).clamp(min=1.0))
        out = {"z": z}
        if self.mode != "g":
            B, Tr = rmask.shape
            out["k"] = self.k(rp).view(B, Tr, self.heads, self.dk).transpose(1, 2)
            out["v"] = self.venc(rmel).transpose(1, 2).reshape(B, Tr, self.heads, self.dv).transpose(1, 2)
            out["rmask"] = rmask
        return out

    def forward(self, p: torch.Tensor, s: dict, return_attn: bool = False):
        h = self.inp(p.transpose(1, 2))
        attn = None
        for i, blk in enumerate(self.blocks):
            h = blk(h, s["z"])
            if self.mode != "g" and i == 3:
                B, C, T = h.shape
                if self.mode == "x":
                    q = self.q(h.transpose(1, 2)).view(B, T, self.heads, self.dk).transpose(1, 2)
                else:
                    q = self.q.view(1, self.heads, 1, self.dk).expand(B, self.heads, T, self.dk)
                logit = q @ s["k"].transpose(-1, -2) / math.sqrt(self.dk)
                logit = logit.masked_fill(~s["rmask"][:, None, None, :], -1e9)
                attn = logit.softmax(-1)
                ctx = (attn @ s["v"]).transpose(1, 2).reshape(B, T, self.heads * self.dv)
                h = h + self.o(ctx).transpose(1, 2)
        y = self.head(h)
        return (y, attn) if return_attn else y


def l1(y: torch.Tensor, mel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask[:, None]
    return ((y - mel).abs() * w).sum() / (w.sum() * N_MEL).clamp(min=1.0)


def unit_tests() -> None:
    t_src = np.arange(150) / SPARC_HZ
    t_dst = np.arange(int(2 * MEL_HZ)) / MEL_HZ
    ramp = np.interp(t_dst, t_src, t_src)
    assert np.allclose(ramp, t_dst, atol=1e-6), "補間の時刻軸がずれる"
    for mode in ("x", "xs"):
        g = G(mode)
        p = torch.randn(1, 200, P_DIM)
        s = g.structure(torch.randn(1, N_MEL, 50), torch.randn(1, 50, P_DIM), torch.ones(1, 50, dtype=torch.bool))
        y1, a1 = g(p, s, return_attn=True)
        p2 = p.clone()
        p2[:, 120:] = torch.randn(1, 80, P_DIM)
        y2 = g(p2, s)
        assert torch.allclose(y1[..., :120], y2[..., :120], atol=1e-5) and not torch.allclose(y1[..., 120:], y2[..., 120:]), f"G({mode}) が因果でない"
        assert (mode == "xs") == bool(torch.allclose(a1, a1[:, :, :1].expand_as(a1))), f"G({mode}) の読みが内容で変わる/変わらない の取り違え"
    print("unit tests ok(補間の時刻軸・G の因果性)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["g", "x", "xs"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print("prereg", out / "prereg.yaml", flush=True)
    torch.manual_seed(a.seed)
    unit_tests()
    voices = load_corpus()
    names = sorted(voices)
    val = names[-VAL_VOICES:]
    tr = names[:-VAL_VOICES]
    mu, sd = norm_stats({n: voices[n] for n in tr})
    np.savez(out / "norm.npz", mu=mu, sd=sd)
    print("voices", len(voices), "train", len(tr), "val", len(val), "utts", sum(len(voices[n]) for n in tr), flush=True)
    dev = "cuda"
    g = G(a.mode).to(dev)
    print("params", sum(p.numel() for p in g.parameters()) / 1e6, flush=True)
    opt = torch.optim.AdamW(g.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    trb, vab = Batches(voices, tr, mu, sd, a.seed), Batches(voices, val, mu, sd, 12345)
    vbs = [vab.sample(a.bs) for _ in range(8)]
    mel_mean = torch.from_numpy(np.concatenate([m for n in tr for m, _ in voices[n]], 1).mean(1)).to(dev)

    def evaluate() -> tuple[float, float]:
        g.eval()
        with torch.no_grad():
            ls, triv = [], []
            for b in vbs:
                b = {k: v.to(dev) for k, v in b.items()}
                ls.append(float(l1(g(b["p"], g.structure(b["rmel"], b["rp"], b["rmask"])), b["mel"], b["mask"])))
                triv.append(float(l1(mel_mean[None, :, None].expand_as(b["mel"]), b["mel"], b["mask"])))
        g.train()
        return float(np.mean(ls)), float(np.mean(triv))

    v0, triv = evaluate()
    print(f"L-4 floors: untrained val L1 {v0:.4f} / trivial mean-mel L1 {triv:.4f}", flush=True)
    log = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = {k: v.to(dev, non_blocking=True) for k, v in trb.sample(a.bs).items()}
        loss = l1(g(b["p"], g.structure(b["rmel"], b["rp"], b["rmask"])), b["mel"], b["mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        if not torch.isfinite(gn):
            print("non-finite grad, skip", step, flush=True)
            continue
        opt.step()
        sched.step()
        if step % 500 == 0:
            vl, _ = evaluate()
            rec = {"step": step, "train_l1": float(loss), "val_l1": vl, "sec": time.time() - t0}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if step == 10000 and vl >= 0.8 * triv:
                print(f"TRIPWIRE: step 10k の検証 L1 {vl:.4f} が自明な予測 × 0.8 = {0.8 * triv:.4f} を下回らない → 中止", flush=True)
                (out / "train_log.json").write_text(json.dumps({"floors": {"untrained": v0, "trivial": triv}, "log": log, "aborted": "tripwire"}, indent=1))
                return 2
        if step % 10000 == 0 or step == a.steps:
            torch.save({"g": g.state_dict(), "mode": a.mode, "step": step}, out / "last.pt")
    (out / "train_log.json").write_text(json.dumps({"floors": {"untrained": v0, "trivial": triv}, "log": log}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

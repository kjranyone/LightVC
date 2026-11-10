"""P13(synth_nam_vc §8.3 rev5.2・予測 P-11): 話者を跨いだ対応の手がかり = 同じ文を別の声が読んだ組(ペア構成・時刻を揃えない)で演奏と回路の分け方を決める。
演奏 a = E(y, 有声; S_操作者)(P11 の EncS = 操作者自身の参照を読む回路の逆・平均だけ使う決定的な演奏)・回路 C = 表の型 x(目標の参照を読む)。
損失 = 自己の reamp の L1(P3 と同じ)+ λ × 同じ文の組の演奏のフレームの袋の MMD²(有声のフレーム・多重の帯域の RBF・帯域は束の中の距離の中央値 × {1/4, 1, 4}・勾配を止める)。
組 = E1 の学習の声の中で同じ文を読んだ別の声(1665 組・男女を跨ぐ 547)。各声の参照 = その声の別の普通の発話 2 本(組の発話を除く)。時刻の揃え・1 対 1 の音響の回帰・音素は使わない。
事前登録 = results/snv_p13/prereg.yaml。
    cd training && uv run --no-sync python train_snv_p13.py --tag snv_p13_m --prereg results/snv_p13/prereg.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = Path(__file__).resolve().parent.parent
N_PAIR = 4
MMD_FR = 256
LAM = 0.1


def text_pairs(names: list[str]) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    m = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
    keep = set(names)
    by: dict = {}
    for v in m["voices"]:
        if v["voice"] in keep:
            for f in v["files"]:
                by.setdefault(f["text"], []).append((v["voice"], int(f["t"][1:])))
    return [(a, b) for us in by.values() for a, b in itertools.combinations(us, 2) if a[0] != b[0]]


def mmd2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.cat([x, y])
    d = torch.cdist(z, z).pow(2)
    med = d.detach()[d.detach() > 0].median().clamp(min=1e-6)
    k = sum(torch.exp(-d / (med * s)) for s in (0.25, 1.0, 4.0))
    n = len(x)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


class PairBatches:
    def __init__(self, corpus: dict, pairs: list, seed: int) -> None:
        self.c, self.pairs, self.rng = corpus, pairs, random.Random(seed)

    def _ref(self, voice: str, skip: int) -> tuple[np.ndarray, np.ndarray]:
        idx = [i for i in range(len(self.c[voice]["norm"])) if i != skip]
        ys, vs = [], []
        for i in self.rng.sample(idx, T.REF_UTT):
            u = self.c[voice]["norm"][i]
            n = u["y"].shape[1]
            s = self.rng.randint(0, max(0, n - T.REF_MAX))
            ys.append(u["y"][:, s:s + T.REF_MAX]), vs.append(u["v"][s:s + T.REF_MAX])
        return np.concatenate(ys, 1), np.concatenate(vs)

    def sample(self, n: int) -> dict:
        items = [x for p in self.rng.sample(self.pairs, n) for x in p]
        L = max(self.c[v]["norm"][j]["y"].shape[1] for v, j in items)
        refs = [self._ref(v, j) for v, j in items]
        Lr = max(r[0].shape[1] for r in refs)
        B = len(items)
        Y, V, OK = np.zeros((B, T.D_Y, L), np.float32), np.zeros((B, L), np.float32), np.zeros((B, L), np.float32)
        RY, RV, RM = np.zeros((B, T.D_Y, Lr), np.float32), np.zeros((B, Lr), np.float32), np.zeros((B, Lr), bool)
        for i, ((v, j), (ry, rv)) in enumerate(zip(items, refs)):
            u = self.c[v]["norm"][j]
            k = u["y"].shape[1]
            Y[i, :, :k], V[i, :k], OK[i, :k] = u["y"], u["v"], 1.0
            RY[i, :, :ry.shape[1]], RV[i, :ry.shape[1]], RM[i, :ry.shape[1]] = ry, rv, True
        t = torch.from_numpy
        return {"y": t(Y), "v": t(V), "ok": t(OK), "ry": t(RY), "rv": t(RV), "rmask": t(RM)}


def pair_loss(E: P11.EncS, b: dict, g: torch.Generator) -> torch.Tensor:
    sE = E.structure(P11.xin(b["ry"], b["rv"]), b["rmask"])
    a = E(P11.xin(b["y"], b["v"]), sE)[0]
    m = (b["ok"] * b["v"]) > 0
    m[:, :T.WARM_SELF] = False
    out = []
    for i in range(0, a.shape[0], 2):
        xs = []
        for j in (i, i + 1):
            f = a[j].T[m[j]]
            sel = torch.randperm(len(f), generator=g, device=f.device)[:MMD_FR]
            xs.append(f[sel])
        out.append(mmd2(xs[0], xs[1]))
    return torch.stack(out).mean()


def unit_tests() -> None:
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    assert float(mmd2(x, x)) < 1e-6, "同じ袋で MMD が 0 でない"
    assert float(mmd2(x, x + 3.0)) > float(mmd2(x, torch.randn(200, 8))) + 0.1, "ずれた袋の MMD が大きくならない"
    P11.unit_tests()
    print("unit tests ok(MMD の恒等とずれ・P11 の E)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default="results/snv_p13/prereg.yaml")
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (ROOT / a.prereg).exists():
        print(f"{a.prereg} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, flush=True)
    torch.manual_seed(a.seed)
    unit_tests()
    t0 = time.time()
    corpus = T.load_corpus()
    tr, va = T.split(sorted(corpus))
    assert all(len(corpus[v]["norm"]) == 10 for v in tr), "普通の発話が t00〜t09 の 10 本でない声がある(組の添字がずれる)"
    pairs = text_pairs(tr)
    print(f"voices {len(corpus)} train {len(tr)} val {len(va)} pairs {len(pairs)} load {time.time() - t0:.0f}s", flush=True)
    (out / "split.json").write_text(json.dumps({"train": tr, "val": va}))
    dev = "cuda"
    E, C = P11.EncS().to(dev), T.Circ().to(dev)
    params = list(E.parameters()) + list(C.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = 200 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    trb, vab, pb = T.Batches(corpus, tr, a.seed), T.Batches(corpus, va, 12345, p_nv=0.0), PairBatches(corpus, pairs, a.seed)
    vbatches = [vab.sample(a.bs) for _ in range(8)]
    vpairs = [PairBatches(corpus, pairs, 777).sample(N_PAIR) for _ in range(4)]
    to_dev = lambda b: {k: x.to(dev, non_blocking=True) for k, x in b.items()}
    g = torch.Generator(device=dev).manual_seed(0)

    @torch.no_grad()
    def evaluate() -> dict:
        E.eval(), C.eval()
        r: dict = {k: [] for k in ("val_l1", "trivial", "fwd", "amax", "pair_mmd")}
        for vb in vbatches:
            b = to_dev(vb)
            m = b["ok"].clone()
            m[:, :T.WARM_SELF] = 0
            sE, sC = P11.structures(E, C, b["ry"], b["rv"], b["rmask"])
            aa = E(P11.xin(b["y"], b["v"]), sE)[0]
            r["val_l1"].append(float(T.masked_l1(C(aa, b["lf"], b["v"], sC), b["y"], m)))
            rmean = (b["ry"] * b["rmask"][:, None]).sum(-1, keepdim=True) / b["rmask"].sum(-1)[:, None, None].clamp(min=1)
            r["trivial"].append(float(T.masked_l1(rmean.expand_as(b["y"]), b["y"], m)))
            perm = torch.roll(torch.arange(aa.shape[0], device=dev), 1)
            r["fwd"].append(float(T.masked_l1(C(aa, b["lf"], b["v"], {k: x[perm] for k, x in sC.items()}), b["y"], m)))
            h = C.inp(torch.cat([aa, b["lf"][:, None], b["v"][:, None]], 1))
            for i, blk in enumerate(C.blocks):
                h = blk(h, sC["z"])
                if i == 3:
                    B_, Ch, L = h.shape
                    q = C.q(h.transpose(1, 2)).view(B_, L, C.heads, C.dk).transpose(1, 2)
                    p = ((q @ sC["k"].transpose(-1, -2)) / math.sqrt(C.dk)).masked_fill(~sC["rmask"][:, None, None, :], -1e9).softmax(-1)
                    r["amax"].append(float(p.max(-1).values.mean()))
                    break
        gv = torch.Generator(device=dev).manual_seed(1)
        for vp in vpairs:
            r["pair_mmd"].append(float(pair_loss(E, to_dev(vp), gv)))
        E.train(), C.train()
        return {k: float(np.mean(x)) for k, x in r.items()}

    ev0 = evaluate()
    print(f"floors: untrained val L1 {ev0['val_l1']:.4f} / trivial (参照の平均の Y) {ev0['trivial']:.4f} / pair MMD {ev0['pair_mmd']:.4f}", flush=True)
    log = {"floors": {"untrained": ev0["val_l1"], "trivial": ev0["trivial"], "pair_mmd": ev0["pair_mmd"]}, "evals": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        b = to_dev(trb.sample(a.bs))
        m = b["ok"].clone()
        m[:, :T.WARM_SELF] = 0
        sE, sC = P11.structures(E, C, b["ry"], b["rv"], b["rmask"])
        l_self = T.masked_l1(C(E(P11.xin(b["y"], b["v"]), sE)[0], b["lf"], b["v"], sC), b["y"], m)
        l_pair = pair_loss(E, to_dev(pb.sample(N_PAIR)), g)
        loss = l_self + LAM * l_pair
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} self {float(l_self):.4f} pair_mmd {float(l_pair):.4f} gn {float(gn):.2f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 2000 == 0 or step == steps:
            ev = evaluate()
            rec = {"step": step, **ev, "train_self": float(l_self), "train_pair": float(l_pair), "sec": time.time() - t0}
            log["evals"].append(rec)
            print("eval", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            ck = {"Es": E.state_dict(), "C": C.state_dict(), "step": step}
            torch.save(ck, out / "last.pt")
            if step == steps // 2:
                torch.save(ck, out / f"step{step // 1000}k.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
            if ev["amax"] > 0.5:
                print(f"STOP: 注意が検索に寄る(最大 {ev['amax']:.3f})", flush=True)
                return 3
            if step == 10000 and ev["val_l1"] > 0.8 * ev["trivial"]:
                print(f"TRIPWIRE: val L1 {ev['val_l1']:.4f} > 0.8 × trivial {ev['trivial']:.4f} at 10k → stop", flush=True)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

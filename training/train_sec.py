"""条件空間の話者照合器 SE_c(current/ref_reader.md §7)。出力部への条件の軌道(CheapTrick 包絡 c1..c24 / 10 と相対 c0・2 秒)から話者を当てる。
R の同一性の損失の批評家(凍結)に使う。学習 = data/rr_feat の学習用の話者(女声 TTS の評価外 100 人を除く + 学習用の男声 m030〜m045)・AAM-softmax。
検証: (1) 評価外の女声 100 人で 9 文の平均を中心・残り 1 文で top-1(本人 vs 99 人)。(2) 物差しとの向き: 検証の組(train_rr と同じ固定 96 組)で
目標の参照の埋め込みとの cos を 本物 Y・表 TAB・R(--rr の重み)で比べる(評価の物差し D132 では Y > TAB > R)。
    uv run python train_sec.py --tag diag_sec1 --steps 8000 --rr diag_rr1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent
WIN = 400


def feats(env: np.ndarray, c0: np.ndarray | None = None) -> np.ndarray:
    """env [25, T](生の単位・行 0 = CheapTrick の c0)または env24 [24, T] + c0n [T] → [25, T](c1..c24 / 10・c0n − 平均)。"""
    if c0 is None:
        c0 = (env[0] - RRm.C0_SIL) / 10
        env = env[1:25]
    return np.concatenate([env / 10, (c0 - c0.mean())[None]], 0).astype(np.float32)


class SEc(nn.Module):
    def __init__(self, ch: int = 256, emb: int = 128):
        super().__init__()
        self.cfg = {"ch": ch, "emb": emb}
        L = []
        ci = 25
        for d in (1, 2, 4, 8, 1):
            L += [nn.Conv1d(ci, ch, 3, padding=d, dilation=d), nn.LeakyReLU(0.1)]
            ci = ch
        self.net = nn.Sequential(*L)
        self.att = nn.Conv1d(ch, 1, 1)
        self.out = nn.Linear(2 * ch, emb)

    def forward(self, x: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
        h = self.net(x)
        a = self.att(h)
        if m is not None:
            a = a.masked_fill(~m[:, None], -1e4)
        w = a.softmax(-1)
        mu = (h * w).sum(-1)
        sd = ((h - mu[..., None]) ** 2 * w).sum(-1).clamp(min=1e-6).sqrt()
        return F.normalize(self.out(torch.cat([mu, sd], 1)), dim=1)


class AAM(nn.Module):
    def __init__(self, emb: int, n: int, m: float = 0.2, s: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, emb) * 0.01)
        self.m, self.s = m, s

    def forward(self, e: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = e @ F.normalize(self.W, dim=1).T
        th = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        logit = torch.where(F.one_hot(y, cos.shape[1]).bool(), torch.cos(th + self.m), cos) * self.s
        return F.cross_entropy(logit, y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--rr", default="diag_rr1")
    a = ap.parse_args()
    od = ROOT / "results" / a.tag
    od.mkdir(parents=True, exist_ok=True)
    assert (od / "prereg.yaml").exists(), "事前登録が無い"
    log = open(od / "train.log", "a")
    print(f"prereg: {od / 'prereg.yaml'}", file=log, flush=True)
    dev = "cuda"
    torch.manual_seed(0)
    st = RRm.Store()
    held = RRm.held_females()
    fems = sorted(s for s in {s for (c, s, _) in st.f if c == "fem_neutral"} if all(("fem_neutral", s, f"t{t:02d}_neutral") in st.f for t in range(10)))
    tr = [s for s in fems if s not in held] + [m for m in sorted({s for (c, s, _) in st.f if c == "male_ref"}) if m not in RRm.EVAL_MALES and m not in RRm.VAL_MALES]
    va = [s for s in fems if s in held]
    files = {s: [k for k in st.f if k[1] == s and k[0] in ("fem_neutral", "male_ref")] for s in tr}
    spk_id = {s: i for i, s in enumerate(tr)}
    net = SEc().to(dev)
    head = AAM(128, len(tr)).to(dev)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
    rng = random.Random(0)

    def batch() -> tuple:
        X, Y = [], []
        for _ in range(a.bs):
            s = rng.choice(tr)
            d = st.f[rng.choice(files[s])]
            x = feats(d["env"])
            if x.shape[1] > WIN:
                o = rng.randrange(0, x.shape[1] - WIN)
                x = x[:, o:o + WIN]
            else:
                x = np.pad(x, ((0, 0), (0, WIN - x.shape[1])), mode="wrap")
            X.append(x); Y.append(spk_id[s])
        return torch.from_numpy(np.stack(X)).to(dev), torch.tensor(Y, device=dev)

    @torch.no_grad()
    def validate() -> dict:
        net.eval()
        C, Q = [], []
        for s in va:
            es = [net(torch.from_numpy(feats(st.f[("fem_neutral", s, f"t{t:02d}_neutral")]["env"]))[None].to(dev))[0] for t in range(10)]
            C.append(F.normalize(torch.stack(es[:9]).mean(0), dim=0)); Q.append(es[9])
        C, Q = torch.stack(C), torch.stack(Q)
        top1 = float(((Q @ C.T).argmax(1) == torch.arange(len(va), device=dev)).float().mean())
        net.train()
        return {"val_top1_100": round(top1, 3)}

    t0 = time.time()
    for step in range(1, a.steps + 1):
        x, y = batch()
        loss = head(net(x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0:
            print(json.dumps({"step": step, "loss": round(float(loss.detach()), 4), "s_per_step": round((time.time() - t0) / step, 3)}), file=log, flush=True)
        if step % 2000 == 0 or step == a.steps:
            v = validate()
            print("val", step, json.dumps(v), file=log, flush=True); print("val", step, v, flush=True)
            torch.save({"net": net.state_dict(), "cfg": net.cfg, "step": step}, od / "last.pt")
    net.eval()
    ck = torch.load(ROOT / "results" / a.rr / "last.pt", map_location="cpu", weights_only=False)
    rr = RRm.RR(**{k: ck["cfg"][k] for k in ("nq", "ch", "heads", "dk")}).to(dev).eval(); rr.load_state_dict(ck["net"])
    va_m = [m for m in sorted({s for (c, s, _) in st.f if c == "male_par"}) if m in RRm.VAL_MALES and m in st.stats]
    vds = RRm.Pairs(st, va, va_m, 96, 0.5, 1, fixed=True)
    rows = []
    with torch.no_grad():
        for i in range(len(vds)):
            it = vds[i]
            b = RRm.collate([it]); b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            if ck["cfg"].get("no_op_env"):
                b["q"][:, RRm.K:RRm.K + 24] = 0
            env, dc0, _ = rr(b["q"], b["tab"], rr.memory(b["rp"], b["rv"]), b["rm"])
            A = it["A"]
            ref = F.normalize(torch.stack([net(torch.from_numpy(feats(st.f[("fem_neutral", A, f"t{t:02d}_neutral")]["env"]))[None].to(dev))[0] for t in range(10)]).mean(0), dim=0)
            e_y = net(torch.from_numpy(feats(it["y"], it["yc0"]))[None].to(dev))[0]
            e_t = net(torch.from_numpy(feats(it["tab"], it["c0b"]))[None].to(dev))[0]
            e_r = net(torch.from_numpy(feats(env[0].cpu().numpy(), (b["c0b"] + dc0)[0].cpu().numpy()))[None].to(dev))[0]
            rows.append({"male": it["male"], "Y": float(ref @ e_y), "TAB": float(ref @ e_t), "RR": float(ref @ e_r)})
    out = {}
    for kind in (False, True):
        r = [x for x in rows if x["male"] == kind]
        if r:
            out["mf" if kind else "ff"] = {c: round(float(np.median([x[c] for x in r])), 3) for c in ("Y", "TAB", "RR")} | {"RR_gt_TAB": round(float(np.mean([x["RR"] > x["TAB"] for x in r])), 3), "n": len(r)}
    print("direction", json.dumps(out), file=log, flush=True); print("direction", out, flush=True)
    (od / "direction.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

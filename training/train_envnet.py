"""構音の包絡の内容の網(D637・diag_envnet1): [C1 の事後 K・c0n・log f0・有声・表の引き Σp T / 10・C2 の特徴] → 包絡の行 24 のずれ(表の引きに足す)。
因果(左寄せの畳み込み・先読み 0)。損失 = 実女声の自己再構成で、本人の CheapTrick の包絡の行(c1..c24 / 10)との L1(発声のフレーム)。生成器は使わない。
入力の側の性別の摂動(--sexpert・D434)と録音の経路の摂動(--pert・D400)で事後・C2・c0n を作る(目標は実録音の包絡のまま)。規則は results/<tag>/prereg.yaml。
    uv run python train_envnet.py --tag diag_envnet1 [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import nvoc as N
import pae as PA
import rvoc as R
import train_ppgvoc as P
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent
K = P.K
D_C2 = 64
WARM = 100


class EnvNet(torch.nn.Module):
    def __init__(self, ch: int = 192, dils: tuple = (1, 2, 4, 8, 16, 32, 64), d_c2: int = D_C2):
        super().__init__()
        self.inp = R.CConv(K + 3 + 24 + d_c2, ch, 1)
        self.blocks = torch.nn.ModuleList(R.CConv(ch, ch, 3, d) for d in dils)
        self.out = torch.nn.Conv1d(ch, 24, 1)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, p: torch.Tensor, c0n: torch.Tensor, lf: torch.Tensor, vv: torch.Tensor, tab: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        h = self.inp(torch.cat([p, c0n, lf, vv, tab, c2], 1))
        for b in self.blocks:
            h = h + b(F.leaky_relu(h, 0.1))
        return tab + self.out(F.leaky_relu(h, 0.1))


def causal_check(net: EnvNet, xs: tuple, t: int = 300) -> float:
    with torch.no_grad():
        y0 = net(*xs)
        xs2 = tuple(torch.cat([x[..., :t], x[..., t:] + torch.randn_like(x[..., t:])], -1) for x in xs)
        y1 = net(*xs2)
    return float((y0[..., :t] - y1[..., :t]).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="diag_envnet1")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seg_mult", type=int, default=4)
    ap.add_argument("--sexpert", type=float, default=0.5)
    ap.add_argument("--pert", type=float, default=0.0)
    ap.add_argument("--init", default="")
    ap.add_argument("--seed", type=int, default=637)
    ap.add_argument("--cv_noise", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_min", type=float, default=55.0)
    ap.add_argument("--trip_step", type=int, default=1000)
    ap.add_argument("--trip_ratio", type=float, default=0.90)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (out / "prereg.yaml").exists():
        print(f"results/{a.tag}/prereg.yaml が無い: 起動しない", flush=True)
        return 1
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    TR.SEG = 33600 * a.seg_mult
    TR.T = TR.SEG // N.HOP
    SEG, T = TR.SEG, TR.T
    held = P.e2e_held()
    rows, spks, counts = P.load_rows({h[0] for h in held}, True)
    print("data", json.dumps(counts), "区間", T, "フレーム", flush=True)
    (out / "data_counts.json").write_text(json.dumps(counts, indent=1))
    spk_idx = {s: i for i, s in enumerate(spks)}
    TAB = ROOT / "data/rvoc_tables_real"
    tabT = torch.from_numpy(np.stack([np.load(TAB / f"{s}.npz")["T"] for s in spks])).float().to(dev)
    import c1_content as CC
    import train_c1 as T1
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"]); c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    P.cvin_setup(str(ROOT / "results/c2_1/last.pt"), dev)
    assert P.CVIN["d"] == D_C2
    sm = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    front = TR.Front("pae", TR.ckpt_env_smooth(sm, None)).to(dev)
    net = EnvNet().to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["net"])
        print("初期値", a.init, flush=True)
    print("params (M)", round(sum(p.numel() for p in net.parameters()) / 1e6, 3), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    loader = iter(torch.utils.data.DataLoader(TR.DS(rows, rows, a.seed, spk_idx=spk_idx, ctx=T1.PRE, pert=a.pert, sexpert=a.sexpert), batch_size=a.bs, num_workers=a.workers,
                                              pin_memory=True, persistent_workers=True, prefetch_factor=4))

    def batch() -> tuple:
        bt = [t.to(dev, non_blocking=True) for t in next(loader)]
        xa, f0, env, xl, si = bt[:5]
        xp = bt[5] if len(bt) > 5 else xa
        env_o = bt[6] if len(bt) > 6 else env
        with torch.no_grad():
            ct = front(xp, f0, env)
            ct_o = ct if env_o is env else front(xa, f0, env_o)
            xc = torch.cat([xl, xp[:, PA.A:PA.A + SEG]], 1)
            p = P.ppg_form(c1n(mfc(xc))[..., -T:], "soft")
            c2 = P.cvin_frames(xc, T)
            tab = torch.einsum("bdk,bkt->bdt", tabT[si], p) / 10
        return (p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], tab, c2), ct_o[:, 1:25], ct_o[:, 0] > 0.05

    def l1(y: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        mm = m[:, WARM:].float()[:, None]
        return ((y[..., WARM:] - tgt[..., WARM:]).abs() * mm).sum() / (mm.sum() * 24 + 1e-9)

    xs, tgt, m = batch()
    lk = causal_check(net, xs)
    with torch.no_grad():
        w0 = net.out.weight.clone()
        net.out.weight.normal_(0, 0.01)
        lk2 = causal_check(net, xs)
        net.out.weight.copy_(w0)
    print("未来不変性(実音声の特徴・t = 300 以降を乱数で書き換え・t 以前の出力の最大の差): 0 始め", lk, "乱数の重み", lk2, flush=True)
    if lk2 != 0.0:
        print("未来不変性 FAIL: 起動しない", flush=True)
        return 1
    with torch.no_grad():
        base_tab = float(l1(xs[4], tgt, m))
        spk_mean = (tgt[..., WARM:] * m[:, None, WARM:]).sum(-1, keepdim=True) / (m[:, None, WARM:].sum(-1, keepdim=True) + 1e-9)
        base_mean = float(l1(spk_mean.expand_as(tgt), tgt, m))
        base_zero = float(l1(torch.zeros_like(tgt), tgt, m))
    print("損失の床の照合(step 0): 表の引き(網の出力 0)", round(base_tab, 4), "区間の平均の行", round(base_mean, 4), "0", round(base_zero, 4),
          "整列: 有声のフレームの事後の和", round(float(xs[0].sum(1).mean()), 4), "c0n の中央値(発声)", round(float(xs[1][:, 0][m].median()), 3),
          "C2 の標準偏差", round(float(xs[5].std()), 3), "目標の行の |c1| 平均", round(float(tgt[:, 0][m].abs().mean()), 3), flush=True)
    log = open(out / "train.jsonl", "a")
    t0, acc, hist = time.time(), [], []
    total = 30 if a.smoke else a.steps
    for step in range(1, total + 1):
        xs, tgt, m = batch() if step > 1 else (xs, tgt, m)
        c2 = xs[5] + a.cv_noise * torch.randn_like(xs[5]) if a.cv_noise > 0 else xs[5]
        y = net(*xs[:5], c2)
        loss = l1(y, tgt, m)
        with torch.no_grad():
            lt = float(l1(xs[4], tgt, m))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gn):
            print("勾配が有限でない: この step を飛ばす", step, flush=True)
            continue
        opt.step()
        acc.append((float(loss), lt))
        if step % 100 == 0 or a.smoke:
            la, lb = np.mean([x[0] for x in acc]), np.mean([x[1] for x in acc])
            rec = {"step": step, "min": round((time.time() - t0) / 60, 1), "l1": round(float(la), 4), "l1_tab": round(float(lb), 4), "ratio": round(float(la / lb), 4)}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            acc = []
            if step == a.trip_step and rec["ratio"] > a.trip_ratio:
                print(f"打ち切り: step {step} の L1 / 表の引き {rec['ratio']} > {a.trip_ratio}", flush=True)
                torch.save({"net": net.state_dict(), "step": step, "aborted": True}, out / "last.pt")
                return 2
        if (time.time() - t0) / 60 > a.max_min:
            print("時間の上限で止める", step, flush=True)
            break
    torch.save({"net": net.state_dict(), "step": step, "cfg": {"ch": 192, "dils": [1, 2, 4, 8, 16, 32, 64], "d_c2": D_C2}, "hist": hist}, out / "last.pt")
    print("saved", out / "last.pt", "step", step, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

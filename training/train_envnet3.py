"""構音の包絡の内容の網 + 入力の声質への不変性の正則化(D647 の次・diag_envnet3)。同じ実女声の区間の素の入力(C)と摂動した入力(P = 録音の経路: 2 kHz の高域の棚 U(−S, +S) dB・
確率 0.5 で低域通過 5〜14 kHz、+ 確率 --sexpert で性別の摂動)から事後・C2・c0n を作り、損失 = L1(網(P), 目標) + L1(網(C), 目標) + w · L1(網(P), 網(C))(発声のフレーム)。
目標は本人の実の包絡の行のまま。規則は results/<tag>/prereg.yaml。
    uv run python train_envnet3.py --tag diag_envnet3 --init ../results/diag_envnet2/last.pt [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

import nvoc as N
import pae as PA
import train_ppgvoc as P
import train_rvoc as TR
from train_envnet import D_C2, ROOT, WARM, EnvNet, causal_check


class PairDS(torch.utils.data.IterableDataset):
    def __init__(self, rows: list, seed: int, spk_idx: dict, ctx: int, shelf: float, sexpert: float):
        self.rows, self.seed, self.spk_idx, self.ctx, self.shelf, self.sexpert = rows, seed, spk_idx, ctx, shelf, sexpert

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1009 + (wi.id if wi else 0))
        SEG, T = TR.SEG, TR.T
        while True:
            wav, f0p, sr, dur, spk = self.rows[rng.randrange(len(self.rows))][:5]
            try:
                n48 = int(dur * N.SR)
                if n48 < SEG + 4 * PA.A:
                    continue
                s = rng.randrange(PA.A + N.HOP, n48 - SEG - PA.A - N.HOP) // N.HOP * N.HOP
                f0 = np.load(f0p, mmap_mode="r")[s // N.HOP:s // N.HOP + T].astype(np.float32)
                f0 = np.pad(f0, (0, T - len(f0)))
                xa = TR.read_span(wav, sr, s - PA.A, SEG + 2 * PA.A)
                if len(xa) < SEG + 2 * PA.A or np.sqrt((xa[PA.A:PA.A + SEG] ** 2).mean()) < 1e-3:
                    continue
                g = min(2.0 ** rng.uniform(-0.5, 0.5), 0.99 / max(float(np.abs(xa).max()), 1e-6))
                xa = (xa * g).astype(np.float32)
                a0 = max(0, s - self.ctx)
                xl = TR.read_span(wav, sr, a0, s - a0) if s > a0 else np.zeros(0, np.float32)
                xl = (np.concatenate([np.zeros(self.ctx - len(xl), np.float32), xl]) * g).astype(np.float32)
                prm = (rng.uniform(5000.0, 14000.0) if rng.random() < 0.5 else 0.0, rng.uniform(-self.shelf, self.shelf))
                xp, xlp = TR.channel_apply(xa, prm), TR.channel_apply(xl, prm)
                kf = 1.0
                if rng.random() < self.sexpert:
                    sp = TR.sex_params(rng)
                    kf = sp[0]
                    z = TR.sex_apply(np.concatenate([xlp, xp]), sp)
                    xlp, xp = z[:len(xlp)], z[len(xlp):]
                e_o = PA.envelope(xa, f0)
                e_p = PA.envelope(xp, f0 * kf)
                yield (torch.from_numpy(xa), torch.from_numpy(f0), torch.from_numpy(e_p), torch.from_numpy(xlp.astype(np.float32)), torch.tensor(self.spk_idx[spk]),
                       torch.from_numpy(xp.astype(np.float32)), torch.from_numpy(e_o), torch.from_numpy(xl))
            except Exception:
                continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="diag_envnet3")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seg_mult", type=int, default=4)
    ap.add_argument("--sexpert", type=float, default=0.5)
    ap.add_argument("--shelf", type=float, default=15.0)
    ap.add_argument("--inv_w", type=float, default=1.0)
    ap.add_argument("--cv_noise", type=float, default=0.5)
    ap.add_argument("--init", default="")
    ap.add_argument("--seed", type=int, default=647)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max_min", type=float, default=55.0)
    ap.add_argument("--trip_step", type=int, default=1000)
    ap.add_argument("--trip_ratio", type=float, default=0.95)
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
    tabT = torch.from_numpy(np.stack([np.load(ROOT / "data/rvoc_tables_real" / f"{s}.npz")["T"] for s in spks])).float().to(dev)
    import c1_content as CC
    import train_c1 as T1
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"]); c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    P.cvin_setup(str(ROOT / "results/c2_1/last.pt"), dev)
    sm = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    front = TR.Front("pae", TR.ckpt_env_smooth(sm, None)).to(dev)
    net = EnvNet().to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["net"])
        print("初期値", a.init, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    loader = iter(torch.utils.data.DataLoader(PairDS(rows, a.seed, spk_idx, T1.PRE, a.shelf, a.sexpert), batch_size=a.bs, num_workers=a.workers,
                                              pin_memory=True, persistent_workers=True, prefetch_factor=4))

    def feats(x: torch.Tensor, xl: torch.Tensor, f0: torch.Tensor, env: torch.Tensor, si: torch.Tensor) -> tuple:
        ct = front(x, f0, env)
        xc = torch.cat([xl, x[:, PA.A:PA.A + SEG]], 1)
        p = P.ppg_form(c1n(mfc(xc))[..., -T:], "soft")
        c2 = P.cvin_frames(xc, T)
        tab = torch.einsum("bdk,bkt->bdt", tabT[si], p) / 10
        return (p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], tab, c2), ct

    def batch() -> tuple:
        xa, f0, e_p, xlp, si, xp, e_o, xl = [t.to(dev, non_blocking=True) for t in next(loader)]
        with torch.no_grad():
            fp, _ = feats(xp, xlp, f0, e_p, si)
            fc, ct_o = feats(xa, xl, f0, e_o, si)
        return fp, fc, ct_o[:, 1:25], ct_o[:, 0] > 0.05

    def l1(y: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        mm = m[:, WARM:].float()[:, None]
        return ((y[..., WARM:] - tgt[..., WARM:]).abs() * mm).sum() / (mm.sum() * 24 + 1e-9)

    def run(f: tuple) -> torch.Tensor:
        c2 = f[5] + a.cv_noise * torch.randn_like(f[5]) if a.cv_noise > 0 else f[5]
        return net(*f[:5], c2)

    fp, fc, tgt, m = batch()
    with torch.no_grad():
        w0 = net.out.weight.clone()
        net.out.weight.normal_(0, 0.01)
        lk = causal_check(net, fp)
        net.out.weight.copy_(w0)
        y_p, y_c = net(*fp), net(*fc)
        print("未来不変性(乱数の重み)", lk, "step 0: 表の引き(P)", round(float(l1(fp[4], tgt, m)), 4), "網(P)", round(float(l1(y_p, tgt, m)), 4), "網(C)", round(float(l1(y_c, tgt, m)), 4),
              "不変の差 |網(P) − 網(C)|", round(float(l1(y_p, y_c, m)), 4), "C2 の素と摂動の差の RMS", round(float((fp[5] - fc[5]).pow(2).mean().sqrt()), 4), flush=True)
    if lk != 0.0:
        print("未来不変性 FAIL: 起動しない", flush=True)
        return 1
    log = open(out / "train.jsonl", "a")
    t0, acc, hist = time.time(), [], []
    total = 30 if a.smoke else a.steps
    step = 0
    for step in range(1, total + 1):
        if step > 1:
            fp, fc, tgt, m = batch()
        y_p, y_c = run(fp), run(fc)
        lp, lc, li = l1(y_p, tgt, m), l1(y_c, tgt, m), l1(y_p, y_c, m)
        loss = lp + lc + a.inv_w * li
        with torch.no_grad():
            lt = float(l1(fp[4], tgt, m))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gn):
            print("勾配が有限でない: この step を飛ばす", step, flush=True)
            continue
        opt.step()
        acc.append((float(lp), float(lc), float(li), lt))
        if step % 100 == 0 or a.smoke:
            mp_, mc_, mi_, mt_ = (float(np.mean([x[i] for x in acc])) for i in range(4))
            rec = {"step": step, "min": round((time.time() - t0) / 60, 1), "l1_p": round(mp_, 4), "l1_c": round(mc_, 4), "inv": round(mi_, 4), "l1_tab": round(mt_, 4), "ratio_p": round(mp_ / mt_, 4)}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            acc = []
            if step == a.trip_step and rec["ratio_p"] > a.trip_ratio:
                print(f"打ち切り: step {step} の L1(P) / 表の引き {rec['ratio_p']} > {a.trip_ratio}", flush=True)
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

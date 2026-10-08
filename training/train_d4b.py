"""D4b: カテゴリカルWaveNet vocoder(μ-law 256bin・純CE尤度=定理Dの処方)。

d4a(決定論的L1)は定理D予測どおりゼロ文脈ブートストラップで崩壊(100Hz固定+DC)。
本腕はサンプル毎カテゴリカル分布を出力し、広い条件付きでも平均化しない。
推論=softmaxサンプリング(温度0.9)。ここでも条件はmel80+lf0+en+spk(100fps)。

    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
      uv run python train_d4b.py --tag d4b_wavenet --steps 40000
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from d4_vocoder import D4Voc, causal_mel48, HOP, excitation_from_cond
from train_d4a import get_utt_cond, held_paths, F0FIX, FEAT, CROP_F, CROP_N
from train_s1_3 import build_index, load_wav

ROOT = Path(__file__).resolve().parent.parent
N_BINS = 256
MU = 255.0


def mulaw_encode(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(MU * x.abs()) / torch.log1p(torch.tensor(MU))


def pv_bin_shift(x: torch.Tensor, r: float, n_fft: int = 2048, hop: int = 512) -> torch.Tensor:
    """STFTビンr倍写像のf0シフト入力摂動(d4d)。テンポ不変・pitchは正確に×r。
    formantも動く(bin写像のため)。金属的な粗さは入力側decoyとして許容。"""
    w = torch.hann_window(n_fft)
    S = torch.stft(x, n_fft, hop, n_fft, w, return_complex=True)
    Fd = S.shape[0]
    idx = (torch.arange(Fd) * r).long().clamp(max=Fd - 1)
    S2 = torch.zeros_like(S)
    S2[idx] = S
    return torch.istft(S2, n_fft, hop, n_fft, w, length=x.shape[-1])


def mulaw_decode(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * ((1 + MU) ** y.abs() - 1) / MU


class D4Cat(D4Voc):
    def __init__(self, excitation: bool = False):
        super().__init__(excitation=excitation)
        self.head = torch.nn.Conv1d(48, N_BINS, 1)

    def logits(self, wav: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        n = (min(wav.shape[-1], cond.shape[-1] * HOP)) // HOP * HOP
        if self.excitation:
            if wav.dim() == 3 and wav.shape[1] == 3:
                x = wav[:, :, :n]
            else:
                x = torch.cat([wav[:, None, :n],
                               excitation_from_cond(cond, n)], 1)
        else:
            x = wav[:, None, :n]
        cu = self._cond(cond, n)
        h = self.inp(F.pad(x, (6, 0)))
        for b in self.blocks:
            h = b(h, cu)
        return self.head(h).squeeze(1)          # [B, N_BINS, n]


def sample_from(m, cond, n, dev, temp=0.9):
    g = torch.Generator(device=dev).manual_seed(3)
    gen = torch.zeros(1, n, device=dev)
    rf = 3540
    hist = torch.zeros(1, rf, device=dev)
    exc = excitation_from_cond(cond, n) if m.excitation else None
    pos = 0
    while pos < n:
        nb = min(HOP, n - pos)
        ctx = torch.cat([hist, torch.zeros(1, nb, device=dev)], -1)
        if exc is not None:
            e_past = exc[0, :, max(0, pos - rf):pos]
            if pos < rf:
                e_past = torch.nn.functional.pad(e_past, (rf - pos, 0))
            x_in = torch.cat([ctx[:, None, :],
                              torch.cat([e_past, exc[0, :, pos:pos + nb]], -1)[None]], 1)
        else:
            x_in = ctx
        with torch.no_grad():
            lg = m.logits(x_in, cond[:, :, : (pos + nb) // HOP + 1])[:, :, -nb:]
        pr = F.softmax(lg / temp, 1)
        bins = torch.multinomial(pr.transpose(1, 2).reshape(-1, N_BINS), 1)
        lv = bins.float() / (N_BINS - 1) * 2 - 1
        y = mulaw_decode(lv).reshape(1, nb)
        gen[:, pos:pos + nb] = y
        new = gen[:, max(0, pos + nb - rf):pos + nb]
        hist = torch.cat([torch.zeros(1, rf - new.shape[-1], device=dev), new], -1) \
            if new.shape[-1] < rf else new
        pos += nb
    return gen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roll-p", type=float, default=0.0,
                    help="roll-decoyレジーム率: 入力のみ同一crop内巡回シフト(target/cond不変)")
    ap.add_argument("--zero-p", type=float, default=0.0,
                    help="ゼロ履歴レジーム率: 入力をゼロに(target/cond不変)")
    ap.add_argument("--shift-p", type=float, default=0.0,
                    help="f0シフト入力レジーム率(d4d): 入力のみr倍binシフト(target/cond不変)")
    ap.add_argument("--shift-st", type=float, default=10.0,
                    help="f0シフト幅の最大絶対値(半音・符号は一様抽選)")
    ap.add_argument("--excite", action="store_true",
                    help="励起チャンネル入力(lf0位相積算sin/cos・d4c)")
    ap.add_argument("--resume", action="store_true",
                    help="OOM等で落ちたら last.pt の重みとstepから再開(状態は軽視)")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)

    tr, held = build_index()
    tr = [p for p in tr
          if (F0FIX / p.parent.name / (p.stem + ".pt")).exists()
          and (FEAT / p.parent.name / (p.stem + ".pt")).exists()]
    rng.shuffle(tr)
    print(f"  train {len(tr)}", flush=True)

    m = D4Cat(excitation=a.excite).to(dev)
    params = [q for q in m.parameters() if q.requires_grad]
    print(f"  params {sum(q.numel() for q in params)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                              pct_start=0.03)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict = {}

    def get(p: Path):
        if p not in cache:
            if len(cache) > 300:
                cache.pop(next(iter(cache)))
            try:
                import train_d4a as T
                cache[p] = T.get_utt_cond(p)
            except Exception:  # noqa: BLE001
                cache[p] = None
        return cache[p]

    best = 1e9
    reg_ce = {0: [], 1: [], 2: []}
    t0 = time.time()
    step = 0
    if a.resume:
        ck = out_dir / f"{a.tag}_last.pt"
        if ck.exists():
            st = torch.load(ck, map_location=dev, weights_only=False)
            m.load_state_dict(st["net"])
            step = st["step"]
            for _ in range(step):
                sch.step()
            print(f"  resumed at step {step} (opt state reset)", flush=True)
    while step < a.steps:
        ws, ts, cs, rs = [], [], [], []
        while len(ws) < a.batch:
            p = rng.choice(tr)
            it = get(p)
            if it is None:
                continue
            wv, cond = it
            T = min(wv.shape[0] // HOP, cond.shape[1])
            if T <= CROP_F + 4:
                continue
            s0 = rng.randrange(0, T - CROP_F)
            wt = wv[s0 * HOP:(s0 + CROP_F) * HOP]
            wi, rg = wt, 0
            u = rng.random()
            if u < a.shift_p:
                st = rng.uniform(-a.shift_st, a.shift_st)
                wi, rg = pv_bin_shift(wt, 2.0 ** (st / 12.0)), 1
            elif u < a.shift_p + a.zero_p:
                wi, rg = torch.zeros_like(wt), 2
            ws.append(wi)
            ts.append(wt)
            cs.append(cond[:, s0:s0 + CROP_F])
            rs.append(rg)
        step += 1
        xb = torch.stack(ws).to(dev)
        tb = torch.stack(ts).to(dev)
        cb = torch.stack(cs).to(dev)
        lg = m.logits(xb, cb)
        tgt = ((mulaw_encode(tb) + 1) / 2 * (N_BINS - 1)).round().long() \
            .clamp(0, N_BINS - 1)
        ce_all = F.cross_entropy(lg.transpose(1, 2).reshape(-1, N_BINS),
                                 tgt.reshape(-1), reduction="none") \
            .reshape(a.batch, -1).mean(1)
        lge = ce_all.detach()
        for i, rg in enumerate(rs):
            reg_ce[rg].append(float(lge[i]))
        loss = ce_all.mean()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            from eval_d4a_freerun import metrics
            rmean = {k: (sum(v[-400:]) / len(v[-400:]) if v[-400:] else None)
                     for k, v in reg_ce.items()}
            p = held[0]
            it = get(p)
            n = min(it[0].shape[0], 96000)
            y = sample_from(m, it[1][None, :, : n // HOP].to(dev), n, dev)
            y = y[0].cpu().numpy()
            soundfile.write(out_dir / "freerun.wav", np.clip(y, -1, 1), 48000)
            mm = metrics(y)
            print(f"  step {step:6d}  CE {float(loss.detach()):.4f}  "
                  f"reg {['%.3f' % rmean[k] if rmean[k] is not None else '-' for k in (0, 1, 2)]}"
                  f"  fr {mm}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"net": m.state_dict(), "step": step, "cli": vars(a)},
                       out_dir / f"{a.tag}_last.pt")
            score = float(loss.detach()) - (1.0 if mm["hi_mid"] > 0.004 else 0.0) \
                    - (1.0 if abs(mm["f0_median"] - 100.0) > 3.0 else 0.0)
            if score < best:
                best = score
                torch.save({"net": m.state_dict(), "step": step, "cli": vars(a)},
                           out_dir / f"{a.tag}_best.pt")
    print(f"\n{a.tag}: best score {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

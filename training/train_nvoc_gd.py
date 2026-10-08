"""nvoc の段 G 診断: 判別器を新規初期化し 48kHz のまま(再標本化なし)入れる。HiFi-GAN の標準に近い形。

  判別器 = MPD(周期 2,3,5,7,11・新規)+ MRD-log(対数振幅 STFT 3 解像度・新規)。生成器は --init_g の EMA から。
  最初の --d_warm step は判別器だけを学習(生成器は再構成損失 w_mel·logmel + 2·mrstft)、以後 w_mel·logmel + LSGAN + w_fm·FM。
  学習済み BigVGAN 判別器(44.1kHz へ再標本化)を使った段 G(nvoc2〜4・diag_nvoc_g*)が毎回悪化したことの帰属用。

    CUDA_VISIBLE_DEVICES=0 uv run python train_nvoc_gd.py --tag diag_nvoc_gfresh --init_g ../results/nvoc2/snap/ema_40k.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
from train_nvoc import DS, SEG, W0, MRDLog, check_alignment
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init_g", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--d_warm", type=int, default=1000, help="この step までは生成器を再構成損失で学習(判別器は r_only 以降に学習)")
    ap.add_argument("--r_only", type=int, default=0, help="この step までは判別器を学習しない(純粋な段 R)")
    ap.add_argument("--snap_at", type=int, nargs="*", default=[])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dcrop", type=int, default=16384)
    ap.add_argument("--lr_g", type=float, default=2e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--w_mel", type=float, default=45.0)
    ap.add_argument("--w_mel_r", type=float, default=45.0, help="再構成段(d_warm まで)の logmel 重み")
    ap.add_argument("--w_fm", type=float, default=2.0)
    ap.add_argument("--nosrc", action="store_true")
    ap.add_argument("--lr_end", type=float, default=0.0, help="生成器の学習率を指数減衰でこの値まで下げる(0 で一定)")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    check_alignment()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    from train_ddsp_vc import index
    import eval_nvoc as E
    from bigvgan.discriminators import MultiPeriodDiscriminator
    spk, tr, ev = index()
    loader = iter(torch.utils.data.DataLoader(DS(spk, tr, 7), batch_size=a.bs, num_workers=a.workers, pin_memory=True,
                                              persistent_workers=True, prefetch_factor=4))
    model = N.NVoc().to(dev)
    model.load_state_dict(torch.load(a.init_g, map_location=dev, weights_only=False)["ema"])
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    evm = N.NVoc().to(dev)
    mpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False,
                                                   discriminator_channel_mult=1)).to(dev)
    mrd = MRDLog().to(dev)
    gparams = list(model.parameters())
    dparams = list(mpd.parameters()) + list(mrd.parameters())
    opt = torch.optim.AdamW(gparams, lr=a.lr_g, betas=(0.8, 0.99))
    dopt = torch.optim.AdamW(dparams, lr=a.lr_d, betas=(0.8, 0.99))
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, (a.lr_end / a.lr_g) ** (1.0 / a.steps)) if a.lr_end > 0 else None
    items = E.held_items()
    if a.nosrc:
        items = [{**it, "f0": np.zeros_like(it["f0"])} for it in items]
    log = open(out / "train.jsonl", "a")

    def evaluate(step: int) -> None:
        evm.load_state_dict(ema)
        r = {"step": step, "held": E.held_eval(evm, dev, items)}
        model.train()
        print("held", json.dumps(r), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()

    def lsd(rs, gs):
        return sum(((p.float() - 1) ** 2).mean() + (q.float() ** 2).mean() for p, q in zip(rs, gs))

    def feat(fr, fg):
        return sum(F.l1_loss(x.detach().float(), y.float()) for A, B in zip(fr, fg) for x, y in zip(A, B))

    evaluate(0)
    acc: dict = {k: [] for k in ("lm", "adv", "fm", "dl", "dr", "df", "mr", "mf")}
    t0 = time.time()
    step = 0
    model.train()
    while step < a.steps:
        step += 1
        xin, tgt, f0 = (t.to(dev, non_blocking=True) for t in next(loader))
        with torch.no_grad():
            mel = model.mel_ctx(xin)
            exc = torch.stack([N.harmonic_source(f0 * (0.0 if a.nosrc else 1.0)), torch.randn(xin.shape[0], SEG, device=dev)], 1)
        y = model.generate(mel, exc)
        yl, tl = y[:, W0:], tgt[:, W0:]
        with torch.no_grad():
            g = (0.95 / tl.abs().amax(-1, keepdim=True).clamp(min=1e-3)).clamp(max=20.0)
            off = random.randrange(0, tl.shape[-1] - a.dcrop)
            r = (tl * g)[:, off:off + a.dcrop]
        f = (yl * g)[:, off:off + a.dcrop]
        if step > a.r_only:
            dopt.zero_grad(set_to_none=True)
            pr, pg, _, _ = mpd(r[:, None], f.detach()[:, None])
            mr, mg, _, _ = mrd(r, f.detach())
            dl = lsd(pr, pg) + lsd(mr, mg)
            if torch.isfinite(dl):
                dl.backward()
                if torch.isfinite(torch.nn.utils.clip_grad_norm_(dparams, 500.0)):
                    dopt.step()
            acc["dl"].append(float(dl))
            acc["dr"].append(float(sum(p.float().mean() for p in pr) / len(pr)))
            acc["df"].append(float(sum(q.float().mean() for q in pg) / len(pg)))
            acc["mr"].append(float(sum(p.float().mean() for p in mr) / len(mr)))
            acc["mf"].append(float(sum(q.float().mean() for q in mg) / len(mg)))
        lm = logmel_l1(yl[:, None], tl[:, None], mels)
        if step <= a.d_warm:
            loss = a.w_mel_r * lm + 2 * mrstft(yl[:, None], tl[:, None])
            adv = fm = torch.zeros((), device=dev)
        else:
            for p in dparams:
                p.requires_grad_(False)
            _, pg, fr, fg = mpd(r[:, None], f[:, None])
            _, mg, mfr, mfg = mrd(r, f)
            adv = sum(((q.float() - 1) ** 2).mean() for q in pg + mg)
            fm = feat(fr, fg) + feat(mfr, mfg)
            loss = a.w_mel * lm + adv + a.w_fm * fm
            for p in dparams:
                p.requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print("non-finite loss at", step, flush=True)
            continue
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(gparams, 500.0)):
            print("non-finite grad at", step, flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        if sch is not None:
            sch.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        acc["lm"].append(float(lm))
        acc["adv"].append(float(adv))
        acc["fm"].append(float(fm))
        if step % 100 == 0:
            rr = {"step": step, "phase": "R" if step <= a.r_only else ("Dwarm" if step <= a.d_warm else "G"), "min": round((time.time() - t0) / 60, 1)}
            rr.update({k: round(float(np.mean(v)), 4) for k, v in acc.items() if v})
            acc = {k: [] for k in acc}
            print(json.dumps(rr), flush=True)
            log.write(json.dumps(rr) + "\n")
            log.flush()
        if step % a.eval_every == 0:
            evaluate(step)
            torch.save({"model": model.state_dict(), "ema": ema, "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                        "opt": opt.state_dict(), "dopt": dopt.state_dict(), "step": step, "cfg": model.cfg}, out / "last.tmp")
            (out / "last.tmp").replace(out / "last.pt")
        if step in a.snap_at:
            (out / "snap").mkdir(exist_ok=True)
            torch.save({"ema": ema, "step": step, "cfg": model.cfg}, out / "snap" / f"ema_{step // 1000}k.pt")
    print("done", step, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

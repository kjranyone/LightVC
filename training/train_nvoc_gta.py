"""nvoc を ZS-VC の予測 mel で微調整する(GTA)。正解は実音声。

  女声の区間 x → 因果 mel → 凍結した ZS-VC(自己再構成: 同じ話者の参照・本人の f0)→ 予測 mel → nvoc(学習)→ y ≈ x[m − DELAY]
  損失 = 15·多尺度 logmel + 2·mrstft(nvoc の段 R と同じ)。--real_frac の割合は本物の mel を入れて写し合成の品質も保つ。

    CUDA_VISIBLE_DEVICES=0 uv run python train_nvoc_gta.py --tag nvoc_gta1 --zsvc ../results/zsvc3/last.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import zsvc as Z
import train_zsvc as TZ
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft

ROOT = Path(__file__).resolve().parent.parent
W0 = 12000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--zsvc", required=True)
    ap.add_argument("--vocoder", default=str(ROOT / "results/nvoc5r2/last.pt"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--real_frac", type=float, default=0.25)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    from train_ddsp_vc import index, ecapa_model
    import eval_zsvc as EZ
    spk, tr, ev = index()
    _, males_held = TZ.male_index()
    fl = iter(torch.utils.data.DataLoader(TZ.FemaleDS(spk, tr, 11), batch_size=a.bs, num_workers=a.workers, pin_memory=True,
                                          persistent_workers=True, prefetch_factor=4))
    zs = Z.ZSVC(cv=True).to(dev)
    zs.load_state_dict(torch.load(a.zsvc, map_location=dev, weights_only=False)["ema"])
    zs.eval()
    zs.requires_grad_(False)
    voc = N.NVoc().to(dev)
    voc.load_state_dict(torch.load(a.vocoder, map_location=dev, weights_only=False)["ema"])
    ema = {k: v.detach().clone() for k, v in voc.state_dict().items()}
    evv = N.NVoc().to(dev)
    opt = torch.optim.AdamW(voc.parameters(), lr=a.lr, betas=(0.8, 0.99))
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    emb = ecapa_model(dev)
    evset = EZ.build_evalset(spk, ev, males_held)
    log = open(out / "train.jsonl", "a")

    def evaluate(step: int) -> None:
        evv.load_state_dict(ema)
        evv.eval()
        r = {"step": step, **EZ.evaluate(zs, evv, emb, evset, dev)}
        print("eval", json.dumps(r), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()

    evaluate(0)
    t0 = time.time()
    acc: dict = {}
    for step in range(1, a.steps + 1):
        xin, f0, _cv, _hc, ref, nref, _ts = (t.to(dev, non_blocking=True) if torch.is_tensor(t) else t for t in next(fl))
        B = xin.shape[0]
        with torch.no_grad():
            mel = N.NVoc.mel_ctx(zs, xin)
            rmel = N.NVoc.mel_ctx(zs, torch.cat([torch.zeros(B, TZ.CTX, device=dev), ref], -1))
            rmask = (torch.arange(rmel.shape[-1], device=dev)[None] < (nref.to(dev) // N.HOP)[:, None]).float()
            s = zs.spk(rmel, rmask)
            pred = zs(mel, zs.level(mel), f0, s)
            k = int(round(a.real_frac * B))
            cond = torch.cat([mel[:k], pred[k:]], 0)
            exc = torch.stack([torch.zeros(B, TZ.SEG, device=dev), torch.randn(B, TZ.SEG, device=dev)], 1)
            tgt = xin[:, TZ.CTX - N.DELAY:TZ.CTX + TZ.SEG - N.DELAY]
        y = voc.generate(cond, exc)
        yl, tl = y[:, W0:], tgt[:, W0:]
        lm = logmel_l1(yl[:, None], tl[:, None], mels)
        loss = 15 * lm + 2 * mrstft(yl[:, None], tl[:, None])
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(voc.parameters(), 500.0)):
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        with torch.no_grad():
            for kk, v in voc.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[kk].mul_(0.999).add_(v.detach(), alpha=0.001)
        acc.setdefault("lm", []).append(float(lm))
        if step % 100 == 0:
            r = {"step": step, "min": round((time.time() - t0) / 60, 1), "lm": round(float(np.mean(acc["lm"])), 4)}
            acc = {}
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
        if step % a.eval_every == 0:
            evaluate(step)
            torch.save({"ema": ema, "model": voc.state_dict(), "step": step, "cfg": voc.cfg}, out / "last.pt")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

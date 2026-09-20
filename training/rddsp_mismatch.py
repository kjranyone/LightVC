"""The fourth point of 6.4, applied to arm C: is the mel conditioning used at all?

6.4 requires four things of any claim that a learned component contributed, and
one of them is a control in which the conditioning carries no information about
the target. 5.3 is why: a net trained on mel from an UNRELATED utterance reached
4.0988 against the 4.174 of the real thing -- 98% of the "gain" survived the
conditioning being destroyed, because the gain was memorisation, not mapping.

Arm C has never had this control. Its prior already scores 2.65-2.97 on its own,
so the null hypothesis is live and specific:

    the net is a fixed post-filter on the prior, and the mel input does nothing.

The cheap decisive form is at evaluation, not training: take the trained net and
feed it the mel of a DIFFERENT utterance while leaving the prior alone. If the
gain survives, the net is not using mel. If it collapses, mel is load-bearing.
That costs one forward pass per utterance instead of a second training run.

Reported alongside: the prior itself and the intact net, on the same utterances
with the same noise seed, so all three columns differ only in the mel channel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from rddsp_loop import score_one
from rddsp_hf import Wavehax2D, NBIN
from rddsp_gpu import build, to_dev, prior_wave, feats, istft, stft, CACHE_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = Path(a.ckpt)
    ck = torch.load(p if p.exists() else CACHE_DIR / a.ckpt, map_location=dev,
                    weights_only=False)
    cfg = ck["args"]
    print(f"  ckpt {a.ckpt}  step {ck['step']}  ch {cfg['ch']}  "
          f"pnoise {cfg.get('pnoise')}  lmos {cfg['lmos']}", flush=True)

    _, te_raw = build(cfg["ntrain"], cfg["ntest"])
    test = to_dev(te_raw, dev)
    net = Wavehax2D(cin=4, ch=cfg["ch"], layers=cfg["layers"]).to(dev)
    net.load_state_dict(ck["net"])
    net.eval()

    pn = cfg.get("pnoise", 0.05)
    rows = {"prior": [], "intact": [], "mismatch": []}
    ge = torch.Generator(device=dev).manual_seed(999)
    with torch.no_grad():
        for i, it in enumerate(test):
            n = it["gt"].shape[-1]
            pw = prior_wave(it, ge, dev, pn)
            f, P = feats(it, pw)
            # The mel of ANOTHER speaker's utterance, cropped/padded to this
            # one's length. Everything else -- prior, its spectrum, the noise
            # draw, the weights -- is identical.
            other = test[(i + 1) % len(test)]
            om = other["m"]
            T = f.shape[-1]
            mm = (om[:, :, :T] if om.shape[-1] >= T else
                  torch.nn.functional.pad(om, (0, T - om.shape[-1]), mode="replicate"))
            fx = f.clone()
            fx[:1] = mm

            def run(feat):
                o = net(feat[None])[0]
                S = torch.complex(P.real + o[0], P.imag + o[1])
                return istft(S, n).cpu()

            rows["prior"].append(score_one(it["gt"].cpu(), pw[:n].cpu()))
            rows["intact"].append(score_one(it["gt"].cpu(), run(f)))
            rows["mismatch"].append(score_one(it["gt"].cpu(), run(fx)))
            print(f"  u{i:02d}  prior {rows['prior'][-1]:.3f}  "
                  f"intact {rows['intact'][-1]:.3f}  "
                  f"mismatch {rows['mismatch'][-1]:.3f}", flush=True)

    pr = np.array(rows["prior"])
    it_ = np.array(rows["intact"])
    mm_ = np.array(rows["mismatch"])
    def ci(v):
        return 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    print(f"\n  ---- {len(pr)} unseen speakers ----")
    print(f"  prior              {pr.mean():.4f}")
    print(f"  net, real mel      {it_.mean():.4f}   gain {(it_-pr).mean():+.4f} "
          f"+-{ci(it_-pr):.4f}")
    print(f"  net, WRONG mel     {mm_.mean():.4f}   gain {(mm_-pr).mean():+.4f} "
          f"+-{ci(mm_-pr):.4f}")
    surv = (mm_ - pr).mean() / max((it_ - pr).mean(), 1e-9)
    print(f"\n  fraction of the gain surviving destroyed conditioning: {100*surv:.1f}%")
    print(f"  (5.3's memorisation case survived at 98%. A mapping should not.)")


if __name__ == "__main__":
    main()

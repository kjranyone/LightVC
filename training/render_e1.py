"""E1 render — self-recon, held-out utts. arms: gt / ceiling / e1<mode>.
G(cond2, s, None) timbre-AdaIN only (no s_art). cond2 = [scrubbed?content, prosody]."""
from __future__ import annotations
import sys, argparse, random
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F, soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, DEV
from train_m2 import TimbreEncoder, ContentScrub, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig
from render_z1 import load_wav, build_cond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--out", default="../results/e1")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    ga = ck.get("args", {"dim": 384, "layers": 6})
    mode = ck.get("scrub_mode", "none")
    arm = "e1" + mode
    g = MelGen(cond_dim=770, dim=ga["dim"], n_layers=ga["layers"],
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM).to(DEV)
    g.load_state_dict(ck["g"]); g.eval()
    t = TimbreEncoder().to(DEV); t.load_state_dict(ck["t"]); t.eval()
    scrub = None
    if mode != "none" and "scrub" in ck:
        scrub = ContentScrub().to(DEV); scrub.load_state_dict(ck["scrub"]); scrub.eval()
    voc, voc_step = load_freebig(args.freebig)

    heldtxt = Path(args.ckpt).parent / "heldout.txt"
    held = [Path(x) for x in heldtxt.read_text().split("\n") if x.strip()] if heldtxt.exists() else []
    allf = sorted(Path(args.feat).rglob("*.pt"))
    by_spk = defaultdict(list)
    for f in allf:
        by_spk[f.parent.name].append(f)
    held = [f for f in held if f.exists()] or allf
    pick = held if len(held) <= args.n else random.sample(held, args.n)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"render E1 | ckpt step {ck.get('step')} | scrub {mode} (arm {arm}) | "
          f"freebig {voc_step} | {len(pick)} held-out utts -> {out}", flush=True)

    for i, f in enumerate(pick):
        d = torch.load(f, weights_only=False)
        cond, tmel = build_cond(d); cond = cond.unsqueeze(0).to(DEV)
        y = load_wav(d["path"])[: tmel * HOP]
        pool = [x for x in by_spk[f.parent.name] if x != f] or [f]
        rd = torch.load(random.choice(pool), weights_only=False)
        rw = load_wav(rd["path"]); rn = int(3 * SR)
        rw = rw[:rn] if rw.shape[0] >= rn else F.pad(rw, (0, rn - rw.shape[0]))
        with torch.no_grad():
            s = t(mel_of(rw.unsqueeze(0).to(DEV)))
            mel_gt = mel_of(y.unsqueeze(0).to(DEV))
            content_c = scrub(cond[:, :768]) if scrub is not None else cond[:, :768]
            cond2 = torch.cat([content_c, cond[:, 768:]], dim=1)
            m_hat = g(cond2, s, None)
            y_out = voc(m_hat)[0].cpu().numpy()
            y_ceil = voc(mel_gt)[0].cpu().numpy()
        stem = f"{f.parent.name}_{f.stem}"
        sf.write(out / f"{i:02d}_{stem}_gt.wav", np.clip(y.numpy(), -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{i:02d}_{stem}_ceiling.wav", np.clip(y_ceil, -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{i:02d}_{stem}_{arm}.wav", np.clip(y_out, -1, 1), SR, subtype="PCM_16")
        print(f"  [{i}] {stem} tmel {tmel}", flush=True)
    print(f"done -> {out} (arm {arm})", flush=True)


if __name__ == "__main__":
    main()

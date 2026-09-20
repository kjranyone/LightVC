"""Z0-C ceiling gate: does a causal-mel FreeC (freec_B) preserve real-mel
sharpness toward gt? For each held speaker: ceiling = freec_B(causal_mel(gt)),
compare harmonic-sharp to gt and to the freec_F(centered) baseline. Speaker-level
bootstrap. This is the 'ceiling 鋭さ維持' condition of Z0-C (§7.2), NOT product E2E.
"""
from __future__ import annotations
import sys, argparse, json, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, librosa
sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import SR, HOP, DEV
from train_z1 import load_freebig
from f0leak_probe import load_wav
from causal_mel import causal_mel, centered_mel
from stable_res_eval2 import metrics, sha256, speaker_id, bootstrap_speaker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vb", required=True, help="freec_B checkpoint (causal-mel trained)")
    ap.add_argument("--mel-nfft", type=int, required=True, help="mel analysis n_fft used to train --vb")
    ap.add_argument("--vf", default="checkpoints/freeC/foundation_lowlatency_5p8ms.pt",
                    help="freec_F baseline (centered-mel)")
    ap.add_argument("--held", default="../results/interpretable_vc/held_multispk.txt")
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    VB = load_freebig(args.vb)[0]
    VF = load_freebig(args.vf)[0]
    held = sorted(Path(x.strip()) for x in open(args.held) if x.strip())
    rows = []
    for f in held:
        if len(rows) >= args.n:
            break
        d = torch.load(f, weights_only=False)
        w = load_wav(d["path"])
        gt = torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).to(DEV)
        with torch.no_grad():
            mb = causal_mel(gt, n_fft=args.mel_nfft, hop=args.hop)
            yb = VB(mb).squeeze().cpu().numpy()
            mf = centered_mel(gt, n_fft=2048, hop=args.hop)
            yf = VF(mf).squeeze().cpu().numpy()
        rows.append({"path": str(d["path"]), "speaker": speaker_id(d["path"]),
                     "gt_sharp": metrics(np.asarray(w))[0],
                     "ceil_B_sharp": metrics(yb)[0],
                     "ceil_F_sharp": metrics(yf)[0]})

    by_gtB = defaultdict(list); by_gtF = defaultdict(list); by_BF = defaultdict(list)
    for r in rows:
        by_gtB[r["speaker"]].append(r["gt_sharp"] - r["ceil_B_sharp"])
        by_gtF[r["speaker"]].append(r["gt_sharp"] - r["ceil_F_sharp"])
        by_BF[r["speaker"]].append(r["ceil_B_sharp"] - r["ceil_F_sharp"])
    m_gtB, ci_gtB, nsp = bootstrap_speaker(by_gtB, 2000)
    m_gtF, ci_gtF, _ = bootstrap_speaker(by_gtF, 2000)
    m_BF, ci_BF, _ = bootstrap_speaker(by_BF, 2000)
    gt = np.array([r["gt_sharp"] for r in rows])
    cb = np.array([r["ceil_B_sharp"] for r in rows]); cf = np.array([r["ceil_F_sharp"] for r in rows])
    out = {
        "script": "z0c_ceiling_eval.py", "argv": sys.argv, "mel_nfft": args.mel_nfft,
        "vb_path": args.vb, "vb_sha256": sha256(args.vb),
        "vf_path": args.vf, "vf_sha256": sha256(args.vf),
        "n_utt": len(rows), "n_speakers": nsp,
        "gt_sharp_mean": float(gt.mean()),
        "ceil_B_sharp_mean": float(cb.mean()), "ceil_F_sharp_mean": float(cf.mean()),
        "gt_minus_ceilB_mean_spk": m_gtB, "gt_minus_ceilB_ci95_spk": ci_gtB,
        "gt_minus_ceilF_mean_spk": m_gtF, "gt_minus_ceilF_ci95_spk": ci_gtF,
        "ceilB_minus_ceilF_mean_spk": m_BF, "ceilB_minus_ceilF_ci95_spk": ci_BF,
        "guardrail": "harmonic-sharp proxy; Z0-C 最終判定は耳。ceilB が gt に近く(gt-ceilB↓) かつ ceilF 以上ならbetter",
        "per_utt": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[freec_B nfft{args.mel_nfft}] gt {gt.mean():.2f} | ceil_B {cb.mean():.2f} (gt-ceilB {m_gtB:+.2f} CI{[round(x,2) for x in ci_gtB]}) "
          f"| ceil_F {cf.mean():.2f} (gt-ceilF {m_gtF:+.2f}) | ceilB-ceilF {m_BF:+.2f} CI{[round(x,2) for x in ci_BF]}")
    print("  ->", args.out)


if __name__ == "__main__":
    main()

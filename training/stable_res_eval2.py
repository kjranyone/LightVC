"""De-confounded resolution eval (rev4: ABI-matched FreeC hop).

History:
  - stable_res_eval.py: ceiling/self used DIFFERENT vocoders (confound).
  - stable_res_eval2 rev2: same V for both arms, but compared freebig hop512
    foundation vs freeC hop128 snap, and fed hop512 mel_of / hop128 G mel
    across ABIs. Those JSONs must not claim 'co-training degraded FreeC' or
    'G mel ≈ real mel'.

rev4 protocol (required for attribution):
  V_init = freeC/foundation_lowlatency_5p8ms.pt  (hop128)
  V_snap = e6/snap_*.pt                          (hop128, same free_args)
  ceiling = V(real mel on V.hop grid)
  self    = V(G mel at FreeC rate; upsample cond by gck['upsample'] default 4)
  Stats: speaker-level bootstrap; TOST / equivalence margin (default 0.15).

Do NOT pass freebig hop512 as V when G was trained for hop128 FreeC.
"""
from __future__ import annotations
import sys, argparse, json, hashlib, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import librosa
sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import SR, HOP, DEV
from train_m2 import TimbreEncoder, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig
from f0leak_probe import load_wav, build_cond
from bigvgan.meldataset import get_mel_spectrogram
from bigvgan.env import AttrDict
from free_train_universal import SNAP


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metrics(y):
    S = np.abs(librosa.stft(y.astype(np.float64), n_fft=2048, hop_length=512)) + 1e-7
    f = np.linspace(0, SR / 2, S.shape[0])
    v = S.mean(0) > np.percentile(S.mean(0), 60)
    b = (f >= 500) & (f <= 3000)
    sub = np.log(S[b][:, v])
    sharp = float((np.percentile(sub, 90, 0) - np.percentile(sub, 10, 0)).mean())
    hf = (f >= 4000) & (f <= 12000)
    lf = (f >= 300) & (f <= 3000)
    L = np.log(S[:, v]).mean(1)
    return sharp, float(L[hf].mean() - L[lf].mean())


def speaker_id(path: str) -> str:
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p in ("female_real", "female_tts", "male", "wavs", "audio") and i + 1 < len(parts):
            return parts[i + 1]
    return Path(path).parent.name


def mel_for_vocoder(wav_1d: torch.Tensor, hop: int, n_mels: int = 128) -> torch.Tensor:
    """Real mel on the vocoder's frame grid (not train_m1 hop512-only mel_of)."""
    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    h["hop_size"] = hop
    h["num_mels"] = n_mels
    h["sampling_rate"] = SR
    w = wav_1d.detach().float().cpu()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return get_mel_spectrogram(w, h)


def bootstrap_speaker(values_by_spk: dict[str, list[float]], n_boot: int, seed: int = 0):
    """Resample speakers with replacement; within-speaker mean then grand mean."""
    spks = list(values_by_spk.keys())
    means = np.array([float(np.mean(values_by_spk[s])) for s in spks], dtype=np.float64)
    if len(spks) < 2:
        m = float(means.mean()) if len(means) else float("nan")
        return m, [m, m], len(spks)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(spks), size=(n_boot, len(spks)))
    boot = means[idx].mean(axis=1)
    ci = np.percentile(boot, [2.5, 97.5])
    return float(means.mean()), [float(ci[0]), float(ci[1])], len(spks)


def tost_equivalent(mean: float, ci: list[float], delta: float) -> dict:
    """Equivalence if CI entirely inside [-delta, +delta] (simple interval criterion)."""
    lo, hi = ci
    inside = (lo >= -delta) and (hi <= delta)
    return {
        "margin_delta": delta,
        "ci95": ci,
        "mean": mean,
        "equivalent_by_ci_in_margin": bool(inside),
        "note": "CI entirely in [-δ,+δ] only; not a formal TOST p-value. "
                "CI containing 0 is NOT equivalence.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", required=True)
    ap.add_argument("--v", required=True, help="SINGLE frozen vocoder for BOTH arms (prefer hop-matched freeC)")
    ap.add_argument("--held", default="/tmp/e5eval_held.txt")
    ap.add_argument("--n", type=int, default=24, help="max utterances (prefer many speakers)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--equiv-delta", type=float, default=0.15, help="equivalence margin for ceiling-self")
    ap.add_argument("--allow-abi-mismatch", action="store_true",
                    help="allow V.hop != G freeC rate (NOT for attribution claims)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gck = torch.load(args.g, map_location=DEV, weights_only=False)
    ga = gck["args"]
    g = MelGen(cond_dim=770, dim=ga["dim"], n_layers=ga["layers"], timbre_dim=TIMBRE_DIM,
               art_dim=ART_DIM, f0_fourier=ga.get("f0_fourier", 0)).to(DEV)
    g.load_state_dict(gck["g"])
    g.eval()
    t = TimbreEncoder().to(DEV)
    t.load_state_dict(gck["t"])
    t.eval()
    V, _ = load_freebig(args.v)
    v_hop = int(V.hop)
    v_nfft = int(V.nfft)
    v_causal = bool(V.causal)
    up = int(gck.get("upsample", 4))
    g_frame_hop = HOP // up

    abi_ok = (v_hop == g_frame_hop)
    if not abi_ok and not args.allow_abi_mismatch:
        raise SystemExit(
            f"ABI mismatch: V hop={v_hop} nfft={v_nfft} causal={v_causal} but G upsample={up} "
            f"implies frame hop={g_frame_hop}. Use freeC hop128 V for E5/E6 G, or pass "
            f"--allow-abi-mismatch (results must not claim G quality / V degradation)."
        )

    held_lines = [x.strip() for x in open(args.held) if x.strip()]
    held = sorted(Path(x) for x in held_lines)
    rows = []
    for f in held:
        if len(rows) >= args.n:
            break
        d = torch.load(f, weights_only=False)
        if "f0" not in d:
            continue
        cond, tmel = build_cond(d)
        cond = cond.unsqueeze(0).to(DEV)
        w = load_wav(d["path"])[: tmel * HOP]
        gt = torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).to(DEV)
        spk = speaker_id(d["path"])
        with torch.no_grad():
            mel_t = mel_for_vocoder(gt[0, : int(3 * SR)], hop=HOP).to(DEV)
            if mel_t.dim() == 2:
                mel_t = mel_t.unsqueeze(0)
            s = t(mel_t)
            cond4 = F.interpolate(cond, scale_factor=up, mode="linear", align_corners=False)
            mel_g = g(cond4, s, None)
            if not abi_ok:
                if v_hop > g_frame_hop:
                    r = max(1, v_hop // g_frame_hop)
                    mel_g = F.avg_pool1d(mel_g, r, stride=r)
                else:
                    r = max(1, g_frame_hop // v_hop)
                    mel_g = F.interpolate(mel_g, scale_factor=r, mode="linear", align_corners=False)
            y_self = V(mel_g).squeeze().cpu().numpy()
            mel_real = mel_for_vocoder(gt[0], hop=v_hop).to(DEV)
            if mel_real.dim() == 2:
                mel_real = mel_real.unsqueeze(0)
            y_ceil = V(mel_real).squeeze().cpu().numpy()
        sg = metrics(np.asarray(w))
        sc = metrics(y_ceil)
        ss = metrics(y_self)
        rows.append({
            "path": str(d["path"]), "feat": str(f), "speaker": spk,
            "gt_sharp": sg[0], "ceiling_sharp": sc[0], "self_sharp": ss[0],
            "gt_hf": sg[1], "ceiling_hf": sc[1], "self_hf": ss[1],
        })

    by_spk_gc: dict[str, list[float]] = defaultdict(list)
    by_spk_cs: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_spk_gc[r["speaker"]].append(r["gt_sharp"] - r["ceiling_sharp"])
        by_spk_cs[r["speaker"]].append(r["ceiling_sharp"] - r["self_sharp"])

    mean_gc, ci_gc, n_spk = bootstrap_speaker(by_spk_gc, args.boot)
    mean_cs, ci_cs, _ = bootstrap_speaker(by_spk_cs, args.boot)
    eq = tost_equivalent(mean_cs, ci_cs, args.equiv_delta)

    gt = np.array([r["gt_sharp"] for r in rows])
    ce = np.array([r["ceiling_sharp"] for r in rows])
    se = np.array([r["self_sharp"] for r in rows])

    out = {
        "script": "stable_res_eval2.py",
        "protocol": "rev4_abi_matched",
        "argv": sys.argv,
        "same_vocoder_both_arms": True,
        "abi_matched": bool(abi_ok),
        "v_hop": v_hop, "v_nfft": v_nfft, "v_causal": v_causal,
        "g_upsample": up, "g_frame_hop": g_frame_hop,
        "g_path": args.g, "g_sha256": sha256(args.g),
        "v_path": args.v, "v_sha256": sha256(args.v),
        "held_manifest": str(Path(args.out).parent / "held_manifest.txt"),
        "n_utt": len(rows),
        "n_speakers": n_spk,
        "speakers": sorted(by_spk_cs.keys()),
        "bootstrap": "speaker_level",
        "gt_sharp_mean": float(gt.mean()) if len(gt) else None,
        "ceiling_sharp_mean": float(ce.mean()) if len(ce) else None,
        "self_sharp_mean": float(se.mean()) if len(se) else None,
        "paired_gt_minus_ceiling_mean_speaker": mean_gc,
        "paired_gt_minus_ceiling_ci95_speaker": ci_gc,
        "paired_ceiling_minus_self_mean_speaker": mean_cs,
        "paired_ceiling_minus_self_ci95_speaker": ci_cs,
        "equivalence_ceiling_self": eq,
        "interpretation_guardrails": [
            "CI containing 0 is NOT equivalence of equivalence.",
            "Only equivalence_ceiling_self.equivalent_by_ci_in_margin may claim near-equality.",
            "V degradation requires comparing ceiling(V_init) vs ceiling(V_snap) on SAME hop ABI.",
            "Do not compare freebig hop512 ceiling to freeC hop128 ceiling as co-training loss.",
        ],
        "per_utt": rows,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    shutil.copyfile(args.held, outp.parent / "held_manifest.txt")
    print(
        f"[{Path(args.v).name}] hop={v_hop} nfft={v_nfft} causal={v_causal} "
        f"abi_ok={abi_ok} n_utt={len(rows)} n_spk={n_spk}"
    )
    print(
        f"  gt {gt.mean():.2f} ceiling {ce.mean():.2f} self {se.mean():.2f} | "
        f"ceiling->self (spk-boot) {mean_cs:+.2f} CI[{ci_cs[0]:+.2f},{ci_cs[1]:+.2f}] "
        f"equiv@δ={args.equiv_delta}: {eq['equivalent_by_ci_in_margin']}"
    )
    print(f"  -> {outp}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import DEVICE, DAC_SR, load_dac, hard_quantize_all
import kansei_proxies as kp

RESULTS = Path("../results")
FT_WEIGHTS = Path("../models/dac_44khz_finetuned.safetensors")

GATE0_THRESHOLDS = {
    "hf_preserve":          (">", 0.70),
    "brilliance_preserve":  (">", 0.55),
    "eight_k_cliff_ratio":  (">", 0.50),
    "centroid_delta_hz":    (">", -350.0),
    "rolloff85_delta_hz":   (">", -800.0),
    "hf_flatness_delta":    ("<", 0.15),
    "sib_delta":            ("<", 0.020),
    "cpp_delta_db":         (">", -1.5),
    "hnr_delta_db":         (">", -3.0),
}


def load_finetuned_decoder():
    from safetensors.torch import load_file
    dac = load_dac()
    if not FT_WEIGHTS.exists():
        print(f"  WARN: {FT_WEIGHTS} missing; finetuned == base")
        return dac
    sd = load_file(str(FT_WEIGHTS))
    full = dac.state_dict()
    n = 0
    for k, v in sd.items():
        if k in full and full[k].shape == v.shape:
            full[k] = v
            n += 1
    dac.load_state_dict(full)
    dac.eval()
    print(f"  finetuned decoder: patched {n} tensors from {FT_WEIGHTS.name}")
    return dac


@torch.no_grad()
def roundtrip(dac, audio_44k: np.ndarray, quantize: bool = True) -> np.ndarray:
    x = torch.from_numpy(audio_44k).float().view(1, 1, -1).to(DEVICE)
    z = dac.encoder(x)
    z_use = hard_quantize_all(dac, z) if quantize else z
    out = dac.decoder(z_use).squeeze().detach().cpu().numpy().astype(np.float32)
    return out


def load_audio_44k(path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != DAC_SR:
        audio = librosa.resample(audio.astype(np.float64), orig_sr=sr,
                                 target_sr=DAC_SR).astype(np.float32)
    return audio


def gather_wavs(args) -> list[Path]:
    if args.wavs:
        return [Path(w) for w in args.wavs]
    if args.from_dir:
        d = Path(args.from_dir)
        wavs = sorted(d.rglob("*.wav"))
        rng = random.Random(args.seed)
        rng.shuffle(wavs)
        return wavs[:args.n]
    raise SystemExit("provide --wavs or --from-dir")


def check_gates(delta: dict) -> tuple[dict, list]:
    results = {}
    hard_fail = []
    for key, (op, thr) in GATE0_THRESHOLDS.items():
        v = delta.get(key)
        if v is None:
            continue
        ok = (v > thr) if op == ">" else (v < thr)
        results[key] = {"value": round(float(v), 4), "op": op, "thr": thr, "pass": bool(ok)}
        if not ok:
            hard_fail.append(key)
    return results, hard_fail


def summarize(rows: list, key: str) -> float:
    vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", nargs="*")
    ap.add_argument("--from-dir")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decoder", choices=["base", "finetuned", "both"], default="both")
    ap.add_argument("--tag", default="gate0")
    ap.add_argument("--export", type=int, default=0, help="export N orig/base/ft triplets for human AB")
    ap.add_argument("--fast", action="store_true", help="skip pyin metrics (faster)")
    args = ap.parse_args()

    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    wavs = gather_wavs(args)
    print(f"Gate 0 on {len(wavs)} wavs | decoder={args.decoder} | device={DEVICE}")

    dac_base = load_dac() if args.decoder in ("base", "both") else None
    dac_ft = load_finetuned_decoder() if args.decoder in ("finetuned", "both") else None

    export_dir = RESULTS / f"gate0_{args.tag}_ab"
    if args.export:
        export_dir.mkdir(exist_ok=True)

    per_path = {"ceiling": [], "base": [], "finetuned": []}
    per_utt = []
    full = not args.fast

    for i, w in enumerate(wavs):
        try:
            orig = load_audio_44k(w)
            if len(orig) < DAC_SR // 2:
                continue
            rec = {}
            dref = dac_ft if dac_ft is not None else dac_base
            rec["ceiling"] = roundtrip(dref, orig, quantize=False)
            if dac_base is not None:
                rec["base"] = roundtrip(dac_base, orig, quantize=True)
            if dac_ft is not None:
                rec["finetuned"] = roundtrip(dac_ft, orig, quantize=True)

            entry = {"utt": w.name, "path": str(w)}
            for pth, ry in rec.items():
                cmp = kp.compare(orig, ry, full=full)
                per_path[pth].append(cmp["delta"])
                entry[pth] = cmp["delta"]
            per_utt.append(entry)

            if args.export and i < args.export:
                stem = f"{i:02d}_{w.stem[:20]}"
                sf.write(export_dir / f"{stem}_orig.wav", orig, DAC_SR)
                for pth, ry in rec.items():
                    sf.write(export_dir / f"{stem}_{pth}.wav", ry, DAC_SR)

            print(f"  [{i+1}/{len(wavs)}] {w.name[:36]:36s} "
                  f"hf_pres={entry.get('finetuned',entry.get('base',{})).get('hf_preserve',0):.2f} "
                  f"cliff={entry.get('finetuned',entry.get('base',{})).get('eight_k_cliff_ratio',0):.2f}",
                  flush=True)
        except Exception as e:
            print(f"  SKIP {w.name}: {e}")

    report = {"tag": args.tag, "n": len(per_utt), "decoder": args.decoder,
              "thresholds": GATE0_THRESHOLDS, "paths": {}}
    for pth, rows in per_path.items():
        if not rows:
            continue
        agg = {k: round(summarize(rows, k), 4) for k in rows[0].keys()}
        gates, fails = check_gates(agg)
        report["paths"][pth] = {"mean_delta": agg, "gates": gates,
                                "hard_fail": fails, "verdict": "PASS" if not fails else "FAIL"}

    out_path = RESULTS / f"gate0_{args.tag}.json"
    out_path.write_text(json.dumps({"report": report, "per_utt": per_utt},
                                   indent=2, ensure_ascii=False))

    print("\n=== Gate 0 objective summary ===")
    for pth, r in report["paths"].items():
        print(f"[{pth:9s}] {r['verdict']}  fails={r['hard_fail']}")
        md = r["mean_delta"]
        print(f"           hf_preserve={md.get('hf_preserve')}, cliff_ratio={md.get('eight_k_cliff_ratio')}, "
              f"centroid_delta={md.get('centroid_delta_hz')}, hf_flat_delta={md.get('hf_flatness_delta')}, "
              f"sib_delta={md.get('sib_delta')}, cpp_delta={md.get('cpp_delta_db')}, mel_l1={md.get('mel_l1_db')}")
    print(f"\nwrote {out_path}  ({time.time()-t0:.0f}s)")
    if args.export:
        print(f"human-AB triplets: {export_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import (DEVICE, DAC_SR, PairDataset, collate,
                           soft_rvq_requantize, hard_rvq_requantize, hard_quantize_all)
from torch.utils.data import DataLoader
from eval_phase3c_full import load_checkpoint
from gate0_codec import load_finetuned_decoder
import kansei_proxies as kp

ABS_KEYS = ["hf_ratio", "sib_ratio", "centroid_hz", "flatness", "hf_flatness",
            "eight_k_cliff", "cpp_db", "hnr_db", "tilt_db_per_khz"]


def np_audio(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().detach().cpu().numpy().astype(np.float32)


def save_clip(y: np.ndarray, out_dir: Path, name: str, sr: int = DAC_SR, clip_s: float = 3.0, start: float = 0.3) -> None:
    a, b = int(start * sr), int((start + clip_s) * sr)
    seg = y[a:b] if b <= len(y) else (y[-int(clip_s * sr):] if len(y) > int(clip_s * sr) else y)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / name, np.clip(seg, -1, 1).astype(np.float32), sr, subtype="PCM_16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/phase3c_ao_b1_ecapa/best.pt")
    ap.add_argument("--data-dir", default="../data/phase3_10k")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--export-audio", type=int, default=0,
                    help="save N diag trios (source/oracle/gen) for listen_gui.py")
    ap.add_argument("--export-dir", default=None)
    args = ap.parse_args()
    export_dir = Path(args.export_dir) if args.export_dir else \
        Path("../results") / f"diag_{Path(args.checkpoint).parent.name}"

    ck, generator, adapter = load_checkpoint(args.checkpoint)
    ck_args = ck["args"]
    tau = ck_args.get("tau", 5.0)
    max_frames = ck_args.get("max_frames", 256)
    dac = load_finetuned_decoder()
    if generator:
        generator = generator.to(DEVICE).eval()
    adapter = adapter.to(DEVICE).eval()

    ds = PairDataset(Path(args.data_dir) / "eval", max_frames)
    ds.files = [f for f in ds.files if not f.stem.endswith("_feat")]
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    print(f"generated-latent Gate 0 | ckpt={args.checkpoint} | tau={tau} | pairs<= {args.n}")

    agg = {p: defaultdict(list) for p in ("source", "oracle", "gen")}
    cmp_agg = defaultdict(list)
    delta_norms = []
    done = 0

    for batch in dl:
        z_s, q0_s, z_t, f0, energy, timbre, ref = [
            (x.to(DEVICE) if torch.is_tensor(x) else x) for x in batch]
        with torch.no_grad():
            if generator:
                z_pred = generator(z_s.transpose(1, 2), f0, energy, timbre).transpose(1, 2)
            else:
                z_pred = z_s
            z_q = soft_rvq_requantize(dac, q0_s, z_pred, tau)
            z_gen = adapter(z_q, timbre, z_t)
            gen_audio = dac.decoder(z_gen).squeeze(1)
            src_audio = dac.decoder(hard_quantize_all(dac, z_s)).squeeze(1)
            ora_audio = dac.decoder(hard_rvq_requantize(dac, q0_s, z_t)).squeeze(1)
            dnorm = (z_gen - z_q).pow(2).mean(dim=(1, 2)).sqrt().cpu().numpy()

        for i in range(gen_audio.shape[0]):
            g, s, o = np_audio(gen_audio[i]), np_audio(src_audio[i]), np_audio(ora_audio[i])
            mg, ms, mo = (kp.analyze(g, full=False), kp.analyze(s, full=False),
                          kp.analyze(o, full=False))
            for k in ABS_KEYS:
                agg["gen"][k].append(mg[k]); agg["source"][k].append(ms[k]); agg["oracle"][k].append(mo[k])
            cd = kp.compare(o, g, full=False)["delta"]
            for k in ("hf_preserve", "brilliance_preserve", "hf_flatness_delta",
                      "sib_delta", "cpp_delta_db", "centroid_delta_hz", "mel_delta_var_ratio"):
                cmp_agg[k].append(cd[k])
            delta_norms.append(float(dnorm[i]))
            if done < args.export_audio:
                for sig, role in [(s, "source"), (o, "oracle"), (g, "gen")]:
                    save_clip(sig, export_dir, f"diag{done}_{role}.wav")
            done += 1
        print(f"  {done} pairs", flush=True)
        if done >= args.n:
            break

    def m(d: dict) -> dict:
        return {k: round(float(np.mean(v)), 4) for k, v in d.items() if v}

    report = {
        "checkpoint": args.checkpoint, "n": done, "tau": tau,
        "delta_norm_mean": round(float(np.mean(delta_norms)), 4),
        "absolute": {p: m(agg[p]) for p in agg},
        "gen_vs_oracle_delta": m(cmp_agg),
    }
    out = Path(args.out) if args.out else Path("../results") / f"gate0_generated_{Path(args.checkpoint).parent.name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== absolute proxies (higher hf_flatness / lower hf_ratio+cpp = worse) ===")
    print(f"{'path':8} {'hf_ratio':>9} {'hf_flat':>8} {'flatness':>9} {'cpp_db':>8} {'hnr_db':>8} {'cliff':>7} {'centroid':>9}")
    for p in ("source", "oracle", "gen"):
        a = report["absolute"][p]
        print(f"{p:8} {a['hf_ratio']:>9} {a['hf_flatness']:>8} {a['flatness']:>9} {a['cpp_db']:>8} {a['hnr_db']:>8} {a['eight_k_cliff']:>7} {a['centroid_hz']:>9}")
    print(f"\ndelta_norm (gen vs pre-adapter) = {report['delta_norm_mean']}")
    print(f"gen_vs_oracle: {report['gen_vs_oracle_delta']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

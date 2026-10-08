"""生成潜在の誤差を時間変調帯域に分解して欠陥を帰属する(0学習・2026-09-23)。

e = z_gen − z_gt(生スケール)を時間方向FFTで 低(<4Hz)/中(4–16Hz)/高(16–50Hz, 100fps) に分け、
gt+e_band を decode して生参照(gt decode)との差を測る。gt_lp16=GT自身の高帯域を除去(欠落か誤りかの切り分け)・gen_lp16=生成出力自身の高帯域を除去
(非因果FFT=診断専用・製品経路ではない)。f0_st=decode音のf0中央値の参照比[半音]。どの帯域の誤差がコーラス型(調波スメア:
d_comb<0・d_ncc<0)や高域ノイズ(hi_mid)や包絡誤差(env)を生むかを帰属する。
測定は eval_d1_g0 v2 と同じ物差し(生スケールdecode・元wav一致assert)。

    CUDA_VISIBLE_DEVICES=0 uv run python probe_err_bands.py --arms s11_cfm_melin,d1_g1full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d1_model import sample_frame_ar
from train_d1 import build_index, cond_of
from train_cfmys import LAT, F0FIX, sample_k
from eval_d1_g0 import load_arm, pick_utts, assert_ref_matches_source, logmel_l1
from eval_d4b_gates import band_metrics
from diag_cfm_audit import decode_f0
import chorus_proxy as cp

ROOT = Path(__file__).resolve().parent.parent
BANDS = {"low": (0.0, 4.0), "mid": (4.0, 16.0), "high": (16.0, 50.01)}


def band_part(e: torch.Tensor, lo: float, hi: float, fps: float = 100.0) -> torch.Tensor:
    E = torch.fft.rfft(e, dim=-1)
    f = torch.fft.rfftfreq(e.shape[-1], d=1.0 / fps).to(e.device)
    m = ((f >= lo) & (f < hi)).to(E.dtype)
    return torch.fft.irfft(E * m, n=e.shape[-1], dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="s11_cfm_melin,d1_g1full")
    ap.add_argument("--n-spk", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-wav-utt", type=int, default=0, help="この番号の発話のみwav保存")
    ap.add_argument("--out", default="results/earbattery/err_bands")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from causal_codec import CausalCodec
    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev,
                     weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()
    abi = torch.load(LAT / "abi.pt", map_location=dev, weights_only=False)
    MU, SD = abi["mu"].to(dev), abi["sd"].to(dev)
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt", map_location="cpu",
                         weights_only=False)
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    def dec(z_raw):
        with torch.no_grad():
            return codec.decode(z_raw[None].to(dev))[0, 0].cpu().numpy().astype(np.float64)

    pairs, lats, held_spk = build_index(0)
    utts = pick_utts(pairs, lats, held_spk, "train", a.n_spk)
    mel_cache: dict = {}

    def mel_of(f, d):
        if f not in mel_cache:
            import librosa
            from causal_mel import causal_mel
            wv, _ = librosa.load(d["path"], sr=44100, mono=True)
            mel_cache[f] = causal_mel(torch.from_numpy(wv) * 32768.0, n_fft=1024, hop=256,
                                      num_mels=80, sr=44100)[0].half()
        return mel_cache[f]

    res = {"bands_hz": BANDS, "seed": a.seed, "utts": [f.stem for f in utts], "arms": {}}
    for tag in a.arms.split(","):
        kind, net, ck = load_arm(tag, dev, "best" if "cfm" in tag else "last")
        cli = ck["cli"]
        rows = {k: {"d_comb": [], "d_ncc": [], "hi_mid": [], "env": []}
                for k in ("full", "low", "mid", "high", "full_minus_high", "gt_lp16", "gen_lp16")}
        for r in rows.values():
            r["f0_st"] = []
        gen_hi_over_gt_hi = []
        efrac = {k: [] for k in BANDS}
        for ui, f in enumerate(utts):
            d = torch.load(f, map_location="cpu", weights_only=False)
            d = {**d, "f0": torch.load(F0FIX / f.parent.name / f.name, map_location="cpu",
                                       weights_only=False)["f0"]}
            z = torch.load(lats[f.stem], map_location="cpu", weights_only=False)["z"].float().T
            T = min(z.shape[1], 600)
            zg = z[:, :T].to(dev)
            y_ref = dec(zg)
            assert_ref_matches_source(y_ref, d["path"], len(y_ref))
            m_ref = cp.measure(y_ref)
            use_mel = cli.get("mel80") if kind == "d1" else cli.get("mel_in")
            cond = cond_of(d, T, mel_of(f, d) if use_mel else None)[None].to(dev)
            with torch.no_grad():
                if kind == "d1":
                    zh = sample_frame_ar(net, cond, K=8, seed=a.seed,
                                         z0_rho=float(cli.get("z0_rho", 0)))
                    zr = (zh[0] * SD[:, None] + MU[:, None]).clamp(-8, 8)
                else:
                    s_ = None if cli.get("no_spk") else spk_emb.get(d.get("speaker"))
                    s_ = s_[None].to(dev) if s_ is not None else None
                    g = torch.Generator(device=dev).manual_seed(a.seed)
                    mu, sd = ck["abi"]["mu"].to(dev), ck["abi"]["sd"].to(dev)
                    zh = sample_k(net, T, cli.get("rho", 0.9), g, dev, cond, s_, 8)
                    zr = (zh[0] * sd[:, None] + mu[:, None]).clamp(-8, 8)
            e = zr - zg
            parts = {k: band_part(e, lo, hi) for k, (lo, hi) in BANDS.items()}
            tot = float(e.pow(2).sum())
            for k in BANDS:
                efrac[k].append(float(parts[k].pow(2).sum()) / max(tot, 1e-12))
            gt_hi = band_part(zg - zg.mean(-1, keepdim=True), *BANDS["high"])
            gen_hi = band_part(zr - zr.mean(-1, keepdim=True), *BANDS["high"])
            gen_hi_over_gt_hi.append(float(gen_hi.pow(2).sum() / gt_hi.pow(2).sum().clamp(min=1e-12)))
            variants = {"full": zg + e, "low": zg + parts["low"], "mid": zg + parts["mid"],
                        "high": zg + parts["high"], "full_minus_high": zg + e - parts["high"],
                        "gt_lp16": zg - gt_hi, "gen_lp16": zr - gen_hi}
            f0_ref = decode_f0(np.clip(y_ref, -1, 1))["f0_median"]
            for k, zv in variants.items():
                y = dec(zv)
                mg = cp.measure(y)
                rows[k]["d_comb"].append(mg["comb_db"] - m_ref["comb_db"])
                rows[k]["d_ncc"].append(mg["period_ncc"] - m_ref["period_ncc"])
                rows[k]["hi_mid"].append(band_metrics(y)["hi_mid"]
                                         / max(band_metrics(y_ref)["hi_mid"], 1e-4))
                rows[k]["env"].append(logmel_l1(y, y_ref))
                f0v = decode_f0(np.clip(y, -1, 1))["f0_median"]
                rows[k]["f0_st"].append(12 * np.log2(max(f0v, 1e-3) / max(f0_ref, 1e-3))
                                        if f0v > 0 and f0_ref > 0 else float("nan"))
                if ui == a.save_wav_utt:
                    soundfile.write(out_dir / f"{tag}_{f.stem}_{k}.wav",
                                    np.clip(y, -1, 1).astype(np.float32), 48000)
            if ui == a.save_wav_utt:
                soundfile.write(out_dir / f"gt_{f.stem}.wav",
                                np.clip(y_ref, -1, 1).astype(np.float32), 48000)
            print(f"  {tag} {f.stem} efrac " + " ".join(
                f"{k}:{efrac[k][-1]:.2f}" for k in BANDS), flush=True)
        res["arms"][tag] = {
            "ckpt": int(ck["step"]),
            "err_energy_frac_median": {k: round(float(np.median(v)), 3) for k, v in efrac.items()},
            "gen_high_energy_over_gt_high_median": round(float(np.median(gen_hi_over_gt_hi)), 3),
            "variants_median": {k: {m: round(float(np.nanmedian(v)), 3) for m, v in r.items()}
                                for k, r in rows.items()}}
        print(tag, json.dumps(res["arms"][tag], ensure_ascii=False), flush=True)
    (out_dir / "err_bands.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ys1_nrft(decoder潜在摂動頑健化FT)の評価と盲検A/B素材生成(prereg: results/ys1_nrft/prereg.yaml)。

GT decode 自体が変わるため参照は元音声。元decoder=s1_3_c32 EMA、FT decoder=ys1_nrft EMA(encoderは同一)。
1) held24発話(各話者先頭・≤8s): GT潜在→両decoder→logmel L1/mrstft 対元音声・f0差[半音]・有声率差
2) s11潜在(best・K16・seed0)を両decoderで: 同指標(train側6話者・held側3話者)
3) 合成雑音潜在(GT + 帯域雑音 rms0.2): 同指標
4) 盲検: 有声率の高い3発話 × {source, gt_orig, gt_ft, s11_orig, s11_ft} → results/earbattery/nrft_ab/
   (試行内で元音声にRMS整合→共通減衰・鍵は _key_聴取後に開く.json)

    CUDA_VISIBLE_DEVICES=0 uv run python eval_ys1_nrft.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft
from train_s1_3 import build_index as build_wav_index, load_wav
from train_d1 import build_index, cond_of
from train_cfmys import F0FIX, sample_k
from eval_d1_g0 import load_arm, pick_utts, assert_ref_matches_source
from eval_d4b_gates import band_metrics
from diag_cfm_audit import decode_f0
from render_d1_ab import norm_trial, pick_voiced
from train_ys1_nrft import unit_noise, BANDS

ROOT = Path(__file__).resolve().parent.parent
LAT = ROOT / "data/ys1_latent"
AB = ROOT / "results/earbattery/nrft_ab"


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def codec_from(path: Path) -> CausalCodec:
        ck = torch.load(path, map_location=dev, weights_only=False)
        c = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
        c.load_state_dict(ck["ema"])
        return c.eval()

    orig = codec_from(ROOT / "results/s1_3_c32/s1_3_c32_last.pt")
    ft = codec_from(ROOT / "results/ys1_nrft/ys1_nrft_last.pt")
    enc_same = all(torch.equal(a, b) for (ka, a), (kb, b) in
                   zip(orig.state_dict().items(), ft.state_dict().items()) if ka.startswith("encoder."))
    assert enc_same, "encoderが凍結されていない"
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    sd = torch.load(LAT / "abi.pt", map_location=dev, weights_only=False)["sd"]
    w_dim = sd / sd.pow(2).mean().sqrt()

    def metrics(y: torch.Tensor, x: torch.Tensor, f0x: dict) -> dict:
        n = min(y.shape[-1], x.shape[-1])
        y, x = y[..., :n], x[..., :n]
        yn = np.clip(y.detach().cpu().numpy().astype(np.float64).reshape(-1), -1, 1)
        fy = decode_f0(yn)
        return {"logmel": float(logmel_l1(y.reshape(1, 1, -1), x.reshape(1, 1, -1), mels)),
                "mrstft": float(mrstft(y.reshape(1, 1, -1), x.reshape(1, 1, -1))),
                "f0_st": (float(12 * np.log2(fy["f0_median"] / f0x["f0_median"]))
                          if fy["f0_median"] > 0 and f0x["f0_median"] > 0 else float("nan")),
                "d_voiced": fy["voiced_ratio"] - f0x["voiced_ratio"],
                "hi_mid_vs_src": band_metrics(yn)["hi_mid"] / max(band_metrics(
                    x.detach().cpu().numpy().astype(np.float64).reshape(-1))["hi_mid"], 1e-4)}

    def summarize(rows: list) -> dict:
        out = {}
        for k in rows[0]:
            v = np.array([r[k] for r in rows], dtype=np.float64)
            out[k] = {"mean": round(float(np.nanmean(v)), 4), "median": round(float(np.nanmedian(v)), 4)}
            if k == "f0_st":
                out[k]["median_abs"] = round(float(np.nanmedian(np.abs(v))), 3)
        return out

    rep: dict = {"note": "参照=元音声。orig=s1_3_c32 EMA / ft=ys1_nrft EMA"}
    _, held_wavs = build_wav_index()
    rows = {"orig": [], "ft": []}
    for p in held_wavs:
        w = load_wav(p)
        n = min(len(w), 8 * SAMPLE_RATE) // HOP_LENGTH * HOP_LENGTH
        x = torch.from_numpy(w[:n].copy()).to(dev)[None, None]
        f0x = decode_f0(np.clip(w[:n].astype(np.float64), -1, 1))
        with torch.no_grad():
            z = orig.encode(x)
            for nm, c in (("orig", orig), ("ft", ft)):
                rows[nm].append(metrics(c.decode(z), x, f0x))
    rep["held24_gt_decode"] = {k: summarize(v) for k, v in rows.items()}
    print("held24 GT decode:", json.dumps(rep["held24_gt_decode"], ensure_ascii=False), flush=True)

    kind, s11, ck = load_arm("s11_cfm_melin", dev, "best")
    cli = ck["cli"]
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt", map_location="cpu", weights_only=False)
    pairs, lats, held_spk = build_index(0)

    def s11_latent(f, T):
        d = torch.load(f, map_location="cpu", weights_only=False)
        d = {**d, "f0": torch.load(F0FIX / f.parent.name / f.name, map_location="cpu",
                                   weights_only=False)["f0"]}
        wv, _ = librosa.load(d["path"], sr=44100, mono=True)
        from causal_mel import causal_mel
        mel = causal_mel(torch.from_numpy(wv) * 32768.0, n_fft=1024, hop=256, num_mels=80,
                         sr=44100)[0].half()
        cond = cond_of(d, T, mel)[None].to(dev)
        s_ = spk_emb.get(d.get("speaker"))
        s_ = s_[None].to(dev) if s_ is not None else None
        g = torch.Generator(device=dev).manual_seed(0)
        mu, sdv = ck["abi"]["mu"].to(dev), ck["abi"]["sd"].to(dev)
        with torch.no_grad():
            zh = sample_k(s11, T, cli.get("rho", 0.9), g, dev, cond, s_, 16)
        return (zh[0] * sdv[:, None] + mu[:, None]).clamp(-8, 8), d

    def src_of(d, T):
        y, _ = librosa.load(d["path"], sr=SAMPLE_RATE, mono=True)
        return torch.from_numpy(y[:T * HOP_LENGTH].astype(np.float32)).to(dev)[None, None]

    for split in ("train", "held"):
        rows = {"orig": [], "ft": [], "orig_synth": [], "ft_synth": []}
        for f in pick_utts(pairs, lats, held_spk, split, 6 if split == "train" else 3):
            z = torch.load(lats[f.stem], map_location="cpu", weights_only=False)["z"].float().T
            T = min(z.shape[1], 600)
            zr, d = s11_latent(f, T)
            x = src_of(d, T)
            f0x = decode_f0(np.clip(x.cpu().numpy().reshape(-1).astype(np.float64), -1, 1))
            zg = z[:, :T].to(dev)
            with torch.no_grad():
                y0 = orig.decode(zg[None])
            assert_ref_matches_source(y0[0, 0].cpu().numpy(), d["path"], y0.shape[-1])
            gg = torch.Generator(device=dev).manual_seed(7)
            zs = zg + 0.2 * unit_noise(32, T, 0.0, 50.01, dev, gg) * w_dim[:, None]
            with torch.no_grad():
                for nm, c in (("orig", orig), ("ft", ft)):
                    rows[nm].append(metrics(c.decode(zr[None]), x, f0x))
                    rows[nm + "_synth"].append(metrics(c.decode(zs[None]), x, f0x))
        rep[f"s11K16_{split}"] = {k: summarize(v) for k, v in rows.items()}
        print(f"s11 K16 {split}:", json.dumps(rep[f"s11K16_{split}"], ensure_ascii=False), flush=True)
    (ROOT / "results/ys1_nrft/eval.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))

    AB.mkdir(parents=True, exist_ok=True)
    key: dict = {}
    for f in pick_voiced(pairs, lats, held_spk, 3):
        z = torch.load(lats[f.stem], map_location="cpu", weights_only=False)["z"].float().T
        T = min(z.shape[1], 600)
        zr, d = s11_latent(f, T)
        x = src_of(d, T)
        zg = z[:, :T].to(dev)
        with torch.no_grad():
            clips = {"source": x[0, 0].cpu().numpy().astype(np.float64),
                     "gt_orig": orig.decode(zg[None])[0, 0].cpu().numpy().astype(np.float64),
                     "gt_ft": ft.decode(zg[None])[0, 0].cpu().numpy().astype(np.float64),
                     "s11_orig": orig.decode(zr[None])[0, 0].cpu().numpy().astype(np.float64),
                     "s11_ft": ft.decode(zr[None])[0, 0].cpu().numpy().astype(np.float64)}
        normed = norm_trial(clips, clips["source"])
        names = list(normed)
        trial = f"nrft_{f.stem}"
        random.Random(trial).shuffle(names)
        td = AB / trial
        td.mkdir(parents=True, exist_ok=True)
        key[trial] = {"utt": f.stem, "map": {}}
        for i, nm in enumerate(names):
            soundfile.write(td / f"{'ABCDE'[i]}.wav", normed[nm], SAMPLE_RATE)
            key[trial]["map"]["ABCDE"[i]] = nm
        print(trial, "ok", flush=True)
    (AB / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

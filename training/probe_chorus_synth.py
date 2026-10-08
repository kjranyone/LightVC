"""コーラスの帰属: decoderの一般応答か、s11誤差に固有の構造か(0学習・盲検クリップ生成・2026-09-23)。

耳の既知事実: s11誤差は0–4Hz/4–16Hz帯だけでもコーラス・16–50Hz除去でも残る(帯域に局在しない)。
同一発話・s11(best・K8・seed0)の誤差 e=z_s11−z_gt(生スケール)に対し、以下をGTに足してdecode:
  s11_full(陽性対照) / s11_low×0.5 / s11_mid×0.5(大きさの閾値)
  synth_low/mid/high/all(帯域制限ガウス雑音・次元ごとにs11の同帯域誤差とエネルギー一致)
+ gt_decode(陰性の錨)。試行内RMS整合→共通減衰で X1..X8 に盲検化し、鍵は _key_聴取後に開く.json。
読み: 合成雑音でもコーラス→decoderが任意の潜在ずれをコーラス化(decoder/潜在幾何側)。
      合成雑音ではコーラスなし→s11誤差に固有の構造(生成器側)。×0.5で消える→大きさの閾値。

    CUDA_VISIBLE_DEVICES=0 uv run python probe_chorus_synth.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_d1 import build_index, cond_of
from train_cfmys import F0FIX, sample_k
from eval_d1_g0 import load_arm, pick_utts, assert_ref_matches_source
from eval_d4b_gates import band_metrics
from diag_cfm_audit import decode_f0
from probe_err_bands import band_part, BANDS
from render_d1_ab import norm_trial

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/earbattery/chorus_probe"


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from causal_codec import CausalCodec
    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev,
                     weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt", map_location="cpu",
                         weights_only=False)

    def dec(z_raw):
        with torch.no_grad():
            return codec.decode(z_raw[None].to(dev))[0, 0].cpu().numpy().astype(np.float64)

    pairs, lats, held_spk = build_index(0)
    f = pick_utts(pairs, lats, held_spk, "train", 6)[0]
    d = torch.load(f, map_location="cpu", weights_only=False)
    d = {**d, "f0": torch.load(F0FIX / f.parent.name / f.name, map_location="cpu",
                               weights_only=False)["f0"]}
    z = torch.load(lats[f.stem], map_location="cpu", weights_only=False)["z"].float().T
    T = min(z.shape[1], 600)
    zg = z[:, :T].to(dev)
    y_ref = dec(zg)
    assert_ref_matches_source(y_ref, d["path"], len(y_ref))

    kind, net, ck = load_arm("s11_cfm_melin", dev, "best")
    cli = ck["cli"]
    import librosa
    from causal_mel import causal_mel
    wv, _ = librosa.load(d["path"], sr=44100, mono=True)
    mel = causal_mel(torch.from_numpy(wv) * 32768.0, n_fft=1024, hop=256, num_mels=80,
                     sr=44100)[0].half()
    cond = cond_of(d, T, mel)[None].to(dev)
    s_ = spk_emb.get(d.get("speaker"))
    s_ = s_[None].to(dev) if s_ is not None else None
    g = torch.Generator(device=dev).manual_seed(0)
    mu, sd = ck["abi"]["mu"].to(dev), ck["abi"]["sd"].to(dev)
    with torch.no_grad():
        zh = sample_k(net, T, cli.get("rho", 0.9), g, dev, cond, s_, 8)
    zr = (zh[0] * sd[:, None] + mu[:, None]).clamp(-8, 8)
    e = zr - zg
    eb = {k: band_part(e, lo, hi) for k, (lo, hi) in BANDS.items()}

    torch.manual_seed(1234)

    def synth(target: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        n = band_part(torch.randn_like(target), lo, hi)
        scale = target.pow(2).sum(-1, keepdim=True).sqrt() / n.pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-9)
        return n * scale

    variants = {
        "gt_decode": zg,
        "s11_full": zr,
        "s11_low_x0.5": zg + 0.5 * eb["low"],
        "s11_mid_x0.5": zg + 0.5 * eb["mid"],
        "synth_low": zg + synth(eb["low"], *BANDS["low"]),
        "synth_mid": zg + synth(eb["mid"], *BANDS["mid"]),
        "synth_high": zg + synth(eb["high"], *BANDS["high"]),
        "synth_all": zg + synth(e, 0.0, 50.01),
    }
    clips = {k: dec(v) for k, v in variants.items()}
    f0r = decode_f0(np.clip(y_ref, -1, 1))["f0_median"]
    meas = {}
    for k, y in clips.items():
        f0v = decode_f0(np.clip(y, -1, 1))
        meas[k] = {"f0_st": round(float(12 * np.log2(max(f0v["f0_median"], 1e-3) / f0r)), 2),
                   "voiced": round(f0v["voiced_ratio"], 3),
                   "hi_mid_ratio": round(band_metrics(y)["hi_mid"] / max(band_metrics(y_ref)["hi_mid"], 1e-4), 3),
                   "latent_err_rms": round(float((variants[k] - zg).pow(2).mean().sqrt()), 4)}
    normed = norm_trial(clips, y_ref)
    names = list(normed)
    random.Random("chorus_probe_20260923").shuffle(names)
    OUT.mkdir(parents=True, exist_ok=True)
    key = {"utt": f.stem, "map": {}, "measures": meas}
    for i, nm in enumerate(names):
        soundfile.write(OUT / f"X{i + 1}.wav", normed[nm], 48000)
        key["map"][f"X{i + 1}"] = nm
    (OUT / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print(json.dumps(meas, ensure_ascii=False), flush=True)
    print("->", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

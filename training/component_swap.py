"""
成分スワップ Oracle — WORLD パラメータの各成分が話者性にどれだけ寄与するか測定。

WORLD パラメータ z = (f0, mcep, ap) を成分ごとに source/target 入れ替えて合成し、
SECS(target) と SECS(source) を測る。

Test  F0              mcep    ap      目的
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A     source shifted  target  source  現在の上限 (= O1d ≈ 0.35)
B     source shifted  target  target  AP が話者性を持つか
C     target          target  source  F0/prosody の寄与
D     target          target  target  WORLD target oracle (上限)
E     source          target  target  F0 shift の必要性
F     tgt mean only   target  target  完全 target F0 なしで足りるか

Go/No-Go:
  B >= 0.50 → AP/noise 学習に価値あり
  D >= 0.60 → WORLD パラメータ空間で目標品質に届く
  D < 0.40 → 低遅延 neural vocoder / NSF 方向へ移行
"""
import sys, os, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import pyworld as world
import pysptk as sptk
import torch
import torch.nn.functional as F
import librosa
from fastdtw import fastdtw

DEVICE = torch.device("cuda")
SR = 16000
FRAME_PERIOD = 5.0
FFTL = 2048
ALPHA = 0.410
MC_ORDER = 24

VCTK_WAV = Path("../data/vctk_200")


def analyze_wav(wav_path):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=SR)
    wav = wav.astype(np.float64)

    f0, t = world.dio(wav, SR, frame_period=FRAME_PERIOD)
    f0 = world.stonemask(wav, f0, t, SR)
    sp = world.cheaptrick(wav, f0, t, SR, fft_size=FFTL)
    ap = world.d4c(wav, f0, t, SR, fft_size=FFTL)
    mc = sptk.sp2mc(sp, MC_ORDER, ALPHA)
    codeap = world.code_aperiodicity(ap, SR)

    return {
        "f0": f0.astype(np.float64),
        "mc": mc.astype(np.float64),
        "sp": sp.astype(np.float64),
        "ap": ap.astype(np.float64),
        "codeap": codeap.astype(np.float64),
    }


def synth(f0, mc_or_sp, ap_or_codeap, use_mc=True, use_codeap=True):
    if use_mc:
        mc = np.ascontiguousarray(mc_or_sp, dtype=np.float64)
        sp = sptk.mc2sp(mc, ALPHA, FFTL)
    else:
        sp = np.ascontiguousarray(mc_or_sp, dtype=np.float64)

    if use_codeap:
        codeap = np.ascontiguousarray(ap_or_codeap, dtype=np.float64)
        ap = world.decode_aperiodicity(codeap, SR, FFTL)
    else:
        ap = np.ascontiguousarray(ap_or_codeap, dtype=np.float64)

    f0 = np.ascontiguousarray(f0, dtype=np.float64)
    return world.synthesize(f0, sp, ap, SR, frame_period=FRAME_PERIOD).astype(np.float32)


def dtw_align(src_mc, tgt_mc):
    dist, path = fastdtw(src_mc, tgt_mc, radius=30)
    src_map = np.zeros(len(src_mc), dtype=int)
    last = 0
    for s, t in path:
        if s < len(src_mc) and t < len(tgt_mc):
            src_map[s] = t
            last = t
    for i in range(1, len(src_map)):
        if src_map[i] == 0:
            src_map[i] = src_map[i - 1]
    return src_map


def shift_f0(f0_src, f0_tgt_ref):
    voiced = f0_src[f0_src > 0]
    if len(voiced) == 0:
        return f0_src.copy()
    src_mean = np.exp(np.mean(np.log(voiced)))
    tgt_voiced = f0_tgt_ref[f0_tgt_ref > 0]
    if len(tgt_voiced) == 0:
        return f0_src.copy()
    tgt_mean = np.exp(np.mean(np.log(tgt_voiced)))
    ratio = tgt_mean / src_mean
    return np.where(f0_src > 0, f0_src * ratio, 0.0)


def find_same_text_pairs(n_pairs=20):
    text_groups = defaultdict(list)
    for spk_dir in sorted(VCTK_WAV.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for wav_path in spk_dir.glob("*.wav"):
            parts = wav_path.stem.split("_")
            if len(parts) >= 2:
                text_groups[parts[1]].append((spk, str(wav_path)))

    pairs = []
    used = set()
    for text_id, utts in sorted(text_groups.items()):
        if len(utts) < 2:
            continue
        for i in range(len(utts)):
            for j in range(i + 1, len(utts)):
                sa, wa = utts[i]
                sb, wb = utts[j]
                if sa == sb or sa in used or sb in used:
                    continue
                pairs.append({"src_spk": sa, "src_wav": wa, "tgt_spk": sb, "tgt_wav": wb})
                used.add(sa)
                used.add(sb)
                if len(pairs) >= n_pairs:
                    return pairs
    return pairs


def load_secs():
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )


def secs(model, wav_ref, wav_syn):
    with torch.no_grad():
        e_ref = model.encode_batch(torch.from_numpy(wav_ref.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
        e_syn = model.encode_batch(torch.from_numpy(wav_syn.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
    return F.cosine_similarity(e_ref, e_syn, dim=-1).item()


def main():
    print("=== 成分スワップ Oracle ===\n")
    secs_model = load_secs()
    pairs = find_same_text_pairs(20)
    print(f"ペア数: {len(pairs)}\n")

    results = defaultdict(list)

    for idx, p in enumerate(pairs):
        try:
            feat_s = analyze_wav(p["src_wav"])
            feat_t = analyze_wav(p["tgt_wav"])
        except Exception as e:
            print(f"  [{idx+1}] 分析エラー: {e}")
            continue

        src_map = dtw_align(feat_s["mc"], feat_t["mc"])
        T = len(feat_s["f0"])

        mc_t_aligned = feat_t["mc"][src_map[:T]]
        ap_t_aligned = feat_t["ap"][src_map[:T]]
        codeap_t_aligned = feat_t["codeap"][src_map[:T]]
        ap_s = feat_s["ap"][:T]
        codeap_s = feat_s["codeap"][:T]
        f0_s = feat_s["f0"][:T]
        f0_t_aligned = feat_t["f0"][src_map[:T]]

        f0_s_shifted = shift_f0(f0_s, feat_t["f0"])

        tgt_voiced = feat_t["f0"][feat_t["f0"] > 0]
        tgt_mean_f0 = np.exp(np.mean(np.log(tgt_voiced))) if len(tgt_voiced) > 0 else 200.0
        vuv_s = (f0_s > 0).astype(float)
        f0_tgt_mean_only = np.where(vuv_s > 0, tgt_mean_f0, 0.0)

        tests = {
            "A_srcF0shift_tgtMC_srcAP": lambda: synth(f0_s_shifted[:T], mc_t_aligned[:T], codeap_s[:T], use_mc=True, use_codeap=True),
            "B_srcF0shift_tgtMC_tgtAP": lambda: synth(f0_s_shifted[:T], mc_t_aligned[:T], codeap_t_aligned[:T], use_mc=True, use_codeap=True),
            "C_tgtF0_tgtMC_srcAP":      lambda: synth(f0_t_aligned[:T], mc_t_aligned[:T], codeap_s[:T], use_mc=True, use_codeap=True),
            "D_tgtF0_tgtMC_tgtAP":      lambda: synth(f0_t_aligned[:T], mc_t_aligned[:T], codeap_t_aligned[:T], use_mc=True, use_codeap=True),
            "E_srcF0_tgtMC_tgtAP":      lambda: synth(f0_s[:T], mc_t_aligned[:T], codeap_t_aligned[:T], use_mc=True, use_codeap=True),
            "F_tgtMeanF0_tgtMC_tgtAP":  lambda: synth(f0_tgt_mean_only[:T], mc_t_aligned[:T], codeap_t_aligned[:T], use_mc=True, use_codeap=True),
        }

        wav_tgt_ref, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR:
            wav_tgt_ref = librosa.resample(wav_tgt_ref, orig_sr=sr, target_sr=SR)
        wav_src_ref, sr = sf.read(p["src_wav"], dtype="float32")
        if sr != SR:
            wav_src_ref = librosa.resample(wav_src_ref, orig_sr=sr, target_sr=SR)

        row = f"  [{idx+1}/{len(pairs)}] {p['src_spk']}→{p['tgt_spk']}:"
        for name, fn in tests.items():
            try:
                wav_syn = fn()
                tgt_sim = secs(secs_model, wav_tgt_ref, wav_syn)
                results[name].append(tgt_sim)
                row += f" {name[0]}={tgt_sim:.3f}"
            except Exception as e:
                row += f" {name[0]}=ERR"
                results[name].append(0.0)
        print(row, flush=True)

    print("\n=== 成分スワップ結果 ===")
    print(f"{'Test':40s} {'SECS(tgt)':>10s} {'±std':>8s}  目的")
    print("─" * 90)
    purposes = {
        "A_srcF0shift_tgtMC_srcAP": "現在の上限 (O1d相当)",
        "B_srcF0shift_tgtMC_tgtAP": "APが話者性を持つか",
        "C_tgtF0_tgtMC_srcAP":      "F0/prosodyの寄与",
        "D_tgtF0_tgtMC_tgtAP":      "WORLD target oracle (上限)",
        "E_srcF0_tgtMC_tgtAP":      "F0 shiftの必要性",
        "F_tgtMeanF0_tgtMC_tgtAP":  "完全target F0なしで足りるか",
    }
    for name in ["A_srcF0shift_tgtMC_srcAP", "B_srcF0shift_tgtMC_tgtAP", "C_tgtF0_tgtMC_srcAP",
                  "D_tgtF0_tgtMC_tgtAP", "E_srcF0_tgtMC_tgtAP", "F_tgtMeanF0_tgtMC_tgtAP"]:
        arr = np.array(results[name])
        print(f"{name:40s} {arr.mean():10.4f} {arr.std():8.4f}  {purposes[name]}")

    b_mean = np.mean(results["B_srcF0shift_tgtMC_tgtAP"])
    d_mean = np.mean(results["D_tgtF0_tgtMC_tgtAP"])
    print(f"\n=== Go/No-Go ===")
    print(f"  B (tgt mcep + tgt ap + src-shift F0) = {b_mean:.4f}  → >= 0.50 ? {'GO' if b_mean >= 0.50 else 'NO-GO'}")
    print(f"  D (tgt mcep + tgt ap + tgt F0)       = {d_mean:.4f}  → >= 0.60 ? {'GO' if d_mean >= 0.60 else 'NO-GO'}")
    print(f"  D < 0.40 ? → {'NSF移行推奨' if d_mean < 0.40 else 'WORLD継続可能'}")

    out = {name: {"mean": float(np.mean(arr)), "std": float(np.std(arr))} for name, arr in results.items()}
    with open("results/component_swap.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n結果保存: results/component_swap.json")


if __name__ == "__main__":
    main()

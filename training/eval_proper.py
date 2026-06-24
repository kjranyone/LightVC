"""
proper eval — component_swap と同じ評価方法でモデルを測る。

same-text pair を分析→モデル予測→WORLD合成→
実際のtarget発話とSECS比較。

F0 shiftあり、source AP使用（component swap test A と同じ条件）。
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

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device("cuda")
SR = 16000
FRAME_PERIOD = 5.0
FFTL = 2048
ALPHA = 0.410
MC_ORDER = 24
VCTK_WAV = Path("../data/vctk_200")


def analyze_wav(wav_path):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1: wav = wav[:, 0]
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=SR)
    wav = wav.astype(np.float64)
    f0, t = world.dio(wav, SR, frame_period=FRAME_PERIOD)
    f0 = world.stonemask(wav, f0, t, SR)
    sp = world.cheaptrick(wav, f0, t, SR, fft_size=FFTL)
    ap = world.d4c(wav, f0, t, SR, fft_size=FFTL)
    mc = sptk.sp2mc(sp, MC_ORDER, ALPHA)
    return {"f0": f0.astype(np.float32), "mc": mc.astype(np.float32), "ap": ap}


def synth(f0, mc, ap):
    mc64 = np.ascontiguousarray(mc, dtype=np.float64)
    sp = sptk.mc2sp(mc64, ALPHA, FFTL)
    ap64 = np.ascontiguousarray(ap, dtype=np.float64)
    f064 = np.ascontiguousarray(f0, dtype=np.float64)
    return world.synthesize(f064, sp, ap64, SR, frame_period=FRAME_PERIOD).astype(np.float32)


def shift_f0(f0, tgt_mean):
    voiced = f0[f0 > 0]
    if len(voiced) == 0: return f0.astype(np.float64)
    src_mean = float(np.exp(np.mean(np.log(voiced))))
    return np.where(f0 > 0, f0 * tgt_mean / src_mean, 0).astype(np.float64)


def find_pairs(n=20):
    groups = defaultdict(list)
    for d in sorted(VCTK_WAV.iterdir()):
        if not d.is_dir(): continue
        for w in d.glob("*.wav"):
            parts = w.stem.split("_")
            if len(parts) >= 2:
                groups[parts[1]].append((d.name, str(w)))
    pairs = []
    used = set()
    for tid, utts in sorted(groups.items()):
        if len(utts) < 2: continue
        for i in range(len(utts)):
            for j in range(i+1, len(utts)):
                sa, wa = utts[i]; sb, wb = utts[j]
                if sa == sb or sa in used or sb in used: continue
                pairs.append({"src": sa, "src_wav": wa, "tgt": sb, "tgt_wav": wb, "text": tid})
                used.add(sa); used.add(sb)
                if len(pairs) >= n: return pairs
    return pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_eval", type=int, default=20)
    args = parser.parse_args()

    print("=== モデル proper評価 (same-utterance ref) ===\n")

    from train_sf_vc import EnvelopeConverter, load_speaker_embeddings
    spk_emb = load_speaker_embeddings()
    spk_dim = next(iter(spk_emb.values())).shape[0]

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = EnvelopeConverter(mc_dim=cfg["mc_dim"], spk_dim=cfg["spk_dim"],
                               hidden=cfg["hidden"], n_blocks=cfg["n_blocks"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    pairs = find_pairs(args.n_eval)
    print(f"ペア数: {len(pairs)}\n")

    model_scores = []
    oracle_scores = []

    for idx, p in enumerate(pairs):
        feat_s = analyze_wav(p["src_wav"])
        feat_t = analyze_wav(p["tgt_wav"])

        mc_s = feat_s["mc"]
        f0_s = feat_s["f0"]
        ap_s = feat_s["ap"]
        mc_t = feat_t["mc"]
        f0_t = feat_t["f0"]
        ap_t = feat_t["ap"]

        T = len(mc_s)

        # F0 shift
        tgt_voiced = f0_t[f0_t > 0]
        tgt_mean_f0 = float(np.exp(np.mean(np.log(tgt_voiced)))) if len(tgt_voiced) > 0 else 200.0
        f0_shifted = shift_f0(f0_s, tgt_mean_f0)

        # モデル予測
        mc_s_t = torch.from_numpy(mc_s[:T]).float().unsqueeze(0).to(DEVICE).transpose(1, 2)
        spk_t = spk_emb[p["tgt"]].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mc_pred = model(mc_s_t, spk_t)
        mc_pred_np = mc_pred.squeeze(0).cpu().numpy().T

        # モデル合成: predicted mc + shifted F0 + source AP (test A条件)
        wav_model = synth(f0_shifted[:T], mc_pred_np[:T], ap_s[:T])

        # Oracle合成: DTW aligned target mc + shifted F0 + source AP
        dist, path = fastdtw(mc_s, mc_t, radius=30)
        src_map = np.zeros(T, dtype=int)
        for s, t in path:
            if s < T: src_map[s] = min(t, len(mc_t)-1)
        for i in range(1, T):
            if src_map[i] == 0: src_map[i] = src_map[i-1]
        mc_t_aligned = mc_t[src_map]
        wav_oracle = synth(f0_shifted[:T], mc_t_aligned[:T], ap_s[:T])

        # 参照: 実際のtarget発話
        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt, orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = emb(wav_tgt)
            e_model = emb(wav_model)
            e_oracle = emb(wav_oracle)

        m_sim = F.cosine_similarity(e_tgt, e_model, dim=-1).item()
        o_sim = F.cosine_similarity(e_tgt, e_oracle, dim=-1).item()
        model_scores.append(m_sim)
        oracle_scores.append(o_sim)
        print(f"  [{idx+1}/{len(pairs)}] {p['src']}→{p['tgt']}: model={m_sim:.3f} oracle={o_sim:.3f}", flush=True)

    m_arr = np.array(model_scores)
    o_arr = np.array(oracle_scores)
    print(f"\n=== 結果 (same-utterance ref, F0 shift, source AP) ===")
    print(f"モデル:   {m_arr.mean():.4f} ± {m_arr.std():.4f}")
    print(f"Oracle:   {o_arr.mean():.4f} ± {o_arr.std():.4f}")
    print(f"モデル/Oracle: {m_arr.mean()/o_arr.mean():.1%}")


if __name__ == "__main__":
    main()

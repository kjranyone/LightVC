"""
v1モデル再評価 — same-utterance reference で正しいSECSを測る。

generic reference (別発話) だとECAPAのcontent sensitivityで
SECSが半分程度に低下する。same-utterance ref がVC標準評価。
"""
import sys, os, json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import pyworld as world
import pysptk as sptk
import librosa
from fastdtw import fastdtw

sys.path.insert(0, str(Path(__file__).parent))
from train_sf_vc import EnvelopeConverter, load_speaker_embeddings, SFPairDataset

DEVICE = torch.device("cuda")
SR = 16000
FFTL = 2048
ALPHA = 0.410
FP = 5.0
VCTK_WAV = Path("../data/vctk_200")
MC_CACHE = Path("data/mc_cache")
PAIRS_DIR = Path("data/sf_pairs")


def shift_f0(f0, tgt_mean_f0):
    voiced = f0[f0 > 0]
    if len(voiced) == 0: return f0
    src_mean = float(np.exp(np.mean(np.log(voiced))))
    ratio = tgt_mean_f0 / src_mean
    return np.where(f0 > 0, f0 * ratio, 0).astype(np.float64)


def synth_mc(f0, mc, codeap):
    mc64 = np.ascontiguousarray(mc, dtype=np.float64)
    sp = sptk.mc2sp(mc64, ALPHA, FFTL)
    codeap64 = np.ascontiguousarray(codeap, dtype=np.float64)
    ap = world.decode_aperiodicity(codeap64, SR, FFTL) if codeap.shape[1] > 0 else np.ones_like(sp)
    f064 = np.ascontiguousarray(f0, dtype=np.float64)
    return world.synthesize(f064, sp, ap, SR, frame_period=FP).astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_eval", type=int, default=20)
    args = parser.parse_args()

    print("=== v1モデル same-utterance再評価 ===\n")
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

    f0_stats = {}
    for spk_dir in sorted(MC_CACHE.iterdir()):
        if not spk_dir.is_dir(): continue
        spk = spk_dir.name
        f0s = []
        for npz_path in spk_dir.glob("*.npz"):
            d = np.load(npz_path)
            v = d["f0"][d["f0"] > 0]
            if len(v) > 0: f0s.extend(v.tolist())
        if f0s: f0_stats[spk] = float(np.exp(np.mean(np.log(np.array(f0s)))))

    pair_files = sorted(PAIRS_DIR.glob("pair_*.npz"))[:args.n_eval]

    scores_generic = []
    scores_sameutt = []
    oracle_scores = []

    for i, npz_path in enumerate(pair_files):
        data = np.load(npz_path, allow_pickle=True)
        mc_src = data["mc_src"]
        mc_tgt = data["mc_tgt"]
        f0_src = data["f0_src"]
        codeap_tgt = data["codeap_tgt"]
        codeap_src = data.get("codeap_src", None)
        spk_src = str(data["spk_src"])
        spk_tgt = str(data["spk_tgt"])

        T = min(len(mc_src), 800)

        mc_src_t = torch.from_numpy(mc_src[:T]).float().unsqueeze(0).to(DEVICE).transpose(1, 2)
        spk_t = spk_emb[spk_tgt].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mc_pred = model(mc_src_t, spk_t)

        mc_pred_np = mc_pred.squeeze(0).cpu().numpy().T

        tgt_mean_f0 = f0_stats.get(spk_tgt, 200.0)
        f0_shifted = shift_f0(f0_src[:T].astype(np.float64), tgt_mean_f0)

        # モデル予測 + source AP
        codeap_for_synth = codeap_tgt[:T]
        wav_model = synth_mc(f0_shifted[:T], mc_pred_np[:T], codeap_for_synth[:T])

        # Oracle (ground truth target mcep)
        wav_oracle = synth_mc(f0_shifted[:T], mc_tgt[:T], codeap_tgt[:T])

        # same-utterance reference (実際のtarget発話)
        src_wavs = list((VCTK_WAV / spk_src).glob("*.wav"))
        tgt_wavs = list((VCTK_WAV / spk_tgt).glob("*.wav"))
        if not src_wavs or not tgt_wavs: continue

        wav_src_ref, sr = sf.read(str(src_wavs[0]), dtype="float32")
        wav_tgt_ref_generic, sr = sf.read(str(tgt_wavs[0]), dtype="float32")
        if sr != SR:
            wav_src_ref = librosa.resample(wav_src_ref, orig_sr=sr, target_sr=SR)
            wav_tgt_ref_generic = librosa.resample(wav_tgt_ref_generic, orig_sr=sr, target_sr=SR)

        # same-utterance: ペアのtarget発話を探す
        # sf_pairsのペアは同じtext_idの別話者なので、target発話のwav pathが必要
        # text_idから推測
        stem_parts = src_wavs[0].stem.split("_")
        text_id = stem_parts[1] if len(stem_parts) >= 2 else "001"
        tgt_sameutt_wavs = list((VCTK_WAV / spk_tgt).glob(f"*_{text_id}.wav"))
        if not tgt_sameutt_wavs:
            tgt_sameutt_wavs = tgt_wavs
        wav_tgt_sameutt, sr = sf.read(str(tgt_sameutt_wavs[0]), dtype="float32")
        if sr != SR:
            wav_tgt_sameutt = librosa.resample(wav_tgt_sameutt, orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            def get_emb(w):
                return secs_model.encode_batch(torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)

            e_tgt_gen = get_emb(wav_tgt_ref_generic)
            e_tgt_same = get_emb(wav_tgt_sameutt)
            e_model = get_emb(wav_model)
            e_oracle = get_emb(wav_oracle)

        gen_sim = F.cosine_similarity(e_tgt_gen, e_model, dim=-1).item()
        same_sim = F.cosine_similarity(e_tgt_same, e_model, dim=-1).item()
        oracle_sim = F.cosine_similarity(e_tgt_same, e_oracle, dim=-1).item()

        scores_generic.append(gen_sim)
        scores_sameutt.append(same_sim)
        oracle_scores.append(oracle_sim)

        print(f"  [{i+1}/{len(pair_files)}] {spk_src}→{spk_tgt}: generic={gen_sim:.3f} same-utt={same_sim:.3f} oracle={oracle_sim:.3f}", flush=True)

    gen_arr = np.array(scores_generic)
    same_arr = np.array(scores_sameutt)
    oracle_arr = np.array(oracle_scores)

    print(f"\n=== 結果 ===")
    print(f"モデル generic ref:  {gen_arr.mean():.4f} ± {gen_arr.std():.4f}")
    print(f"モデル same-utt ref: {same_arr.mean():.4f} ± {same_arr.std():.4f}")
    print(f"Oracle same-utt ref: {oracle_arr.mean():.4f} ± {oracle_arr.std():.4f}")
    print(f"モデル/Oracle比:     {same_arr.mean()/oracle_arr.mean():.1%}")
    print(f"generic/same-utt比:  {gen_arr.mean()/same_arr.mean():.1%}")


if __name__ == "__main__":
    main()

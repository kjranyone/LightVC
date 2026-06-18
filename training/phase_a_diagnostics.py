import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import csv
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from converter import FlowConverter, ConverterConfig
from infer_flow import decode, encode, load_dac, load_flow_converter

VCTK_WAV = Path("../data/vctk_200")
DEVICE = torch.device("cuda")
SCALES = [1, 3, 5, 10, 20]

def split_speakers():
    index_path = Path("data/vctk_latents_200/index.tsv")
    speakers = set()
    with open(index_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            speakers.add(row["speaker_id"])
    spk_list = sorted(speakers)
    np.random.seed(42)
    heldout = set(np.random.choice(spk_list, 19, replace=False))
    return sorted(heldout)

def load_secs_model():
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )
    return model

@torch.no_grad()
def secs_score(secs_model, wav_44k):
    import librosa
    wav16k = librosa.resample(wav_44k.astype(np.float32), orig_sr=44100, target_sr=16000)
    t = torch.from_numpy(wav16k).float().unsqueeze(0).to(DEVICE)
    embed = secs_model.encode_batch(t).squeeze().cpu()
    return embed

def cos_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()

def main():
    print("=== Phase A: 現行モデル診断 ===\n")

    dac, device = load_dac()
    converter, config = load_flow_converter("checkpoints/phase_c_utte_1024/best.pt", device)
    converter.eval()
    secs_model = load_secs_model()

    heldout = split_speakers()
    print(f"Held-out speakers: {heldout[:5]}... ({len(heldout)})")

    pairs = []
    for spk in heldout:
        spk_dir = VCTK_WAV / spk
        if not spk_dir.exists():
            continue
        wavs = sorted(spk_dir.glob("*.wav"))
        if len(wavs) >= 2:
            pairs.append((spk, wavs[0], wavs[1]))
        if len(pairs) >= 8:
            break

    print(f"Eval pairs: {len(pairs)}")
    if not pairs:
        print("No held-out pairs found. Using samples.")
        pairs = [("sample", Path("samples/wav/vctk_source_m.wav"),
                  Path("samples/wav/vctk_ref_f.wav"))]

    src_embeds_ref = {}
    tgt_embeds_ref = {}

    print("\nComputing reference embeddings...")
    for spk, src_wav, ref_wav in pairs:
        src_w, sr = sf.read(str(src_wav), dtype="float32")
        if sr != 44100:
            import librosa
            src_w = librosa.resample(src_w, orig_sr=sr, target_sr=44100)
        ref_w, sr = sf.read(str(ref_wav), dtype="float32")
        if sr != 44100:
            import librosa
            ref_w = librosa.resample(ref_w, orig_sr=sr, target_sr=44100)
        src_embeds_ref[spk] = secs_score(secs_model, src_w)
        tgt_embeds_ref[spk] = secs_score(secs_model, ref_w)

    print("\n--- A1: velocity_scale sweep ---")
    print(f"{'scale':>6}  {'SECS(tgt)':>10}  {'SECS(src)':>10}  {'leakage':>10}")
    print("-" * 52)

    all_v_preds = []
    all_v_shifts = []

    for scale in SCALES:
        tgt_sims = []
        src_sims = []

        for spk, src_wav, ref_wav in pairs:
            src_w, sr = sf.read(str(src_wav), dtype="float32")
            if sr != 44100:
                import librosa
                src_w = librosa.resample(src_w, orig_sr=sr, target_sr=44100)
            ref_w, sr = sf.read(str(ref_wav), dtype="float32")
            if sr != 44100:
                import librosa
                ref_w = librosa.resample(ref_w, orig_sr=sr, target_sr=44100)

            rem = len(src_w) % 512
            if rem > 0:
                src_w = np.pad(src_w, (0, 512 - rem))
            rem = len(ref_w) % 512
            if rem > 0:
                ref_w = np.pad(ref_w, (0, 512 - rem))

            z_src = encode(dac, src_w, device).unsqueeze(0)
            z_ref = encode(dac, ref_w, device).unsqueeze(0)

            with torch.no_grad():
                t = torch.ones(1, device=device)
                v = converter.forward_velocity(z_src, t, z_ref)
                z_out = z_src + scale * v

                if scale == 1 and spk == pairs[0][0]:
                    spk_mean_src = z_src.mean(dim=-1)
                    all_v_preds.append(v.cpu())
                    v_shift = z_ref.mean(dim=-1) - spk_mean_src
                    all_v_shifts.append(v_shift.cpu())

            conv_wav = decode(dac, z_out.squeeze(0), device)

            conv_embed = secs_score(secs_model, conv_wav)
            tgt_sim = cos_sim(conv_embed, tgt_embeds_ref[spk])
            src_sim = cos_sim(conv_embed, src_embeds_ref[spk])
            tgt_sims.append(tgt_sim)
            src_sims.append(src_sim)

        mean_tgt = np.mean(tgt_sims)
        mean_src = np.mean(src_sims)
        leakage = mean_src - mean_tgt
        print(f"{scale:>6}  {mean_tgt:>10.4f}  {mean_src:>10.4f}  {leakage:>+10.4f}")

    print(f"\n--- A3: leakage summary ---")
    print(f"SECS(converted, target): {mean_tgt:.4f}")
    print(f"SECS(converted, source): {mean_src:.4f}")
    print(f"Leakage (src - tgt):     {leakage:+.4f}")
    print(f"→ {'Source speaker leaking' if mean_src > mean_tgt else 'No significant leakage'}")

    if all_v_preds and all_v_shifts:
        print(f"\n--- A4: v_pred direction analysis ---")
        v_pred = all_v_preds[0].squeeze(0).flatten()
        v_shift_1d = all_v_shifts[0].squeeze(0)
        T = all_v_preds[0].shape[-1]
        v_shift = v_shift_1d.unsqueeze(-1).expand(-1, T).flatten()
        cos = F.cosine_similarity(v_pred.unsqueeze(0), v_shift.unsqueeze(0), dim=-1).item()
        norm_pred = v_pred.norm().item()
        norm_shift = v_shift.norm().item()
        print(f"cos(v_pred, v_speaker_shift) = {cos:.6f}")
        print(f"||v_pred||          = {norm_pred:.4f}")
        print(f"||v_speaker_shift|| = {norm_shift:.4f}")
        print(f"magnitude ratio     = {norm_pred/max(norm_shift, 1e-8):.4f}")
        if cos > 0.1:
            print("→ v_pred has POSITIVE correlation with speaker shift direction")
            print("→ Direction is partially correct, amplitude is the issue")
        elif cos < -0.1:
            print("→ v_pred has NEGATIVE correlation — converting AWAY from target")
        else:
            print("→ v_pred is uncorrelated with speaker shift — direction is noise")

    print(f"\n=== Phase A Summary ===")
    best_scale = SCALES[np.argmax([mean_tgt])]
    print(f"Best velocity_scale: {best_scale} (SECS={max([mean_tgt]):.4f})")
    print(f"DP-A judgment: {'FM has directional signal → Phase B+C' if max([mean_tgt]) > 0.2 else 'No signal → Phase C priority'}")

if __name__ == "__main__":
    main()

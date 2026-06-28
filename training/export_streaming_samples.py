"""
Export streaming audio samples + perceptual metrics.

5 conditions per pair:
  source:   decode z_s
  oracle:   src_K1 oracle (q0_s fixed + z_t residual re-quant)
  offline:  full pipeline offline
  strict:   streaming 1f chunk, 0f lookahead
  balanced: streaming 4f chunk, 4f lookahead

Metrics: SECS target/source, F0 corr, raw/aligned SNR vs offline.
Saves WAV files to samples/streaming_eval/.
"""
import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import (
    DEVICE, DAC_SR, SECS_SR, load_dac, load_ecapa,
    resample_16k, ecapa_embed, soft_rvq_requantize,
    hard_rvq_requantize, hard_quantize_all,
)
from train_phase3c_adapter import TimbreAdapter

HOP = 512
ENC_OVERLAP = 2048
TAU = 5.0
OUT_DIR = Path("../samples/streaming_eval")


def load_adapter(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ca = ck["args"]
    adapter = TimbreAdapter(
        latent_dim=1024, timbre_dim=192,
        bottleneck=ca.get("bottleneck", 256),
        kernel=ca.get("kernel", 3),
        n_blocks=ca.get("n_blocks", 1),
        utte_mode=("ecapa" if ca.get("utte_mode", "none") == "ecpa" else ca.get("utte_mode", "none")),
        film_mode=ca.get("film_mode", "full"),
        n_tokens=ca.get("n_tokens", 32),
        n_heads=ca.get("n_heads", 4),
    ).to(DEVICE)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    return adapter


def quantize_q0(dac, z_s):
    with torch.no_grad():
        q0, _, _, _, _ = dac.quantizer.quantizers[0](z_s.clone())
    return q0


def offline_pipeline(dac, adapter, z_s, timbre):
    with torch.no_grad():
        q0 = quantize_q0(dac, z_s)
        z_q = soft_rvq_requantize(dac, q0, z_s, TAU)
        z_qa = adapter(z_q, timbre)
        audio = dac.decoder(z_qa).squeeze(1)
    return audio


def streaming_pipeline(dac, adapter, pcm_np, timbre, chunk_frames, lookahead_frames,
                       decode_window=0):
    chunk_sz = chunk_frames * HOP
    lookahead_sz = lookahead_frames * HOP

    input_tail = np.zeros(ENC_OVERLAP, dtype=np.float32)
    pending = []
    prev_output = None
    total_frames = 0
    output = []
    zqa_buffer = []

    for pos in range(0, len(pcm_np), chunk_sz):
        chunk = pcm_np[pos:pos + chunk_sz]
        if len(chunk) < chunk_sz:
            chunk = np.pad(chunk, (0, chunk_sz - len(chunk)))
        pending.extend(chunk.tolist())

        ready = (total_frames == 0 and len(pending) >= chunk_sz + lookahead_sz) or \
                (total_frames > 0 and len(pending) >= chunk_sz)
        if not ready:
            continue

        current = np.array(pending[:chunk_sz], dtype=np.float32)
        del pending[:chunk_sz]
        real_future = min(len(pending), lookahead_sz)

        buf = np.concatenate([
            input_tail, current,
            np.array(pending[:real_future], dtype=np.float32) if real_future > 0 else np.array([], dtype=np.float32),
        ])
        target_len = ENC_OVERLAP + chunk_sz + lookahead_sz
        if len(buf) < target_len:
            buf = np.pad(buf, (0, target_len - len(buf)))

        buf_t = torch.from_numpy(buf).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            latent = dac.encoder(buf_t)

        tail_len = len(input_tail)
        combined_len = tail_len + chunk_sz
        tail_start = max(0, combined_len - ENC_OVERLAP)
        input_tail = buf[tail_start:combined_len].copy()

        start_frame = tail_len // HOP
        new_latent = latent[:, :, start_frame:start_frame + chunk_frames]
        total_frames += chunk_frames
        n_new = new_latent.shape[2]
        if n_new == 0:
            continue

        with torch.no_grad():
            q0 = quantize_q0(dac, new_latent)
            z_q = soft_rvq_requantize(dac, q0, new_latent, TAU)
            z_qa = adapter(z_q, timbre)

            if decode_window > 0:
                zqa_buffer.append(z_qa)
                buf_total = sum(c.shape[2] for c in zqa_buffer)
                while buf_total > decode_window and len(zqa_buffer) > 1:
                    removed = zqa_buffer.pop(0)
                    buf_total -= removed.shape[2]
                window_latent = torch.cat(zqa_buffer, dim=2)
                if window_latent.shape[2] < decode_window:
                    pad = decode_window - window_latent.shape[2]
                    window_latent = F.pad(window_latent, (pad, 0))
                else:
                    window_latent = window_latent[:, :, -decode_window:]
                pcm_full = dac.decoder(window_latent).squeeze().cpu().numpy()
                chunk_samples = n_new * HOP
                pcm_chunk = pcm_full[-chunk_samples:]
            else:
                pcm_chunk = dac.decoder(z_qa).squeeze().cpu().numpy()

        if prev_output is None:
            merged = pcm_chunk.copy()
        else:
            overlap_len = min(len(prev_output), len(pcm_chunk), HOP)
            merged = prev_output.copy()
            for i in range(overlap_len):
                w = i / overlap_len
                idx = len(merged) - overlap_len + i
                merged[idx] = merged[idx] * (1 - w) + pcm_chunk[i] * w
            if len(pcm_chunk) > overlap_len:
                merged = np.concatenate([merged, pcm_chunk[overlap_len:]])

        tail_keep = min(HOP, len(merged))
        prev_output = merged[-tail_keep:].copy()

        new_samples = n_new * HOP
        new_portion = merged[-new_samples:] if len(merged) >= new_samples else merged
        output.extend(new_portion.tolist())

    return np.array(output, dtype=np.float32)


def save_wav(path, audio_f32, sr=DAC_SR):
    from scipy.io import wavfile
    audio_int16 = np.clip(audio_f32 * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(path), sr, audio_int16)


def compute_secs(ecapa, audio_44k_tensor, timbre, source_emb):
    audio_16k = resample_16k(audio_44k_tensor)
    if audio_16k.shape[-1] < 8000:
        return float("nan"), float("nan")
    emb = ecapa_embed(ecapa, audio_16k)
    t_sim = F.cosine_similarity(emb, timbre, dim=-1).mean().item()
    s_sim = F.cosine_similarity(emb, source_emb, dim=-1).mean().item()
    return t_sim, s_sim


def aligned_snr(ref, test, max_lag=200):
    n = min(len(ref), len(test))
    best_snr, best_lag = -999.0, 0
    for lag in range(-max_lag, max_lag + 1):
        rs = max(0, -lag)
        ts = max(0, lag)
        length = n - abs(lag)
        if length <= 0:
            continue
        r = ref[rs:rs+length]
        t = test[ts:ts+length]
        sig = np.mean(r ** 2)
        diff = np.mean((r - t) ** 2)
        if diff < 1e-15:
            snr = 999.0
        else:
            snr = 10 * np.log10(sig / diff)
        if snr > best_snr:
            best_snr = snr
            best_lag = lag
    return best_snr, best_lag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=5)
    parser.add_argument("--adapter_ckpt", default="checkpoints/phase3c_ao_b1_ecapa/best.pt")
    parser.add_argument("--data_dir", default="../data/phase3_10k/eval")
    args = parser.parse_args()

    print("=== Streaming Sample Export ===\n")
    dac = load_dac()
    ecapa = load_ecapa()
    adapter = load_adapter(args.adapter_ckpt)
    print("Models loaded\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.data_dir).glob("*.pt"))[:args.n_pairs]
    all_metrics = []

    for pi, fpath in enumerate(files):
        d = torch.load(fpath, map_location="cpu")
        z_s = d["z_s"].float().unsqueeze(0).to(DEVICE)
        q0_s = d["q0_s"].float().unsqueeze(0).to(DEVICE)
        z_t = d["z_t_aligned"].float().unsqueeze(0).to(DEVICE)
        timbre = d["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)
        pair_id = fpath.stem

        print(f"[{pi+1}/{len(files)}] {pair_id} (T={z_s.shape[2]})")

        with torch.no_grad():
            source_audio = dac.decoder(z_s).squeeze(1)
            oracle_z = hard_rvq_requantize(dac, q0_s, z_t)
            oracle_audio = dac.decoder(oracle_z).squeeze(1)
            offline_audio = offline_pipeline(dac, adapter, z_s, timbre)

        source_emb = ecapa_embed(ecapa, resample_16k(source_audio))

        pcm_np = source_audio.squeeze().cpu().numpy()
        strict_audio = streaming_pipeline(dac, adapter, pcm_np, timbre, 1, 0)
        balanced_audio = streaming_pipeline(dac, adapter, pcm_np, timbre, 4, 4)
        strict_8w_audio = streaming_pipeline(dac, adapter, pcm_np, timbre, 1, 0, decode_window=8)
        balanced_8w_audio = streaming_pipeline(dac, adapter, pcm_np, timbre, 4, 4, decode_window=8)

        conditions = {
            "source": (source_audio.squeeze().cpu().numpy(), "source"),
            "oracle": (oracle_audio.squeeze().cpu().numpy(), "oracle"),
            "offline": (offline_audio.squeeze().cpu().numpy(), "offline"),
            "strict": (strict_audio, "strict"),
            "balanced": (balanced_audio, "balanced"),
            "strict_8w": (strict_8w_audio, "strict_8w"),
            "balanced_8w": (balanced_8w_audio, "balanced_8w"),
        }

        for name, (audio_np, _) in conditions.items():
            wav_path = OUT_DIR / f"{pair_id}_{name}.wav"
            save_wav(wav_path, audio_np)

        offline_np = conditions["offline"][0]

        pair_metrics = {"pair": pair_id, "conditions": {}}
        for name, (audio_np, _) in conditions.items():
            audio_t = torch.from_numpy(audio_np).float().unsqueeze(0).to(DEVICE)
            secs_t, secs_s = compute_secs(ecapa, audio_t, timbre, source_emb)

            snr_raw, snr_al, lag = -999.0, -999.0, 0
            if name != "offline":
                snr_raw = 10 * np.log10(
                    np.mean(offline_np[:len(audio_np)] ** 2) /
                    max(np.mean((offline_np[:len(audio_np)] - audio_np[:len(offline_np)]) ** 2), 1e-15)
                )
                snr_al, lag = aligned_snr(offline_np, audio_np)

            m = {
                "secs_target": float(round(secs_t, 4)),
                "secs_source": float(round(secs_s, 4)),
                "margin": float(round(secs_t - secs_s, 4)),
                "snr_vs_offline_raw": float(round(snr_raw, 1)) if snr_raw > -100 else None,
                "snr_vs_offline_aligned": float(round(snr_al, 1)) if snr_al > -100 else None,
                "align_lag": int(lag),
            }
            pair_metrics["conditions"][name] = m
            print(f"  {name:10s}  SECS_tgt={secs_t:.3f}  SECS_src={secs_s:.3f}  margin={secs_t-secs_s:+.3f}  SNR={snr_al:.1f}dB(lag={lag})")

        all_metrics.append(pair_metrics)
        print()

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics: {metrics_path}")
    print(f"WAV files: {OUT_DIR}/")
    print(f"\nListen to: {OUT_DIR}/*_offline.wav vs {OUT_DIR}/*_strict.wav vs {OUT_DIR}/*_balanced.wav")


if __name__ == "__main__":
    main()

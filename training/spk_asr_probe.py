"""s7サンプルの他軸交絡測定: ECAPA話者類似度(cos)とWhisper CER(内容保持)。

対象: results/s5_render/namikawa_s7_st{0,17}.wav(フルレンダ)と
results/diag_cfm_audit/s7_cfm_itp_*_{base,st12}.wav(学習規約内)。
対照: codec天井=GT latentデコード、ASRノーズ=GT音声のwhisper vs .lab、
話者天井=GT音声vs同一話者参照、クロス話者floor。

    CUDA_VISIBLE_DEVICES=0 uv run python spk_asr_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
import whisper
from speechbrain.inference.speaker import EncoderClassifier

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec
from diag_cfm_audit import LAT, ROOT

OUT = ROOT / "results/diag_cfm_audit"
AUDIT_SPK = "fe8c22d913075909"
AUDIT_STEM = "fe8c22d913075909_00046177"
TARGET_SPK = "ab97e212acbb6d6b"
CHUNK = 20 * 16000


def ecapa():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa", run_opts={"device": "cuda"})


def embed(m, path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=16000, mono=True)
    es = []
    for i in range(0, len(y), CHUNK):
        seg = y[i:i + CHUNK]
        if len(seg) < 1600:
            continue
        with torch.no_grad():
            e = m.encode_batch(
                torch.from_numpy(seg).float().unsqueeze(0).cuda()
            ).squeeze().cpu().numpy()
        es.append(e)
    v = np.mean(es, 0)
    return v / (np.linalg.norm(v) + 1e-6)


def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[n] / m


def main() -> int:
    dev = "cuda"
    m = ecapa()

    def spk_ref(spk: str, k: int = 8) -> np.ndarray:
        wavs = sorted((ROOT / "female-dataset" / spk).glob("*.wav"))[:k]
        return np.mean([embed(m, str(w)) for w in wavs], 0)

    tgt_ref = spk_ref(TARGET_SPK)
    aud_ref = spk_ref(AUDIT_SPK)
    gt_wav = ROOT / "female-dataset" / AUDIT_SPK / f"{AUDIT_STEM}.wav"

    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()
    z1 = torch.load(LAT / "female_real" / AUDIT_SPK / f"{AUDIT_STEM}.pt",
                    map_location="cpu", weights_only=False)["z"].float()
    with torch.no_grad():
        stream = codec.decoder.stream()
        y = torch.cat([stream.decode_step(z1.transpose(0, 1)[None].to(dev)
                                           [:, :, i:i + 1])
                       for i in range(z1.shape[0])], -1)[0, 0].cpu().numpy()
    gtdec = OUT / f"{AUDIT_SPK}_gtdecode.wav"
    soundfile.write(gtdec, np.clip(y, -1, 1), 48000)

    src_nmk = str(ROOT / "namikawa.mp3")
    renders = {
        "namikawa_src": src_nmk,
        "namikawa_s7_st0": str(ROOT / "results/s5_render/namikawa_s7_st0.wav"),
        "namikawa_s7_st17": str(ROOT / "results/s5_render/namikawa_s7_st17.wav"),
        f"{AUDIT_SPK}_gt": str(gt_wav),
        f"{AUDIT_SPK}_gtdecode": str(gtdec),
        "s7_base": str(OUT / f"s7_cfm_itp_{AUDIT_STEM}_base.wav"),
        "s7_st12": str(OUT / f"s7_cfm_itp_{AUDIT_STEM}_st12.wav"),
    }
    emb = {k: embed(m, p) for k, p in renders.items()}

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b)

    spk = {
        "ceiling_gt_vs_own_ref": cos(emb[f"{AUDIT_SPK}_gt"], aud_ref),
        "cross_floor_gt_vs_target_ref": cos(emb[f"{AUDIT_SPK}_gt"], tgt_ref),
        "floor_src_vs_target_ref": cos(emb["namikawa_src"], tgt_ref),
        "namikawa_st0_vs_target": cos(emb["namikawa_s7_st0"], tgt_ref),
        "namikawa_st17_vs_target": cos(emb["namikawa_s7_st17"], tgt_ref),
        "namikawa_st0_vs_src": cos(emb["namikawa_s7_st0"], emb["namikawa_src"]),
        "namikawa_st17_vs_src": cos(emb["namikawa_s7_st17"], emb["namikawa_src"]),
        "gtdecode_vs_own_ref": cos(emb[f"{AUDIT_SPK}_gtdecode"], aud_ref),
        "s7_base_vs_own_ref": cos(emb["s7_base"], aud_ref),
        "s7_st12_vs_own_ref": cos(emb["s7_st12"], aud_ref),
        "s7_base_vs_target": cos(emb["s7_base"], tgt_ref),
    }

    wm = whisper.load_model("base").cuda()

    def tx(p: str) -> str:
        return wm.transcribe(p, language="ja", fp16=False)["text"]

    lab = (ROOT / "female-dataset" / AUDIT_SPK / f"{AUDIT_STEM}.lab")
    lab_txt = lab.read_text().strip() if lab.exists() else ""
    texts = {k: tx(p) for k, p in renders.items()}
    asr = {
        "audit_lab_text": lab_txt,
        "asr_noise_gt_vs_lab": cer(lab_txt, texts[f"{AUDIT_SPK}_gt"]) if lab_txt else None,
        "codec_ceiling_gtdec_vs_gt": cer(texts[f"{AUDIT_SPK}_gt"], texts[f"{AUDIT_SPK}_gtdecode"]),
        "s7_base_vs_gt": cer(texts[f"{AUDIT_SPK}_gt"], texts["s7_base"]),
        "s7_st12_vs_gt": cer(texts[f"{AUDIT_SPK}_gt"], texts["s7_st12"]),
        "namikawa_st0_vs_src": cer(texts["namikawa_src"], texts["namikawa_s7_st0"]),
        "namikawa_st17_vs_src": cer(texts["namikawa_src"], texts["namikawa_s7_st17"]),
        "transcripts": texts,
    }

    rep = {"speaker_cos": spk, "asr_cer": asr}
    (OUT / "spk_asr_s7.json").write_text(json.dumps(rep, indent=2,
                                                    ensure_ascii=False))
    print(json.dumps({"speaker_cos": spk, "asr_cer": {k: v for k, v in asr.items()
                                                      if k != "transcripts"}},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

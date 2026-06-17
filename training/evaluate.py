"""
Offline evaluation: SECS / UTMOS / WER for a trained FlowConverter.

Implements MODEL_TRAINING.md "Validation Protocol" (lines 380-403) and
plan/04_training_pipeline.md [04-5].

Each metric is isolated so that a missing model or transient error does not
abort the whole run; the metric is reported as None and a warning is printed.

Usage:
    uv run python evaluate.py \
        --converter checkpoints/phase_c/best.pt \
        --manifest eval_manifest.json \
        --output eval_results.json

Manifest format (JSON):
    {
      "pairs": [
        {"source": "path/src.wav", "reference": "path/ref.wav", "text": "hello world"},
        ...
      ]
    }

The optional "text" field enables content-preservation WER. When omitted,
WER is computed between Whisper(src) and Whisper(converted) as a proxy.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from converter import ConverterConfig, FlowConverter
from infer_flow import decode, encode, load_dac, load_flow_converter

SAMPLE_RATE = 44100


def load_wav(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    import librosa

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    rem = len(wav) % 512
    if rem > 0:
        wav = np.pad(wav, (0, 512 - rem))
    return wav.astype(np.float32)


class Metric:
    name = "base"

    def __init__(self, device: str):
        self.device = device
        self._failed = False

    def _fail(self, msg: str):
        if not self._failed:
            print(f"[{self.name}] disabled: {msg}", flush=True)
            self._failed = True


class SecsMetric(Metric):
    """Speaker Encoder Cosine Similarity via ECAPA-TDNN (speechbrain)."""

    name = "SECS"

    def __init__(self, device: str):
        super().__init__(device)
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="hf_models/spkrec-ecapa",
                run_opts={"device": device},
            )
        except Exception as e:
            self.model = None
            self._fail(f"speechbrain load failed: {e}")

    @torch.no_grad()
    def embed(self, wav_44k: np.ndarray) -> Optional[torch.Tensor]:
        if self.model is None:
            return None
        wav16k = _resample(wav_44k, SAMPLE_RATE, 16000)
        t = torch.from_numpy(wav16k).float().unsqueeze(0).to(self.device)
        return self.model.encode_batch(t).squeeze().cpu()

    def score(self, converted: np.ndarray, reference: np.ndarray) -> Optional[float]:
        if self.model is None:
            return None
        e_c = self.embed(converted)
        e_r = self.embed(reference)
        if e_c is None or e_r is None:
            return None
        return float(torch.nn.functional.cosine_similarity(e_c, e_r, dim=0))


class UtmosMetric(Metric):
    """UTMOS naturalness predictor (1-5 scale)."""

    name = "UTMOS"

    def __init__(self, device: str):
        super().__init__(device)
        self.model = None
        try:
            self.model = self._load(device)
        except Exception as e:
            self._fail(f"UTMOS load failed: {e}")

    def _load(self, device: str):
        import torch.nn as nn

        class _UtmosWrapper(nn.Module):
            def __init__(self, inner, dev):
                super().__init__()
                self.inner = inner.to(dev).eval()
                self.dev = dev

            @torch.no_grad()
            def forward(self, wav_44k: np.ndarray) -> float:
                wav16k = _resample(wav_44k, SAMPLE_RATE, 16000)
                t = torch.from_numpy(wav16k).float().unsqueeze(0).to(self.dev)
                return float(self.inner(t, sr=16000).item())

        from transformers import AutoModel

        inner = AutoModel.from_pretrained("sarulab-speech/utmos-strong")
        return _UtmosWrapper(inner, device)

    @torch.no_grad()
    def score(self, converted: np.ndarray) -> Optional[float]:
        if self.model is None:
            return None
        try:
            return self.model(converted)
        except Exception as e:
            self._fail(f"UTMOS inference failed: {e}")
            return None


class WerMetric(Metric):
    """Word Error Rate via Whisper ASR + jiwer."""

    name = "WER"

    def __init__(self, device: str):
        super().__init__(device)
        self.processor = None
        self.model = None
        try:
            from transformers import (
                AutoProcessor,
                WhisperForConditionalGeneration,
            )

            mid = "openai/whisper-large-v3"
            self.processor = AutoProcessor.from_pretrained(mid)
            self.model = (
                WhisperForConditionalGeneration.from_pretrained(mid).to(device).eval()
            )
            forced = self.processor.get_decoder_prompt_ids(
                language="en", task="transcribe"
            )
            self._forced = forced
        except Exception as e:
            self._fail(f"Whisper load failed: {e}")

    @torch.no_grad()
    def transcribe(self, wav_44k: np.ndarray) -> Optional[str]:
        if self.model is None:
            return None
        wav16k = _resample(wav_44k, SAMPLE_RATE, 16000)
        inputs = self.processor(
            wav16k, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)
        ids = self.model.generate(inputs, max_new_tokens=440, prompt_ids=self._forced)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def wer(self, ref: str, hyp: str) -> Optional[float]:
        try:
            import jiwer

            return float(jiwer.wer(ref.lower(), hyp.lower()))
        except Exception as e:
            self._fail(f"jiwer failed: {e}")
            return None


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)


def convert_sample(
    converter: FlowConverter,
    dac,
    device: str,
    src_wav: np.ndarray,
    ref_wav: np.ndarray,
) -> np.ndarray:
    if len(ref_wav) > 15 * SAMPLE_RATE:
        ref_wav = ref_wav[: 15 * SAMPLE_RATE]
    z_src = encode(dac, src_wav, device)
    z_ref = encode(dac, ref_wav, device)
    z_out = converter.convert(z_src, z_ref)
    return decode(dac, z_out, device)


def evaluate(args):
    dac, device = load_dac(args.dac_model)
    converter, config = load_flow_converter(args.converter, device)

    with open(args.manifest) as f:
        manifest = json.load(f)
    pairs = manifest["pairs"]
    print(f"Evaluating {len(pairs)} pairs on device={device}", flush=True)

    secs = SecsMetric(device)
    utmos = UtmosMetric(device)
    wer = WerMetric(device)

    secs_scores: list[float] = []
    utmos_scores: list[float] = []
    wer_src_vs_conv: list[float] = []
    wer_text_vs_conv: list[float] = []
    per_pair: list[dict] = []

    for i, pair in enumerate(pairs):
        src_path = pair["source"]
        ref_path = pair["reference"]
        gt_text = pair.get("text")
        if not os.path.isfile(src_path) or not os.path.isfile(ref_path):
            print(f"[{i}] skip (missing file): {src_path} / {ref_path}", flush=True)
            continue

        src_wav = load_wav(src_path)
        ref_wav = load_wav(ref_path)
        try:
            conv_wav = convert_sample(converter, dac, device, src_wav, ref_wav)
        except Exception as e:
            print(f"[{i}] conversion failed: {e}", flush=True)
            traceback.print_exc()
            continue

        s = secs.score(conv_wav, ref_wav)
        u = utmos.score(conv_wav)
        hyp_conv = wer.transcribe(conv_wav)

        entry = {"source": src_path, "reference": ref_path}
        if s is not None:
            secs_scores.append(s)
            entry["secs"] = s
        if u is not None:
            utmos_scores.append(u)
            entry["utmos"] = u
        if hyp_conv is not None:
            entry["whisper_converted"] = hyp_conv
            hyp_src = wer.transcribe(src_wav)
            if hyp_src is not None:
                entry["whisper_source"] = hyp_src
                w = wer.wer(hyp_src, hyp_conv)
                if w is not None:
                    wer_src_vs_conv.append(w)
                    entry["wer_src_vs_converted"] = w
            if gt_text is not None:
                entry["ground_truth"] = gt_text
                w = wer.wer(gt_text, hyp_conv)
                if w is not None:
                    wer_text_vs_conv.append(w)
                    entry["wer_text_vs_converted"] = w

        print(
            f"[{i + 1}/{len(pairs)}] {os.path.basename(src_path)} "
            f"SECS={s if s is None else f'{s:.4f}'} "
            f"UTMOS={u if u is None else f'{u:.3f}'}",
            flush=True,
        )
        per_pair.append(entry)

    summary = {
        "n_pairs": len(pairs),
        "n_evaluated": len(per_pair),
        "secs_mean": float(np.mean(secs_scores)) if secs_scores else None,
        "secs_n": len(secs_scores),
        "utmos_mean": float(np.mean(utmos_scores)) if utmos_scores else None,
        "utmos_n": len(utmos_scores),
        "wer_src_vs_converted_mean": (
            float(np.mean(wer_src_vs_conv)) if wer_src_vs_conv else None
        ),
        "wer_text_vs_converted_mean": (
            float(np.mean(wer_text_vs_conv)) if wer_text_vs_conv else None
        ),
        "targets": {"secs_gt": 0.70, "utmos_gt": 3.5, "wer_lt": 0.05},
        "pairs": per_pair,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===", flush=True)
    sm = summary["secs_mean"]
    um = summary["utmos_mean"]
    print(
        f"SECS:  {sm:.4f}" if sm is not None else "SECS:  n/a",
        f"(target > 0.70)" if sm is not None else "",
        flush=True,
    )
    print(
        f"UTMOS: {um:.3f}" if um is not None else "UTMOS: n/a",
        f"(target > 3.5)" if um is not None else "",
        flush=True,
    )
    wc = summary["wer_src_vs_converted_mean"]
    print(
        f"WER (src vs converted): {wc:.4f}" if wc is not None else "WER: n/a",
        f"(target degradation < 0.02)" if wc is not None else "",
        flush=True,
    )
    print(f"\nResults saved: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FlowConverter: SECS / UTMOS / WER"
    )
    parser.add_argument("--converter", required=True, help="FlowConverter checkpoint")
    parser.add_argument(
        "--manifest", required=True, help="JSON manifest of evaluation pairs"
    )
    parser.add_argument("--output", required=True, help="Output JSON results path")
    parser.add_argument("--dac-model", default="descript/dac_44khz")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

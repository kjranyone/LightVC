"""Rust クライアント出力への R-X 判定 (SECS + CER)。判定対象=製品そのもの。

    uv run python eval_rust_x.py --dir <rust 変換 wav のディレクトリ> --spk <話者>
wav 名は eval_x --dump と同じ <spk>_<stem>.wav。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_vc_g import load_wav, wav_path_of
from eval_cer import cer

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--spk", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    wavs = sorted(d.glob("*.wav"))
    if not wavs:
        sys.exit("wav が無い")

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/sb_ecapa",
        run_opts={"device": "cpu"})
    import librosa
    import soundfile as sf_

    def emb_of(w44):
        w16 = librosa.resample(w44, orig_sr=44100, target_sr=16000)
        e = ecapa.encode_batch(torch.from_numpy(w16)[None]).squeeze()
        return e / e.norm()

    tf = sorted((ROOT / "data/female_tts_feat" / a.spk).glob("*.pt"))
    cen = []
    for f in tf[:20]:
        dd = torch.load(f, map_location="cpu", weights_only=False)
        cen.append(emb_of(load_wav(wav_path_of(dd)).numpy()))
    cen = torch.stack(cen).mean(0)
    cen = cen / cen.norm()

    import whisper
    model = whisper.load_model("base")

    def src_wav(stem: str) -> Path:
        parts = stem.split("_", 2)
        spk = parts[0] + "_" + parts[1]
        f = ROOT / "data/male_feat" / spk / f"{parts[2]}.pt"
        dd = torch.load(f, map_location="cpu", weights_only=False)
        return wav_path_of(dd)

    ss, cs = [], []
    lang = None
    for p in wavs:
        w, sr = sf_.read(str(p), dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        s = float((emb_of(w) * cen).sum())
        r = model.transcribe(str(src_wav(p.stem)), fp16=False, temperature=0.0)
        if lang is None:
            lang = r["language"]
        hyp = model.transcribe(str(p), language=lang, fp16=False,
                               temperature=0.0)["text"]
        c = cer(r["text"], hyp)
        ss.append(s)
        cs.append(c)
        print(f"  {p.stem}: SECS {s:.3f}  CER {c:.3f}", flush=True)
    print(f"\n  Rust 出力: SECS {sum(ss)/len(ss):.3f}（線 0.50）  "
          f"CER {sum(cs)/len(cs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

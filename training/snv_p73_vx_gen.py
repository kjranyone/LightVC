"""段 P73 step c(results/snv_p73/prereg.yaml): 別の生成器 VoxCPM2(Apache-2.0)の、説明文で作った男声。測りの入力。
t00 = 説明文を文の頭の括弧に置いて作る(実在の人の音声を参照にしない)。t01〜t04 = t00 を参照に読む。説明文と文 = data/snv_p73/vx/manifest.json(qw と同じ説明文・別の文)。出力 = data/snv_p73/vx/<声>/tNN.wav。
    ../voxcpm_env/bin/python training/snv_p73_vx_gen.py
"""
from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import torch
from voxcpm import VoxCPM

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_p73/vx"


def main() -> int:
    man = json.loads((OUT / "manifest.json").read_text())
    m = VoxCPM.from_pretrained(man["model"], load_denoiser=False)
    sr = m.tts_model.sample_rate
    for v in man["voices"]:
        d = OUT / v["voice"]
        d.mkdir(exist_ok=True)
        for j, f in enumerate(v["files"]):
            p = d / f"{f['t']}.wav"
            if p.exists():
                continue
            torch.manual_seed(v["seed"] + j)
            w = m.generate(text=f"({v['instruct']}){f['text']}", cfg_value=2.0, inference_timesteps=10) if j == 0 else \
                m.generate(text=f["text"], reference_wav_path=str(d / "t00.wav"), cfg_value=2.0, inference_timesteps=10)
            sf.write(p, w, sr)
        print(v["voice"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

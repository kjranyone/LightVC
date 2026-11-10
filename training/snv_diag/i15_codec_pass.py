"""段 I15 の起動の前の検査(paper_snv rev29・学習なし): 本物の声を Irodori の生成の経路の後段(Semantic-DACVAE の符号化と復号・SilentCipher の透かし)に通した音を作る。
前段 D が「Irodori の生成の痕跡 = 声」と学んだかを、通した音と元の音で D の外れを比べて測る(snv_diag/i15_pre_probe.py mode=codec)。
組 = VCTK の男声(data/snv_p_feat/vctk の 3 本おき)・JVNV(3 本おき)・試験の伴奏の無い 9 区間・検証の Irodori 16 本(対照)・jvnv_all = i10_jvnv_probe の源と登録(M1・M2 × 6 感情 × regular_01・02 = 24 本)・real_all = 試験の 12 区間と登録 2 本(i8_text_probe realdir= の材料)。
出力 = data/snv_i15_codec/{codec,codec_wm}/{組}/{名前}.wav(48 kHz)。
bg = 段 I14 の前段 D の開発の組の背景(results/snv_i14_sep/bg_split.json の dev の音楽 24・雑音 24・8 s・RMS −26 dBFS)を、そのまま(bg)と SilentCipher の透かしだけを足して(bg_wm)書く = 透かしだけで D が声と読むかの検査の材料。
    cd ../../Irodori-TTS && HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-sync python ../LightVC/training/snv_diag/i15_codec_pass.py [bg] [groups vctk,jvnv,real_nonbgm,irodori_val,jvnv_all,real_all]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[2]
IRODORI = ROOT.parent / "Irodori-TTS"
FEAT = ROOT / "data/snv_p_feat"
JV = ROOT / "data/jvnv/jvnv_v1"
OUT = ROOT / "data/snv_i15_codec"
SR = 48000
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def items() -> dict:
    return {"vctk": [(p.stem, p.with_suffix(".wav")) for p in sorted((FEAT / "vctk").glob("*.npz"))[::3]],
            "jvnv": [(p.stem, JV / p.stem.split("_")[0] / p.stem.split("_")[1] / "regular" / f"{p.stem}.wav") for p in sorted((FEAT / "jvnv").glob("*.npz"))[::3]],
            "real_nonbgm": [(n, FEAT / "real" / f"{n}.wav") for n in [f"uirou_{i}" for i in range(6)] + [f"namikawa_{i}" for i in (3, 4, 5)]],
            "irodori_val": [(f"{v}_t0{j}", ROOT / "data/snv_e1" / v / f"t0{j}.wav") for v in ("em594", "em595", "em596", "em597", "em598", "em599", "ef000", "ef016") for j in (0, 1)],
            "real_all": [(n, FEAT / "real" / f"{n}.wav") for n in [f"uirou_{i}" for i in range(6)] + [f"namikawa_{i}" for i in range(6)] + ["uirou_enroll", "namikawa_enroll"]],
            "jvnv_all": [(f"{s}_{e}_regular_0{j}", JV / s / e / "regular" / f"{s}_{e}_regular_0{j}.wav") for s in ("M1", "M2") for e in ("anger", "disgust", "fear", "happy", "sad", "surprise") for j in (1, 2)]}


def bg_items() -> list[tuple[str, str]]:
    import json
    b = json.loads((ROOT / "results/snv_i14_sep/bg_split.json").read_text())["dev"]
    return [(f"{k}_{Path(q).stem}", q) for k in ("music", "noise") for q in b[k][::max(1, len(b[k]) // 24)][:24]]


def main_bg(wm) -> int:
    for nm, q in bg_items():
        x, sr = sf.read(q, dtype="float32", always_2d=True)
        x = torch.from_numpy(x.mean(1))
        if sr != SR:
            x = torchaudio.functional.resample(x[None], sr, SR)[0]
        st = max(0, (x.shape[0] - 8 * SR) // 2)
        x = x[st: st + 8 * SR]
        x = x * (10 ** (-26 / 20) / (x.pow(2).mean().sqrt() + 1e-9))
        y = wm.encode_batch([torchaudio.functional.resample(x[None], SR, 44100).to(DEV)], sample_rate=44100)[0].reshape(-1).float().cpu()
        y = torchaudio.functional.resample(y[None], 44100, SR)[0][: x.shape[0]]
        for sub, z in (("bg", x), ("bg_wm", y)):
            d = OUT / sub
            d.mkdir(parents=True, exist_ok=True)
            sf.write(d / f"{nm}.wav", np.clip(z.numpy(), -1, 1).astype(np.float32), SR)
    print("bg", len(bg_items()), flush=True)
    return 0


def main() -> int:
    sys.path.insert(0, str(IRODORI))
    if "bg" in sys.argv[1:]:
        from irodori_tts.watermark import SilentCipherWatermarker
        return main_bg(SilentCipherWatermarker(device=DEV))
    from irodori_tts.codec import DACVAECodec
    from irodori_tts.watermark import SilentCipherWatermarker
    codec = DACVAECodec.load(repo_id="Aratako/Semantic-DACVAE-Japanese-32dim", device=DEV, dtype=torch.float32, deterministic_encode=True, deterministic_decode=True)
    wm = SilentCipherWatermarker(device=DEV)
    groups = sys.argv[sys.argv.index("groups") + 1].split(",") if "groups" in sys.argv else list(items())
    for g, its in items().items():
        if g not in groups:
            continue
        for nm, w in its:
            x, sr = sf.read(str(w), dtype="float32", always_2d=True)
            x = torch.from_numpy(x.mean(1))[None]
            lat = codec.encode_waveform(x, sr, normalize_db=None, ensure_max=True)
            y = codec.decode_latent(lat).reshape(-1).float().cpu()
            csr = int(codec.sample_rate)
            n = int(round(x.shape[-1] * csr / sr))
            y = y[:n] if y.shape[0] >= n else torch.nn.functional.pad(y, (0, n - y.shape[0]))
            ywm = wm.encode_batch([y[None].to(DEV)], sample_rate=csr)[0].reshape(-1).float().cpu()
            for sub, z in (("codec", y), ("codec_wm", ywm)):
                z48 = torchaudio.functional.resample(z[None], csr, SR)[0].numpy() if csr != SR else z.numpy()
                d = OUT / sub / g
                d.mkdir(parents=True, exist_ok=True)
                sf.write(d / f"{nm}.wav", np.clip(z48, -1, 1).astype(np.float32), SR)
        print(g, len(its), "codec sr", codec.sample_rate, "wm", wm.ready, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

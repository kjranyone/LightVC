"""段 P38: 擬似の操作者の版の波形(data/snv_p_wav_x16・snv_p38_wav.py)に、Irodori の生成と同じ SilentCipher の透かしを足した版を作る。
前段の重みの学習の族(train_snv_sep の i15〜)は、声(Irodori の生成)と背景の両方に同じ透かしを持たせて、透かしが声と背景を分けないようにしている(paper_snv §5 M21)。
WORLD の分析合成と録音の鎖を通した版は透かしが弱まる(検出の率 1.00 → 0.34)ので、そのまま族に入れると「透かしの強い側 = 背景」の手がかりになる。版にも足し直して揃える(入力側だけ)。
出力 = data/snv_p_wav_x16w/{e1,e1nv}/<声>/<発話>.wav(48 kHz・16 bit・元と同じ長さ)。check = 検出の率を元の版と並べて出す。
段 P52: src=data/snv_p_wav_x52 out=data/snv_p_wav_x52w で粗さの版にも同じ手順で足す。
    cd ../../Irodori-TTS && HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-sync python ../LightVC/training/snv_p38_wm.py [check] [src=data/snv_p_wav_x16] [out=data/snv_p_wav_x16w]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
SRC = ROOT / "data/snv_p_wav_x16"
OUT = ROOT / "data/snv_p_wav_x16w"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    sys.path.insert(0, str(IRODORI))
    from irodori_tts.watermark import SilentCipherWatermarker
    global SRC, OUT
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    SRC, OUT = ROOT / opt.get("src", "data/snv_p_wav_x16"), ROOT / opt.get("out", "data/snv_p_wav_x16w")
    wm = SilentCipherWatermarker(device=DEV)
    assert wm.ready, "SilentCipher が使えない"
    files = sorted(SRC.glob("*/*/*.wav"))
    if "check" in sys.argv[1:]:
        import librosa
        g = np.random.default_rng(0)
        pick = [files[i] for i in g.choice(len(files), 48, replace=False)]

        def det(p: Path) -> bool:
            y = librosa.load(str(p), sr=44100, mono=True)[0]
            return bool(wm.model.decode_wav(torch.from_numpy(y).to(wm.model.device), 44100, False).get("status"))
        a, b = [det(p) for p in pick], [det(OUT / p.relative_to(SRC)) for p in pick]
        print("check", len(pick), "検出の率 元の版", round(float(np.mean(a)), 3), "透かしを足した版", round(float(np.mean(b)), 3), flush=True)
        assert float(np.mean(b)) > 0.9, "透かしが検出されない"
        return 0
    n = 0
    for i, p in enumerate(files):
        q = OUT / p.relative_to(SRC)
        if q.exists():
            continue
        x, sr = sf.read(str(p), dtype="float32")
        y = wm.encode_batch([torch.from_numpy(x)[None].to(DEV)], sample_rate=sr)[0].reshape(-1).float().cpu().numpy()
        y = np.concatenate([y, np.zeros(max(0, len(x) - len(y)), np.float32)])[: len(x)]
        q.parent.mkdir(parents=True, exist_ok=True)
        tmp = q.with_suffix(".tmp.wav")
        sf.write(tmp, np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
        tmp.rename(q)
        n += 1
        if (i + 1) % 1000 == 0:
            print(i + 1, "/", len(files), flush=True)
    print("done", len(files), "新しく作った", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A2(diag): 描画の代役 = BigVGAN v2(44.1 kHz・128 帯・hop 512)の log-mel を書き出す。
対象 = E1 試作のコーパス全部・A0 の目標の登録(j0〜j2 の連結)・A1 の診断 D0 の 10 s(48 kHz の元から 44.1 kHz で切り直す)。
出力 = data/snv_e1_feat/mel/<名前>.npy(float16・[128, T])。
    cd training && uv run --no-sync python snv_a2_mel.py
"""
from __future__ import annotations

import glob
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_e1_feat/mel"
A0 = ROOT / "results/snv_a0"
SR = 44100
GROUPS = ((3, 4, 5), (8, 9, 10), (11, 12, 13))


def bigvgan_h():
    from bigvgan.env import AttrDict
    d = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--nvidia--bigvgan_v2_44khz_128band_512x/snapshots/*"))[0].rstrip("/") + "/"
    return AttrDict(json.load(open(d + "config.json"))), d


def logmel(y: np.ndarray) -> np.ndarray:
    from bigvgan.meldataset import get_mel_spectrogram
    h, _ = bigvgan_h()
    return get_mel_spectrogram(torch.from_numpy(y.astype(np.float32))[None], h)[0].numpy()


def d0_audio(voice: str, gi: int) -> np.ndarray:
    y = np.concatenate([librosa.load(str(A0 / f"wav/{voice}_{j}.wav"), sr=SR, mono=True)[0] for j in GROUPS[gi]])
    return y[: 10 * SR]


def job(args: tuple) -> str:
    name, kind, spec = args
    q = OUT / f"{name}.npy"
    if q.exists():
        return name
    torch.set_num_threads(1)
    if kind == "file":
        y = librosa.load(str(spec), sr=SR, mono=True)[0]
    elif kind == "enroll":
        y = np.concatenate([librosa.load(str(A0 / f"wav/{spec}_{j}.wav"), sr=SR, mono=True)[0] for j in (0, 1, 2)])
    else:
        y = d0_audio(*spec)
    q.parent.mkdir(parents=True, exist_ok=True)
    np.save(q, logmel(y).astype(np.float16))
    if kind != "file":
        import soundfile as sf
        sf.write(OUT.parent / f"wav44/{name.replace('/', '__')}.wav", y, SR)
    return name


def main() -> int:
    (OUT.parent / "wav44").mkdir(parents=True, exist_ok=True)
    js = [(f"e1/{p.parent.name}/{p.stem}", "file", p) for p in sorted((ROOT / "data/snv_e1").glob("e*/t*.wav"))]
    fv = sorted({it["voice"] for it in json.loads((A0 / "wav/items.json").read_text()) if it["sex"] == "F"})
    js += [(f"a0enroll/{v}", "enroll", v) for v in fv]
    js += [(f"a1diag/D0_{v}_{gi}", "d0", (v, gi)) for v in fv for gi in range(3)]
    with ProcessPoolExecutor(8) as ex:
        for i, n in enumerate(ex.map(job, js, chunksize=8)):
            if i % 1000 == 0:
                print(i, n, flush=True)
    print("done", len(js), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

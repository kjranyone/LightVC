"""A4 の評価の素材: N0r の女 24 声の音素の無い発声(自己の reamp の正解)と JNV の男 M1・M2(本物の男声の音素の無い発声・前後の無音を除く)を、
A2 と同じ BigVGAN の log-mel・44.1 kHz の wav・SPARC の特徴にする。N0r の男は snv_n1_prep.py で済み。
出力 = data/snv_e1_feat/mel/n0r/{voice}_{cat}_{k}.npy・mel/jnvm/{stem}.npy・wav44/n0r__*.wav・wav44/jnvm__*.wav・sparc/n0r/*.npz・sparc/jnvm/*.npz。
    cd training && uv run --no-sync python snv_a4_prep.py mel
    cd .. && ../sparc_env/bin/python training/snv_a4_prep.py sparc
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/snv_e1_feat"
N0R = ROOT / "results/snv_n0r/wav"
JNV = ROOT / "data/jnv/JNV"
SR = 44100


def jobs() -> list[tuple[str, Path]]:
    out = [(f"n0r/{it['voice']}_{it['cat']}_{it['k']}", N0R / f"{it['voice']}_{it['cat']}_{it['k']}.wav")
           for it in json.loads((N0R / "items.json").read_text()) if it["sex"] == "F"]
    out += [(f"jnvm/{p.stem}", p) for spk in ("M1", "M2") for p in sorted((JNV / spk).glob("*.wav"))]
    return out


def main() -> int:
    js = jobs()
    if sys.argv[1] == "mel":
        import librosa
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from snv_a2_mel import logmel
        import soundfile as sf
        for name, p in js:
            y = librosa.load(str(p), sr=SR, mono=True)[0]
            if name.startswith("jnvm/"):
                y = librosa.effects.trim(y, top_db=40)[0]
            (FEAT / f"mel/{name}.npy").parent.mkdir(parents=True, exist_ok=True)
            np.save(FEAT / f"mel/{name}.npy", logmel(y).astype(np.float16))
            sf.write(FEAT / f"wav44/{name.replace('/', '__')}.wav", y, SR)
    else:
        from sparc import load_model
        m = load_model("multi", device="cuda")
        for name, _ in js:
            q = FEAT / f"sparc/{name}.npz"
            if q.exists():
                continue
            q.parent.mkdir(parents=True, exist_ok=True)
            try:
                c = m.encode(str(FEAT / f"wav44/{name.replace('/', '__')}.wav"))
            except ValueError as e:
                print("skip", name, e, flush=True)
                continue
            n = min(len(c["ema"]), len(c["pitch"]), len(c["loudness"]), len(c["periodicity"]))
            np.savez(q, ema=c["ema"][:n].astype(np.float32), pitch=c["pitch"][:n, 0].astype(np.float32), loudness=c["loudness"][:n, 0].astype(np.float32),
                     periodicity=c["periodicity"][:n, 0].astype(np.float32), pitch_stats=np.asarray(c["pitch_stats"], dtype=np.float32))
    print("done", len(js), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

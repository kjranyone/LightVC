"""A4 の学習データの特徴: data/snv_e1nv(E1 の声の音素の無い発声)を A2 と同じ BigVGAN の log-mel と、SPARC の特徴(../sparc_env)にする。
出力 = data/snv_e1_feat/mel/e1nv/{voice}/{cat}_{k}.npy・sparc/e1nv/{voice}/{cat}_{k}.npz。
    cd training && uv run --no-sync python snv_e1nv_feats.py mel
    cd .. && ../sparc_env/bin/python training/snv_e1nv_feats.py sparc
第 2 引数 = 声の数の上限(生成の途中で書き終えた声だけを先に処理する)。
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/snv_e1nv"
FEAT = ROOT / "data/snv_e1_feat"


def mel_job(p: Path) -> str:
    import librosa
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from snv_a2_mel import SR, logmel
    q = FEAT / f"mel/e1nv/{p.parent.name}/{p.stem}.npy"
    if not q.exists():
        torch.set_num_threads(1)
        q.parent.mkdir(parents=True, exist_ok=True)
        np.save(q, logmel(librosa.load(str(p), sr=SR, mono=True)[0]).astype(np.float16))
    return p.stem


def main() -> int:
    ps = sorted(SRC.glob("e*/*.wav"))
    if len(sys.argv) > 2:
        keep = sorted({p.parent.name for p in ps})[: int(sys.argv[2])]
        ps = [p for p in ps if p.parent.name in keep]
    print("files", len(ps), flush=True)
    if sys.argv[1] == "mel":
        with ProcessPoolExecutor(8) as ex:
            for i, _ in enumerate(ex.map(mel_job, ps, chunksize=16)):
                if i % 1000 == 0:
                    print(i, flush=True)
    else:
        from sparc import load_model
        m = load_model("multi", device="cuda")
        for i, p in enumerate(ps):
            q = FEAT / f"sparc/e1nv/{p.parent.name}/{p.stem}.npz"
            if q.exists():
                continue
            q.parent.mkdir(parents=True, exist_ok=True)
            try:
                c = m.encode(str(p))
            except ValueError as e:
                print("skip", p, e, flush=True)
                continue
            n = min(len(c["ema"]), len(c["pitch"]), len(c["loudness"]), len(c["periodicity"]))
            np.savez(q, ema=c["ema"][:n].astype(np.float32), pitch=c["pitch"][:n, 0].astype(np.float32), loudness=c["loudness"][:n, 0].astype(np.float32),
                     periodicity=c["periodicity"][:n, 0].astype(np.float32), pitch_stats=np.asarray(c["pitch_stats"], dtype=np.float32))
            if i % 500 == 0:
                print(i, p, flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

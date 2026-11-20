"""段 P91(results/snv_p91/prereg.yaml): 符号が読む包絡を、固定の窓(40 ms のハン窓・左寄せ = 先読み 0)のケプストラムの包絡に替えた特徴を作る。
元の特徴(npz)の env だけを替え、f0・per・ap はそのまま写す。包絡 = |STFT|(PA.NFFT 点)→ 前段と同じ mel の帯 → 対数 → DCT 25。
出力 = data/snv_p_feat_st/{e1,e1nv,d,cv,a0}/…・data/snv_p_feat_st_x16/{e1,e1nv}/…(元と同じ並び)。
    cd training && uv run --no-sync python snv_p91_stenv.py [workers=16] [only=e1,e1nv,d,cv,a0,x16] [limit=0]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np

import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
FEAT0 = ROOT / "data/snv_p_feat"
OUT = {"": ROOT / "data/snv_p_feat_st", "x16": ROOT / "data/snv_p_feat_st_x16"}
WIN_MS = 40.0


def st_env(x: np.ndarray, n: int) -> np.ndarray:
    return P.fixed_env(x, n, WIN_MS)


def work(job: tuple) -> str:
    src, wav, dst = job
    dst = Path(dst)
    if dst.exists():
        return ""
    d = dict(np.load(src))
    x = librosa.load(wav, sr=P.SR, mono=True)[0]
    d["env"] = st_env(x, len(d["f0"]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v, np.float32) for k, v in d.items()})
    tmp.rename(dst)
    return ""


def jobs_of(kind: str) -> list[tuple]:
    out = []
    if kind == "x16":
        for p in sorted((ROOT / "data/snv_p_feat_x16").glob("e1*/*/*.npz")):
            rel = p.relative_to(ROOT / "data/snv_p_feat_x16")
            out.append((str(p), str(ROOT / "data/snv_p_wav_x16" / rel.with_suffix(".wav")), str(OUT["x16"] / rel)))
        return out
    for p in sorted((FEAT0 / kind).rglob("*.npz")):
        rel = p.relative_to(FEAT0)
        w = EV.wav_path(str(rel.with_suffix("")))
        if w is not None:
            out.append((str(p), str(w), str(OUT[""] / rel)))
    return out


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    nw, limit = int(opt.get("workers", 16)), int(opt.get("limit", 0))
    kinds = opt.get("only", "e1,e1nv,d,cv,a0,x16").split(",")
    for k in kinds:
        js = jobs_of(k)
        miss = [j for j in js if not Path(j[1]).exists()]
        js = [j for j in js if Path(j[1]).exists()]
        js = js[:: max(1, len(js) // limit)][:limit] if limit else js
        print(k, "作る", len(js), "音が無い", len(miss), flush=True)
        with ProcessPoolExecutor(nw) as ex:
            list(ex.map(work, js, chunksize=8))
    for nm, root in OUT.items():
        if root.exists():
            (root / "manifest.json").write_text(json.dumps({"stage": "P91", "what": f"env を固定の窓 {WIN_MS:g} ms(ハン・左寄せ)のケプストラムの包絡に替えた特徴(f0・per・ap は元のまま)", "files": sum(1 for _ in root.rglob("*.npz"))}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""D229(学習なし・物差しの検査・results/conv_p0/d229_prereg.yaml): 開発用の組の目標ごとに、声域をそろえた独立の中心を作る。
プール = diag_pool.load_pool と同じ並びの「参照(先頭から 20 秒以上)と評価の 3 本を除いた残り」。各クリップの pYIN の有声の中央値が参照 ref_TTT.wav の中央値から ±3 半音以内のものを近い順につなぎ先頭 8 秒。
合計 4 秒未満の目標は元の独立の中心(--ind)で埋めて候補の集合を保ち、比較から除く(meta の ok = False)。出力 = <s>/cen_reg/cen_TTT.wav・<s>/cen_reg/meta.json。
    uv run python d229_regcen.py --s <dev の組>
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
TOL, SEC, MIN_SEC = 3.0, 8.0, 4.0


def f0med(path: str) -> float:
    import librosa
    x, _ = librosa.load(path, sr=16000)
    x = x[: 16000 * 12]
    f, v, _ = librosa.pyin(x, fmin=90, fmax=900, sr=16000, frame_length=1024)
    f = f[v & np.isfinite(f)]
    return float(np.median(f)) if len(f) > 20 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--targets", default="", help="目標の一覧の json({targets: [...]}・空 = <s>/devset.json)")
    ap.add_argument("--ind", required=True, help="元の独立の中心のディレクトリ(cen_TTT.wav)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    S = Path(a.s)
    tg = json.loads(Path(a.targets or S / "devset.json").read_text())["targets"]
    od = S / "cen_reg"
    od.mkdir(exist_ok=True)
    plan = []
    for ti, spk in enumerate(tg):
        ws = sorted((ROOT / "female-dataset" / spk).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        rest = [w for w in ws if w not in ref]
        plan.append((ti, spk, S / "d37/_work" / f"ref_{ti:03d}.wav", rest[3:]))
    paths = sorted({str(p) for _, _, r, pool in plan for p in [r] + pool})
    with Pool(a.workers) as pool:
        med = dict(zip(paths, pool.map(f0med, paths, chunksize=2)))
    meta = []
    for ti, spk, r, cand in plan:
        fr = med[str(r)]
        d = [(abs(12 * np.log2(med[str(w)] / fr)), w) for w in cand if np.isfinite(med[str(w)])]
        sel = sorted((x for x in d if x[0] <= TOL), key=lambda z: (z[0], z[1].name))
        xs, tot, used = [], 0.0, []
        for dist, w in sel:
            if tot >= SEC:
                break
            x = load48(w); xs.append(x); tot += len(x) / 48000; used.append((w.name, round(float(dist), 2)))
        ok = tot >= MIN_SEC
        y = np.concatenate(xs)[: int(SEC * 48000)] if ok else load48(Path(a.ind) / f"cen_{ti:03d}.wav")
        sf.write(od / f"cen_{ti:03d}.wav", y.astype(np.float32), 48000)
        meta.append({"ti": ti, "spk": spk, "f0_ref": round(fr, 1), "pool": len(cand), "matched": len(sel), "used": used, "sec": round(tot, 2), "ok": ok})
    (od / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print("targets", len(meta), "ok", sum(m["ok"] for m in meta), "excluded", [m["ti"] for m in meta if not m["ok"]], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

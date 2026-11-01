"""diag_rr9c のデータの前処理(学習データだけ・推論の経路に入らない): クローン(data/realx/clone/*_CLN.wav)の音量を、手本のクリップ(items_clone.json の ref_stems)の
発声中の平均パワーに合わせて data/realx/cn/clone/ に 48 kHz で書く(TTS の出力の音量は手本と無関係 = c0 の水準のずれは D186 のガビの原因)。
発声中 = 20 ms 枠の dB が最大から 35 dB 以内。
    uv run python norm_realx_clone.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent


def active_db(x: np.ndarray) -> float:
    n = len(x) // 960
    e = 10 * np.log10((x[: n * 960].reshape(n, 960) ** 2).mean(1) + 1e-12)
    return float(10 * np.log10(np.mean(10 ** (e[e > e.max() - 35] / 10))))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(ROOT / "data/realx/items_clone.json"))
    meta = json.loads(Path(ap.parse_args().items).read_text())
    stem2wav = {r["stem"]: ROOT / r["wav"] for r in json.loads((ROOT / "data/rr_feat/real_fem/manifest.json").read_text())["items"]}
    src, od = ROOT / "data/realx/clone", ROOT / "data/realx/cn/clone"
    od.mkdir(parents=True, exist_ok=True)
    gains = []
    for m in meta:
        f = src / f"{m['clone']}.wav"
        o = od / f.name
        if not f.exists() or o.exists():
            continue
        x = load48(f).astype(np.float64)
        ref = float(np.mean([active_db(load48(ROOT / w if "ref_wavs" in m else stem2wav[w]).astype(np.float64)) for w in m.get("ref_wavs", m["ref_stems"])]))
        g = ref - active_db(x)
        y = x * 10 ** (g / 20)
        pk = np.abs(y).max()
        if pk > 0.99:
            y *= 0.99 / pk
        sf.write(o, y.astype(np.float32), 48000)
        gains.append(g)
    if gains:
        print("n", len(gains), "gain dB median", round(float(np.median(gains)), 2), "p10/p90", np.round(np.percentile(gains, [10, 90]), 2).tolist(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

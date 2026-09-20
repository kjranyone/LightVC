"""V2-2 前半: prosody-style 統計ベクトルの事前計算（発話統計ではなく話者×スタイル定数）。

tts_emotional_live(F200×25文×5caption)と tts_male_ja(M47×37文×5スタイル)から
話者×caption ごとの lf0 統計・Δlf0 分布・vuv 統計・energy 統計を抽出し、
cartridge 用の固定ベクトルテーブルとして保存。P の target prosody-style 条件に使う。

    CUDA_VISIBLE_DEVICES=0 uv run python v22_prosody_stats.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF

ROOT = Path(__file__).resolve().parent.parent
SR = 44100


def f0_of(path: Path) -> torch.Tensor:
    import soundfile as sf_
    w, sr = sf_.read(str(path), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SR:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    x = torch.from_numpy(w) * 32768.0
    f0, _ = SF.causal_f0(x)
    return f0


def stats_of(f0: torch.Tensor) -> dict | None:
    v = f0[f0 > 50]
    if v.numel() < 20:
        return None
    lv = torch.log2(v)
    lf0 = torch.where(f0 > 50, torch.log2(f0.clamp(min=50.0)), torch.zeros_like(f0))
    d = torch.diff(lf0)
    dv = d[(lf0 > 0)[1:] & (lf0 > 0)[:-1]]
    hop = 512
    en = f0.abs()
    return {
        "lf0_med": float(lv.median()),
        "lf0_p10": float(lv.quantile(0.10)),
        "lf0_p90": float(lv.quantile(0.90)),
        "dlf0_std": float(dv.std()) if dv.numel() > 2 else 0.0,
        "dlf0_p90": float(dv.abs().quantile(0.90)) if dv.numel() > 2 else 0.0,
        "voiced": float((f0 > 50).float().mean()),
        "n": v.numel(),
    }


def main() -> int:
    rows = list(csv.DictReader(open(ROOT / "data/kansei_vc/manifests/all_utterances.tsv"),
                               delimiter="\t"))
    groups = defaultdict(list)
    for r in rows:
        if r["source_type"] in ("tts_emotional_live", "tts_male_ja"):
            key = (r["source_type"], r["speaker_id"], r["caption_key"])
            wav = r.get("wav_path") or r["path"]
            wp = ROOT / str(wav).lstrip("./")
            if not wp.exists():
                wp = Path(str(wav))
            groups[key].append(wp)

    table = {}
    for (src, spk, cap), paths in sorted(groups.items()):
        acc = []
        for p in paths[:12]:
            if not p.exists():
                continue
            try:
                s = stats_of(f0_of(p))
            except Exception as e:
                print(f"  ERR {p}: {e}", flush=True)
                continue
            if s:
                acc.append(s)
        if not acc:
            continue
        agg = {k: float(np.median([a[k] for a in acc]))
               for k in ("lf0_med", "lf0_p10", "lf0_p90", "dlf0_std", "dlf0_p90", "voiced")}
        table[f"{src}|{spk}|{cap}"] = torch.tensor(
            [agg["lf0_med"], agg["lf0_p10"], agg["lf0_p90"],
             agg["dlf0_std"], agg["dlf0_p90"], agg["voiced"]], dtype=torch.float32)
    out = ROOT / "data/prosody_style_table.pt"
    torch.save(table, out)
    print(f"{len(table)} (source|spk|caption) rows -> {out}")
    ks = list(table)
    print("sample:", ks[0], [round(float(x), 3) for x in table[ks[0]]])
    print("sample:", [k for k in ks if k.startswith("tts_male")][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

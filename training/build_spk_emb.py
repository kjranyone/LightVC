"""話者をまたいだ出力部の追加学習(交差の標本の同一性の損失)用に、表のある話者ごとの ECAPA の埋め込みを前計算する。
参照 = build_tables_rvoc.py と同じ(台帳の f0 のある行・ファイル名順・20s 以上)。出力 = <out>(npz: spk → 192 次元・正規化済み)。
    uv run python build_spk_emb.py --tab_dir ../data/rvoc_tables_full --out ../data/rvoc_tables_full_ecapa.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/rvoc_f0hi/manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sec", type=float, default=20.0)
    ap.add_argument("--real_only", action="store_true", help="参照を実音声の行だけから(話者 ID が TTS と重なる・build_tables_rvoc.py --real_only と同じ)")
    a = ap.parse_args()
    import idloss as ID
    import nvoc as N
    from train_rvoc import read_span
    dev = "cuda"
    ec = ID.Ecapa(dev)
    spks = sorted(f.stem for f in Path(a.tab_dir).glob("*.npz"))
    rows = json.loads(MANIFEST.read_text())["rows"]
    by: dict = {}
    for r in rows:
        if r.get("f0") and (not a.real_only or r["src"].startswith("real")):
            by.setdefault(r["spk"], []).append((r["wav"], r["sr"], r["dur"]))
    out = {}
    for i, s in enumerate(spks):
        xs, tot = [], 0.0
        for wav, sr, dur in sorted(by.get(s, [])):
            if tot >= a.sec:
                break
            n = int(dur * N.SR) // N.HOP * N.HOP
            if n < 50 * N.HOP:
                continue
            xs.append(read_span(wav, sr, 0, n).astype(np.float32)); tot += n / N.SR
        if not xs:
            continue
        x = torch.from_numpy(np.concatenate(xs)[: int(a.sec * N.SR)])[None].to(dev)
        with torch.no_grad():
            out[s] = ec(x)[0].cpu().numpy().astype(np.float32)
        if i % 200 == 0:
            print(i, len(spks), flush=True)
    np.savez(a.out, **out)
    print("done", len(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

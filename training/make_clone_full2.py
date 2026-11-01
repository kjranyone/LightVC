"""残りの実女声の話者のクローンの組(diag_rr19L NOT_CAND_FLAT の次・データの規模の腕・オーナー判断 10-14 のクローンを学習の目標に使う範囲の中)。
prep_rr.py の real_fem(8 本以上かつ先頭 12 本で 40 秒以上)に入らなかった female-dataset の話者のうち、全長 20 秒以上・held_speakers でも評価プール(diag_pool.load_pool(150, 3))でもない全員。
手本 = その話者のクリップを名前順に 20 秒以上になるまで(ref_stems・ref_wavs)= 学習で R の記憶に置くもの・製品の登録と同じ量。文 = items_clone.json の項目の文(make_clone_full.py と同じ)。
出力 = data/realx/items_clone_full2.json(group = seen4・名前 g<話者番号>_p<item>_CLN)。gen_realx_clone.py --premade・norm_realx_clone.py・prep_rr.py --corpus real_fem2 で使う。
    uv run python make_clone_full2.py --K 8 --held 1
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    import diag_pool as P
    import train_rvoc as TR
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--held", type=int, default=1)
    ap.add_argument("--ref_sec", type=float, default=20.0)
    a = ap.parse_args()
    base = json.loads((ROOT / "data/realx/items_clone.json").read_text())
    old = {it["spk"] for it in json.loads((ROOT / "data/rr_feat/real_fem/manifest.json").read_text())["items"]}
    held = set(TR.held_speakers())
    pool = {t["spk"] for t in P.load_pool(150, 3, "cpu")}
    prev = set()
    for f in ("items_clone.json", "items_clone_multi.json", "items_clone_scale.json", "items_clone_full.json"):
        prev |= {m["target"] for m in json.loads((ROOT / "data/realx" / f).read_text())}
    spk = []
    for d in sorted((ROOT / "female-dataset").iterdir()):
        if not d.is_dir() or d.name in old or d.name in held or d.name in pool or d.name in prev:
            continue
        refs, tot = [], 0.0
        for w in sorted(d.glob("*.wav")):
            if tot >= a.ref_sec:
                break
            refs.append(w); tot += sf.info(str(w)).duration
        if tot < a.ref_sec:
            continue
        spk.append((d.name, refs, round(tot, 1)))
    random.Random(1917).shuffle(spk)
    rng = random.Random(1919)
    out = []
    for si, (A, refs, tot) in enumerate(spk):
        for k, src in enumerate(rng.sample(base, a.K)):
            out.append({"item": src["item"], "ti": src["ti"], "spk": src["spk"], "text": src["text"], "target": A, "ref_stems": [w.stem for w in refs],
                        "ref_wavs": [str(w.relative_to(ROOT)) for w in refs], "ref_sec": tot,
                        "clone": f"g{si:04d}_p{src['item']:04d}_CLN", "val": False, "held": k < a.held, "group": "seen4"})
    assert len({o["clone"] for o in out}) == len(out)
    tg = {o["target"] for o in out}
    assert not (tg & (old | held | pool | prev))
    (ROOT / "data/realx/items_clone_full2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("items", len(out), "speakers", len(spk), "train", sum(1 for o in out if not o["held"]), "held", sum(1 for o in out if o["held"]),
          "ref_wavs", sum(len(r) for _, r, _ in spk), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

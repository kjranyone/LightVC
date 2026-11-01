"""全話者のクローンの組(オーナー判断 10-14・diag_rr14c STEPS_UP の次): make_clone_scale.py と同じ話者の並び(Random(1913) の shuffle)の残り(seen2 の 500 人より後ろ)の全員に K 文(うち held 文を保留)。
items_clone_scale.json(seen2 / unseen2 = 評価の組)は変えない。文 = 他の realx 項目の文・手本 = items_clone.json のその話者の ref_stems。
出力 = data/realx/items_clone_full.json(group = seen3・名前 f<話者番号>_p<item>_CLN)。gen_realx_clone.py --premade で合成する。
    uv run python make_clone_full.py --start 500 --K 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=500)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--held", type=int, default=2)
    a = ap.parse_args()
    base = json.loads((ROOT / "data/realx/items_clone.json").read_text())
    used = {m["target"] for m in json.loads((ROOT / "data/realx/items_clone_multi.json").read_text())}
    tr = [m for m in base if not m["val"] and m["ref_sec"] >= 20.0 and m["target"] not in used]
    random.Random(1913).shuffle(tr)
    scale = json.loads((ROOT / "data/realx/items_clone_scale.json").read_text())
    s2 = {m["target"] for m in scale if m["group"] == "seen2"}
    assert {m["target"] for m in tr[: a.start]} == s2
    rng = random.Random(1915)
    out = []
    for si in range(a.start, len(tr)):
        A = tr[si]
        pool = [m for m in base if m["target"] != A["target"]]
        for k, src in enumerate(rng.sample(pool, a.K)):
            out.append({"item": src["item"], "ti": src["ti"], "spk": src["spk"], "text": src["text"], "target": A["target"], "ref_stems": A["ref_stems"], "ref_sec": A["ref_sec"],
                        "clone": f"f{si:04d}_p{src['item']:04d}_CLN", "val": False, "held": k < a.held, "group": "seen3"})
    assert len({o["clone"] for o in out}) == len(out)
    assert not ({o["target"] for o in out} & ({m["target"] for m in scale} | used))
    (ROOT / "data/realx/items_clone_full.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("items", len(out), "speakers", len(tr) - a.start, "train", sum(1 for o in out if not o["held"]), "held", sum(1 for o in out if o["held"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

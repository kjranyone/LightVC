"""diag_rr13c のデータの組み合わせ(オーナー判断 10-14 のクローン・規模の検査): diag_rr12c の既知 24 人に加えて、realx の学習側の別の話者 S 人にそれぞれ K 文(うち 2 文を保留)・
未知の話者(検証側で items_clone_multi.json の未知 12 人に含まれない全員)にそれぞれ U 文を割り当てる。文 = 他の realx 項目の文(男声 TTS の読みが入力)・手本 = items_clone.json のその話者の ref_stems。
出力 = data/realx/items_clone_scale.json(group = seen2 / unseen2)。gen_realx_clone.py --premade で合成する。
    uv run python make_clone_scale.py --S 400 --K 12 --U 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=400)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--held", type=int, default=2)
    ap.add_argument("--U", type=int, default=5)
    a = ap.parse_args()
    base = json.loads((ROOT / "data/realx/items_clone.json").read_text())
    used = {m["target"] for m in json.loads((ROOT / "data/realx/items_clone_multi.json").read_text())}
    rng = random.Random(1913)
    tr = [m for m in base if not m["val"] and m["ref_sec"] >= 20.0 and m["target"] not in used]
    va = [m for m in base if m["val"] and m["ref_sec"] >= 20.0 and m["target"] not in used]
    rng.shuffle(tr)
    out = []
    for grp, spk, nsent, nheld, pre in (("seen2", tr[: a.S], a.K, a.held, "n"), ("unseen2", va, a.U, a.U, "v")):
        for si, A in enumerate(spk):
            pool = [m for m in base if m["target"] != A["target"]]
            for k, src in enumerate(rng.sample(pool, nsent)):
                out.append({"item": src["item"], "ti": src["ti"], "spk": src["spk"], "text": src["text"], "target": A["target"], "ref_stems": A["ref_stems"], "ref_sec": A["ref_sec"],
                            "clone": f"{pre}{si:04d}_p{src['item']:04d}_CLN", "val": grp == "unseen2", "held": k < nheld, "group": grp})
    assert len({o["clone"] for o in out}) == len(out)
    (ROOT / "data/realx/items_clone_scale.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("items", len(out), "seen2 speakers", min(a.S, len(tr)), "seen2 train", sum(1 for o in out if o["group"] == "seen2" and not o["held"]),
          "seen2 held", sum(1 for o in out if o["group"] == "seen2" and o["held"]), "unseen2 speakers", len(va), "unseen2", sum(1 for o in out if o["group"] == "unseen2"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""diag_rr12c のデータの組み合わせ(オーナー判断 10-14 のクローン): 既知の話者(realx の学習側 24 人)にそれぞれ 50 文・未知の話者(検証側 12 人)にそれぞれ 10 文を割り当てる。
文 = 他の realx 項目の文(男声 TTS の読み p{item}_t{ti}_SRC が既にある = 入力に使う)。手本 = items_clone.json のその話者の ref_stems(記憶と同じ)。
既知の話者は 50 文のうち 10 文を保留(held)= 学習に使わず評価だけ。出力 = data/realx/items_clone_multi.json(gen_realx_clone.py --items で合成)。
    uv run python make_clone_multi.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    base = json.loads((ROOT / "data/realx/items_clone.json").read_text())
    rng = random.Random(1912)
    tr = [m for m in base if not m["val"] and m["ref_sec"] >= 20.0]
    va = [m for m in base if m["val"] and m["ref_sec"] >= 20.0]
    rng.shuffle(tr); rng.shuffle(va)
    seen, unseen = tr[:24], va[:12]
    out = []
    for grp, spk, nsent, nheld in (("seen", seen, 50, 10), ("unseen", unseen, 10, 10)):
        for si, A in enumerate(spk):
            pool = [m for m in base if m["target"] != A["target"]]
            for k, src in enumerate(rng.sample(pool, nsent)):
                out.append({"item": src["item"], "ti": src["ti"], "spk": src["spk"], "text": src["text"], "target": A["target"], "ref_stems": A["ref_stems"], "ref_sec": A["ref_sec"],
                            "clone": f"{grp[0]}{si:03d}_p{src['item']:04d}_CLN", "val": grp == "unseen", "held": k < nheld, "group": grp})
    (ROOT / "data/realx/items_clone_multi.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("items", len(out), "seen train", sum(1 for o in out if o["group"] == "seen" and not o["held"]), "seen held", sum(1 for o in out if o["group"] == "seen" and o["held"]),
          "unseen", sum(1 for o in out if o["group"] == "unseen"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

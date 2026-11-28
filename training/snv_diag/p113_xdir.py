"""段 P113 step 1: 視点の置き場 data/snv_p_feat_xbt1 = 発話ごとに確率 0.5 で x16 の版・0.5 で変換の版(data/snv_p_feat_bt1)の symlink。台帳は x16 の写し。
    python3 training/snv_diag/p113_xdir.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
X, B, O = ROOT / "data/snv_p_feat_x16", ROOT / "data/snv_p_feat_bt1", ROOT / "data/snv_p_feat_xbt1"
rng = random.Random(113)
n = {"x16": 0, "bt": 0}
for sub in ("e1", "e1nv"):
    for p in sorted((X / sub).glob("*/*.npz")):
        rel = p.relative_to(X)
        b = B / rel
        use = "bt" if b.exists() and rng.random() < 0.5 else "x16"
        dst = O / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(b if use == "bt" else p)
        n[use] += 1
(O / "manifest.json").write_text((X / "manifest.json").read_text())
(O / "mix.json").write_text(json.dumps({"stage": "P113", "seed": 113, "count": n}))
print(n)

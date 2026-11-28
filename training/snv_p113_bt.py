"""段 P113(results/snv_p113/prereg.yaml・学習なしの準備): 「自分の変換で作った別の声の版」を入力側の視点として作る。E1・E1-NV の学習と検証の声の各発話を、製品の K3G(門なし)で、別の学習の男声 B の声へ変換した特徴(包絡 25 行 + 周期性 4 行)。
高さは擬似の操作者の版(x16)と同じ比 r を掛ける(台帳 data/snv_p_feat_x16/manifest.json の r・学習の側の「源の高さ + dlf = 当てる先の高さ」の関係を保つ)。当てる先には使わない(入力側だけ)。
出力 = data/snv_p_feat_bt1/{e1,e1nv}/<声>/<発話>.npz(f0・env・per)と manifest.json(x16 の台帳の写し + 声ごとの相手 B)。
    cd training && SNV_ENVC=1 SNV_DEV=cuda uv run --no-sync python snv_p113_bt.py [limit=0]
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_p_feat_bt1"
SYS = "ens:snv_p24_c|snv_p30_c|snv_p91e_c"


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    limit = int(opt.get("limit", 0))
    sp = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    man = json.loads((ROOT / "data/snv_p_feat_x16/manifest.json").read_text())
    S = EV.Ens(SYS, EV.load_gen())
    rng = random.Random(113)
    males = [v for v in sp["train"] if v.startswith("em")]
    pool = rng.sample(males, 48)
    ld = lambda p: T.load_utt(p)
    refs = lambda v: [ld(T.FEAT / "e1" / v / f"t0{j}.npz") for j in (0, 1, 2)]
    st: dict = {}
    voices = sorted(sp["train"] + sp["val"])
    voices = voices[:: max(1, len(voices) // limit)][:limit] if limit else voices
    pair, n = {}, 0
    for a in voices:
        b = rng.choice([q for q in pool if q != a])
        pair[a] = b
        if b not in st:
            st[b] = S.structure("t" + b, refs(b))
        enr = refs(a)
        k = 12.0 * math.log2(man["voices"][a]["r"])
        for sub in ("e1", "e1nv"):
            for p in sorted((T.FEAT / sub / a).glob("*.npz")):
                dst = OUT / sub / a / p.name
                if dst.exists():
                    continue
                u = ld(p)
                _, yo = S.convert(u, enr, st[b], k, with_y=True)
                y = yo[0].detach().cpu()
                env = T.y_env(y[None, :25])[0].numpy().astype(np.float32)
                per = y[25:29].clamp(min=0.0).numpy().astype(np.float32)
                f0 = (u["f0"] * man["voices"][a]["r"]).astype(np.float32)
                assert env.shape[1] == len(u["f0"])
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".tmp.npz")
                np.savez(tmp, f0=f0, env=env, per=per)
                tmp.rename(dst)
                n += 1
        if len(pair) % 50 == 0:
            print(len(pair), "/", len(voices), "声・発話", n, flush=True)
    (OUT / "manifest.json").write_text(json.dumps(man | {"bt": {"stage": "P113", "sys": SYS, "pool": pool, "pair": pair, "what": "製品の K3G で別の学習の男声へ変換した特徴(入力側の視点・高さは x16 と同じ比)"}}, ensure_ascii=False, indent=1))
    print("done", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

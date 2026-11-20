"""段 P82(results/snv_p82/prereg.yaml): 学習の発話の「写し」の特徴を作る。写し = 発話の特徴(包絡・高さ・周期性)を出力部 rvoc3hi に通した音を、前段 v3 で読み直したもの。
k0 = 加工しない発話(data/snv_p_feat)の写し・k1 = 擬似の操作者の版(data/snv_p_feat_x16)の写し。出力 = data/snv_p_feat_cp/{e1,e1nv}/<声>/<名>_k{0,1}.npz(前段の解析の形)。
符号の学習の入力側の視点にだけ使う(当てる先にしない)。
    cd training && SNV_DEV=cuda uv run --no-sync python snv_p82_copyfeat.py [workers=14] [limit=0]
"""
from __future__ import annotations

import json
import sys
import types
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = EV.ROOT
OUT = ROOT / "data/snv_p_feat_cp"
SRC = (T.FEAT, ROOT / "data/snv_p_feat_x16")


def work(job: tuple) -> str:
    w, dst = job
    torch.set_num_threads(1)
    f = P.analyze(np.asarray(w, np.float32))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **{k: np.asarray(v, np.float32) for k, v in f.items()})
    return dst


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    nw, limit = int(opt.get("workers", 14)), int(opt.get("limit", 0))
    tr = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["train"]
    rels = [p.relative_to(T.FEAT) for v in tr for p in sorted((T.FEAT / "e1" / v).glob("t*.npz")) + sorted((T.FEAT / "e1nv" / v).glob("*.npz"))]
    jobs = [(k, r) for r in rels for k in (0, 1) if not (OUT / r.parent / f"{r.stem}_k{k}.npz").exists()]
    jobs = jobs[:limit] if limit else jobs
    print("発話", len(rels), "作る数", len(jobs), flush=True)
    me = types.SimpleNamespace(gen=EV.load_gen())
    pend: list = []
    with get_context("spawn").Pool(nw) as pool:
        for i, (k, r) in enumerate(jobs):
            u = T.load_utt(SRC[k] / r)
            w = EV.System.render(me, u, torch.from_numpy(u["y"])[None].to(EV.DEV), 0)
            pend.append(pool.apply_async(work, ((w, str(OUT / r.parent / f"{r.stem}_k{k}.npz")),)))
            while len(pend) > 4 * nw:
                pend.pop(0).get()
            if i % 1000 == 0:
                print(i, flush=True)
        for q in pend:
            q.get()
    n = sum(1 for _ in OUT.rglob("*.npz"))
    (OUT / "manifest.json").write_text(json.dumps({"stage": "P82", "what": "写し(rvoc3hi → 前段 v3)の特徴・k0 = 加工しない発話・k1 = 擬似の操作者の版", "voices": len(tr), "files": n, "source": ["data/snv_p_feat", "data/snv_p_feat_x16"]}, ensure_ascii=False, indent=1))
    print("files", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

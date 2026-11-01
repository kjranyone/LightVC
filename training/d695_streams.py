"""D695(腕 65 = diag_ppgvoc65・paper_mf E-A1t・規則は results/diag_ppgvoc65/prereg.yaml): 推論の c0 の流れ = 学習の女声の c0 の表(results/diag_ppgvoc65/c0_tab.f32)を
D694 の条件の書き出し(H/d694/dump・Rust の C1 の事後)の argmax で引き、学習と同じ因果な 5 フレームの平均(train_ppgvoc.c0tab_np = c0tab_frames)にする → LIGHTVC_DIAG_STREAM の [c0, NaN]。
--check <wav の書き出し>: Rust が診断の差し替え(one-hot・C2 定数・流れ)の後に生成器へ渡す条件(LIGHTVC_DUMP_COND)が、流れ・定数・one-hot と一致することを確かめる。
    uv run python d695_streams.py
    uv run python d695_streams.py --check <dump.f32> --stream <stream.f32> --c2 <female_mean.f32>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_ppgvoc import c0tab_np

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
ROOT = Path(__file__).resolve().parent.parent
DC, Q0, Q1 = 295, 31, 231
SETS = {"d487_uiro": None, "devvctkm": "ME", "devff2": None}


def pairs(s: str, kind: str | None) -> list:
    return [r for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def check(dump: Path, stream: Path, c2: Path, soft: bool = False) -> int:
    x = np.fromfile(dump, dtype="<f4").reshape(-1, DC)
    st = np.fromfile(stream, dtype="<f4").reshape(-1, 2)
    v = np.fromfile(c2, dtype="<f4")
    q = x[:, Q0:Q1]
    rep = {"frames_dump": len(x), "frames_stream": len(st), "max_c0_diff": float(np.abs(x[:, 0] - st[:len(x), 0]).max()),
           "onehot": bool(np.all(q.max(1) == 1.0) and np.all(q.sum(1) == 1.0)), "max_c2_diff": float(np.abs(x[:, Q1:] - v[None]).max())}
    ok = rep["frames_dump"] == rep["frames_stream"] and rep["max_c0_diff"] == 0.0 and rep["onehot"] != soft and rep["max_c2_diff"] == 0.0
    print(json.dumps(rep), "OK" if ok else "MISMATCH", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", default=str(ROOT / "results/diag_ppgvoc65/c0_tab.f32"))
    ap.add_argument("--check", default="")
    ap.add_argument("--stream", default="")
    ap.add_argument("--c2", default="")
    ap.add_argument("--soft", action="store_true", help="--check で事後が one-hot でないこと(腕 66)を確かめる")
    ap.add_argument("--sets", nargs="*", default=[], help="SETS の代わり: <組>=<種類を写す元の組>(例 devvctkm_mp3=devvctkm)")
    ap.add_argument("--report", default=str(ROOT / "results/diag_ppgvoc65/streams.json"))
    a = ap.parse_args()
    if a.check:
        return check(Path(a.check), Path(a.stream), Path(a.c2), a.soft)
    tab = np.fromfile(a.tab, dtype="<f4")
    assert len(tab) == Q1 - Q0
    rep = {}
    sets = {n.split("=")[0]: SETS[n.split("=")[1]] for n in a.sets} if a.sets else SETS
    for s, kd in sets.items():
        (H / "d695" / "stream" / s).mkdir(parents=True, exist_ok=True)
        cs = []
        for r in pairs(s, kd):
            x = np.fromfile(H / "d694" / "dump" / s / f"{r['pi']}_{r['ti']}.f32", dtype="<f4").reshape(-1, DC)
            c = c0tab_np(x[:, Q0:Q1].argmax(1), tab)
            np.stack([c, np.full_like(c, np.nan)], 1).astype("<f4").tofile(H / "d695" / "stream" / s / f"{r['pi']}_{r['ti']}.f32")
            cs.append(float(np.corrcoef(c, x[:, 0])[0, 1]))
        rep[s] = {"n": len(cs), "corr_neutral_vs_orig_median": round(float(np.median(cs)), 3)}
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(rep, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

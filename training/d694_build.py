"""D694(論文 current/paper_mf.md の E-A1 rev0.4・規則は results/conv_p0/d694/prereg.yaml): c0 を女声の中立の値にする診断の流れを作る。
--table: 声の低い実女声(d688_lowself2・学習の外)の条件の書き出し(cond_ppg)から、事後の argmax の単位ごとの c0 の平均の表を作る(20 フレーム未満の単位は全体の平均)。
--streams: 各組の書き出しの argmax の並びを表で引き、中心の 5 フレームの移動平均で c0 の流れを作る(f0 は NaN = 元のまま)→ LIGHTVC_DIAG_STREAM の形 [c0, f0] f32。
同じ書き出しの c0 そのままの流れ(恒等)も 1 組だけ作る(整列の検査: 描画が元とビット一致すること)。
    uv run python d694_build.py --table && uv run python d694_build.py --streams
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
ROOT = Path(__file__).resolve().parent.parent
W = H / "d694"
DC, Q0, Q1 = 295, 31, 231
SETS = {"d487_uiro": None, "devvctkm": "ME", "devff2": None}


def load(p: Path) -> np.ndarray:
    return np.fromfile(p, dtype="<f4").reshape(-1, DC)


def pairs(s: str, kind: str | None) -> list:
    return [r for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def stream(x: np.ndarray, tab: np.ndarray) -> np.ndarray:
    c = tab[x[:, Q0:Q1].argmax(1)]
    c = np.convolve(np.pad(c, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return np.stack([c, np.full_like(c, np.nan)], 1).astype("<f4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--streams", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results/conv_p0/d694"
    out.mkdir(parents=True, exist_ok=True)
    if a.table:
        X = np.concatenate([load(W / "dump" / "d688_lowself2" / f"{r['pi']}_{r['ti']}.f32") for r in pairs("d688_lowself2", None)])
        u = X[:, Q0:Q1].argmax(1)
        c = X[:, 0]
        n = np.bincount(u, minlength=Q1 - Q0)
        s = np.bincount(u, weights=c, minlength=Q1 - Q0)
        tab = np.where(n >= 20, s / np.maximum(n, 1), c.mean()).astype(np.float32)
        tab.tofile(W / "c0_unit.f32")
        sil = int(np.bincount(u[c <= 0.05], minlength=Q1 - Q0).argmax())
        rep = {"frames": int(len(u)), "units_ge20": int((n >= 20).sum()), "global_mean": round(float(c.mean()), 3), "silence_unit": sil,
               "silence_unit_c0": round(float(tab[sil]), 3), "frac_c0_le_0.05": round(float((c <= 0.05).mean()), 3)}
        (out / "build_table.json").write_text(json.dumps(rep, indent=1))
        print(rep, flush=True)
    if a.streams:
        tab = np.fromfile(W / "c0_unit.f32", dtype="<f4")
        rep = {}
        for s, kd in SETS.items():
            (W / "stream" / s).mkdir(parents=True, exist_ok=True)
            cs, lo = [], []
            for r in pairs(s, kd):
                x = load(W / "dump" / s / f"{r['pi']}_{r['ti']}.f32")
                st = stream(x, tab)
                st.tofile(W / "stream" / s / f"{r['pi']}_{r['ti']}.f32")
                cs.append(float(np.corrcoef(st[:, 0], x[:, 0])[0, 1]))
                lo.append(float((x[:, 0] <= 0.05).mean()))
            rep[s] = {"n": len(cs), "corr_neutral_vs_orig_median": round(float(np.median(cs)), 3), "frac_c0_le_0.05_orig_median": round(float(np.median(lo)), 3)}
        r0 = pairs("d487_uiro", None)[0]
        x = load(W / "dump" / "d487_uiro" / f"{r0['pi']}_{r0['ti']}.f32")
        (W / "stream_id" / "d487_uiro").mkdir(parents=True, exist_ok=True)
        np.stack([x[:, 0], np.full(len(x), np.nan, dtype=np.float32)], 1).astype("<f4").tofile(W / "stream_id" / "d487_uiro" / f"{r0['pi']}_{r0['ti']}.f32")
        rep["identity_pair"] = f"p{r0['pi']:04d}_t{r0['ti']:03d}"
        (out / "build_streams.json").write_text(json.dumps(rep, indent=1))
        print(rep, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

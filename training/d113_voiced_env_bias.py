"""D113(学習なし): D112 = 製品の変換は有声フレームが本物より暗い(傾き −4.65 vs −3.12 dB/oct・4–10kHz −4dB)。条件(出力部の入力)の段で暗いのか、出力部が暗く鳴らすのかを分ける。
Rust の製品(results/rvtab_x/export・F4・確信度 0.5)を LIGHTVC_DUMP_COND つきで走らせ、有声フレーム(条件の行 26)の包絡の行 1..24 の平均を
  CONV_TAB(男声 → 女声・表だけ γ0)・CONV(γ0.7 + 色 = C07F5)・SELF_TAB(目標の実音声 cen → 同じ目標の表だけ・操作者なし)と、
  目標の実音声の教師の包絡(CheapTrick・教師 f0 の有声)で比べる。差を mel の対数振幅(dB)へ戻し 0–1k / 1–4k / 4–10kHz の平均で示す。
  条件の差が D112 の暗さを説明 → 表の引き(男声の入力での単位の事後)か操作者の差の項・条件が本物並み → 出力部。出力 = results/conv_p0/d113.json。
    uv run python d113_voiced_env_bias.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
EXTRA = ["--rvoc", "results/rvtab_x/export", "--f4", "results/f4_1/export", "--f0-conf", "0.5"]


def job(j: tuple) -> None:
    cmd, dump = j
    if not Path(dump).exists():
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env={"LIGHTVC_DUMP_COND": dump, "PATH": "/usr/bin:/bin"}, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import f0hi as H
    import nvoc as N
    import pae as PA
    from d6_unit_cov import load48
    S = Path(a.s)
    work = S / "d85_work"
    od = S / "d113"; od.mkdir(exist_ok=True)
    jobs = []
    pairs = sorted((S / "d56").glob("p*_SRC.wav"))
    for f in pairs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        base = [BIN, "artic", "-i", f, "--enroll", work / f"en_{ti:03d}", "--operator-enroll", work / f"op_{pi:04d}", "--mix", "0", "--per-p75", "--med-from-operator", "--warp", "--alpha", "1.3"] + EXTRA
        jobs.append((base + ["-o", od / f"{f.stem}_CONV_TAB.wav", "--resid", "0"], str(od / f"{f.stem}_CONV_TAB.f32")))
        jobs.append((base + ["-o", od / f"{f.stem}_CONV.wav", "--resid", "0.7", "--color"], str(od / f"{f.stem}_CONV.f32")))
    for t in range(30):
        jobs.append(([BIN, "artic", "-i", S / "d37" / f"cen_{t:03d}.wav", "-o", od / f"t{t:03d}_SELF_TAB.wav", "--enroll", work / f"en_{t:03d}", "--mix", "0", "--per-p75"] + EXTRA,
                     str(od / f"t{t:03d}_SELF_TAB.f32")))
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(job, jobs))
    fb, dct = PA._np_consts()
    fbm = np.asarray(fb)
    fc = (fbm * np.arange(fbm.shape[1])[None]).sum(1) / np.maximum(fbm.sum(1), 1e-9) * (N.SR / 2) / (fbm.shape[1] - 1)
    db = 20 / np.log(10)

    def vmean(path: str) -> np.ndarray:
        c = np.fromfile(path, "<f4").reshape(-1, 31)
        v = c[:, 26] > 0.5
        return c[v, 1:25].mean(0) * 10

    teach = {}
    for t in range(30):
        x = load48(S / "d37" / f"cen_{t:03d}.wav").astype(np.float32)[: 8 * N.SR]
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = PA.envelope(xa, f0)[:, :n]
        teach[t] = e[1:25, f0[: e.shape[1]] > 0].mean(1)
    rows = {"CONV_TAB": [], "CONV": [], "SELF_TAB": []}
    for f in pairs:
        ti = int(f.name[7:10])
        for c in ("CONV_TAB", "CONV"):
            rows[c].append(vmean(str(od / f"{f.stem}_{c}.f32")) - teach[ti])
    for t in range(30):
        rows["SELF_TAB"].append(vmean(str(od / f"t{t:03d}_SELF_TAB.f32")) - teach[t])
    bands = {"0-1k": fc < 1000, "1-4k": (fc >= 1000) & (fc < 4000), "4-10k": (fc >= 4000) & (fc < 10000)}
    rep = {}
    for c, d in rows.items():
        dm = np.mean(d, 0)
        mel_db = db * (np.asarray(dct)[1:25].T @ dm)
        rep[c] = {"c1_c4_diff": [round(float(z), 3) for z in dm[:4]], **{b: round(float(mel_db[m].mean()), 2) for b, m in bands.items()},
                  "hi_minus_lo_db": round(float(mel_db[bands["4-10k"]].mean() - mel_db[bands["0-1k"]].mean()), 2)}
        print(c, json.dumps(rep[c]), flush=True)
    (ROOT / "results/conv_p0/d113.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

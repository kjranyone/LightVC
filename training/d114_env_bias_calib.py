"""D114(学習なし): D113 = 製品の変換の有声フレームの包絡は条件の段で暗い(4–10kHz − 0–1kHz が目標の教師より −3.5dB)。登録の定数で一次の較正をする。
  b = 目標の登録の参照(ref_TTT)の教師の包絡(CheapTrick・教師 f0)の有声平均 − 操作者の登録の参照(op_PPPP)を同じ変換(γ0.7 + 色・F4・確信度 0.5)に通した条件の有声平均(Rust の LIGHTVC_DUMP_COND)。
  変換は有声フレームの行 1..24 に b を足す(Rust `artic --env-bias`)。b は登録の参照だけから作る(評価の発話は使わない・推論の経路に発話全体の統計を置かない)。
  評価の発話 = D56 の 60 組の男声 → <s>/d114/pNNNN_tNNN_C07F5B.wav(+ 比較の C07F5 = 較正なし・SRC・cen)。
  ゆらぎの床 = 同じ目標の ref と cen の教師の有声平均の差(4–10k − 0–1k の dB)。出力 = results/conv_p0/d114.json。
    uv run python d114_env_bias_calib.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
EXTRA = ["--rvoc", "results/rvtab_x/export", "--f4", "results/f4_1/export", "--f0-conf", "0.5"]
CFG = {"C07F5": ["--resid", "0.7", "--color"], "T05": ["--resid", "0", "--soft", "0.5"], "C07S05": ["--resid", "0.7", "--color", "--soft", "0.5"],
       "G00S03": ["--resid", "0", "--soft", "0.3"], "G03S03": ["--resid", "0.3", "--color", "--soft", "0.3"], "G05S03": ["--resid", "0.5", "--color", "--soft", "0.3"],
       "G07S03": ["--resid", "0.7", "--color", "--soft", "0.3"], "G00S10": ["--resid", "0", "--soft", "1"], "G03S10": ["--resid", "0.3", "--color", "--soft", "1"],
       "G04S10": ["--resid", "0.4", "--color", "--soft", "1"], "G05S10": ["--resid", "0.5", "--color", "--soft", "1"],
       "GA07": ["--gamma-adapt", "0,0.7,0.10,0.25", "--color", "--soft", "1"], "GA28": ["--gamma-adapt", "0.2,0.8,0.10,0.25", "--color", "--soft", "1"],
       "DB1": ["--resid", "0.5", "--color", "--soft", "1", "--resid-dims", "1-4:1,5-24:0"], "DB2": ["--resid", "0.5", "--color", "--soft", "1", "--resid-dims", "1-4:0,5-12:1,13-24:0"],
       "DB3": ["--resid", "0.5", "--color", "--soft", "1", "--resid-dims", "1-12:0,13-24:1"],
       "G05S10Z": ["--resid", "0.5", "--color", "--soft", "1"], "G00S10Z": ["--resid", "0", "--soft", "1"],
       "G04S10Z": ["--resid", "0.4", "--color", "--soft", "1"], "G03S10Z": ["--resid", "0.3", "--color", "--soft", "1"],
       "G05L12Z": ["--resid", "0.5", "--color", "--soft", "1", "--resid-dims", "1-12:1,13-24:0"],
       "G05L12hZ": ["--resid", "0.5", "--color", "--soft", "1", "--resid-dims", "1-12:1,13-24:0.5"]}


def job(j: tuple) -> None:
    cmd, dump, out = j
    if not Path(out).exists():
        env = {"PATH": "/usr/bin:/bin"}
        if dump:
            env["LIGHTVC_DUMP_COND"] = dump
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cfg", default="C07F5", choices=list(CFG))
    a = ap.parse_args()
    sfx = "" if a.cfg == "C07F5" else f"_{a.cfg}"
    import f0hi as H
    import nvoc as N
    import pae as PA
    from d6_unit_cov import load48
    S = Path(a.s)
    work = S / "d85_work"
    od = S / "d114"; od.mkdir(exist_ok=True)
    cal = od / f"_cal{sfx}"; cal.mkdir(exist_ok=True)
    pairs = sorted((S / "d56").glob("p*_SRC.wav"))
    pid = sorted({int(f.name[1:5]) for f in pairs})

    def base(pi: int, ti: int) -> list:
        return [BIN, "artic", "--enroll", work / f"en_{ti:03d}", "--operator-enroll", work / f"op_{pi:04d}", "--mix", "0", "--per-p75", "--med-from-operator", "--warp", "--alpha", "1.3"] + EXTRA + CFG[a.cfg]

    if a.cfg.endswith("Z"):
        ucal = od / f"_cal_{a.cfg[:-1]}"
        if not ucal.exists():
            ucal = od / "_cal_G05S10"
        tdir = od / "_cal_tref"; tdir.mkdir(exist_ok=True)
        first = {}
        for f in pairs:
            first.setdefault(int(f.name[7:10]), int(f.name[1:5]))
        with Pool(a.workers) as pool:
            list(pool.imap_unordered(job, [(base(p_, t_) + ["-i", S / "d37/_work" / f"ref_{t_:03d}.wav", "-o", tdir / f"t{t_:03d}.wav"], str(tdir / f"t{t_:03d}.f32"), str(tdir / f"t{t_:03d}.wav")) for t_, p_ in first.items()]))
        vmean = lambda q: float((lambda c: c[c[:, 26] > 0.5, 0].mean())(np.fromfile(q, "<f4").reshape(-1, 31)))
        c0T = {t_: vmean(tdir / f"t{t_:03d}.f32") for t_ in first}
        shift = {}
        for f in pairs:
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            shift[(pi, ti)] = c0T[ti] - vmean(ucal / f"op{pi:04d}_t{ti:03d}.f32")
        (od / f"_c0shift_{a.cfg}.json").write_text(json.dumps({f"{k[0]}_{k[1]}": round(v, 4) for k, v in shift.items()}, indent=1))
        print("c0 shift median", round(float(np.median(list(shift.values()))), 3), "p10/p90", np.round(np.percentile(list(shift.values()), [10, 90]), 3).tolist(), flush=True)
        base0 = base
        base = lambda pi, ti: base0(pi, ti) + ["--c0-shift", f"{shift[(pi, ti)]:.4f}"]

    jobs = []
    for f in pairs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        jobs.append((base(pi, ti) + ["-i", S / "d56/_work" / f"op_{pi:04d}.wav", "-o", cal / f"op{pi:04d}_t{ti:03d}.wav"], str(cal / f"op{pi:04d}_t{ti:03d}.f32"), str(cal / f"op{pi:04d}_t{ti:03d}.wav")))
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(job, jobs))

    def teach_mean(p: Path, sec: float) -> np.ndarray:
        x = load48(p).astype(np.float32)[: int(sec * N.SR)]
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = PA.envelope(xa, f0)[:, :n]
        return e[1:25, f0[: e.shape[1]] > 0].mean(1) / 10

    tref = {t: teach_mean(S / "d37/_work" / f"ref_{t:03d}.wav", 30.0) for t in range(30)}
    tcen = {t: teach_mean(S / "d37" / f"cen_{t:03d}.wav", 8.0) for t in range(30)}
    fb, dct = PA._np_consts()
    fc = (fb * np.arange(fb.shape[1])[None]).sum(1) / np.maximum(fb.sum(1), 1e-9) * (N.SR / 2) / (fb.shape[1] - 1)
    db = 20 / np.log(10)

    def hilo(dc: np.ndarray) -> float:
        m = db * (dct[1:25].T @ (dc * 10))
        return float(m[(fc >= 4000) & (fc < 10000)].mean() - m[fc < 1000].mean())

    floor = [hilo(tcen[t] - tref[t]) for t in range(30)]
    jobs, bs = [], []
    for f in pairs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        c = np.fromfile(cal / f"op{pi:04d}_t{ti:03d}.f32", "<f4").reshape(-1, 31)
        mop = c[c[:, 26] > 0.5, 1:25].mean(0)
        b = (tref[ti] - mop).astype("<f4")
        bp = cal / f"bias_{pi:04d}_t{ti:03d}.f32"
        b.tofile(bp)
        bs.append(hilo(b))
        jobs.append((base(pi, ti) + ["-i", f, "-o", od / f"{f.stem[:-4]}_{a.cfg}B.wav", "--env-bias", bp], None, str(od / f"{f.stem[:-4]}_{a.cfg}B.wav")))
        shutil.copy(f, od / f.name)
        if a.cfg == "C07F5":
            shutil.copy(S / "d110" / f"{f.stem[:-4]}_C07F5.wav", od / f"{f.stem[:-4]}_C07F5.wav")
    for t in range(30):
        shutil.copy(S / "d37" / f"cen_{t:03d}.wav", od / f"cen_{t:03d}.wav")
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(job, jobs))
    rep = {"bias_hilo_db": {"median": round(float(np.median(bs)), 2), "p10": round(float(np.percentile(bs, 10)), 2), "p90": round(float(np.percentile(bs, 90)), 2)},
           "floor_cen_minus_ref_hilo_db": {"median": round(float(np.median(floor)), 2), "abs_p90": round(float(np.percentile(np.abs(floor), 90)), 2)}}
    print(json.dumps(rep), flush=True)
    (ROOT / f"results/conv_p0/d114{sfx}.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

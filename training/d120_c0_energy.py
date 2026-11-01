"""D120 = D115 の派生(--cal で較正の構成を選ぶ・既定 _cal = C07F5 の較正): C07S05E = γ0.7 + 色 + β0.5 + C07S05 の較正 + c0 のエネルギー整合(D119・Rust --c0-energy)。
以下は D115 の説明。D115(学習なし): D109 v2 で較正つきの変換(D114 C07F5B)に残る失格 = ガビ軸の shimmer(隣接 10ms フレームの音量の揺れ 0.782・本物の p85 0.772)。
D106 の帰属: f0・c0・動き・周期性を A の本物に替えても下がらず、A の包絡全部(AWARP)で 0.59 = 表の引きの土台(Σp T)が出どころ。D66 の女声の表の写し合成も 0.84。
表の引きのつまみを振る(Rust・D114 の較正 b つき・他は C07F5B と同じ): S05 / S2 = 柔らかい引きの鋭さ β 0.5 / 2・W2 / W3 = 表の引きの因果平均の幅 2 / 3(遅れ 0.5 / 1 フレーム)。
shimmer は D109・明瞭さは D23 の CER(D27: 平滑で明瞭さが落ちた)で測る。出力 = <s>/d115/pNNNN_tNNN_<cond>.wav。
    uv run python d115_table_smooth.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
EXTRA = ["--rvoc", "results/rvtab_x/export", "--f4", "results/f4_1/export", "--f0-conf", "0.5", "--resid", "0.7", "--color"]
CONDS = {"S05": ["--soft", "0.5"], "S2": ["--soft", "2"], "W2": ["--w-sm", "2"], "W3": ["--w-sm", "3"], "S03": ["--soft", "0.3"], "S07": ["--soft", "0.7"]}
CONDS_CAL = {"_cal_C07S05": {"C07S05E": ["--soft", "0.5", "--c0-energy"]}}


def job(cmd: list) -> None:
    out = cmd[cmd.index("-o") + 1]
    if not Path(out).exists():
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cal", default="_cal")
    ap.add_argument("--out", default="d115")
    a = ap.parse_args()
    S = Path(a.s)
    work, cal, od = S / "d85_work", S / "d114" / a.cal, S / a.out
    conds = CONDS_CAL.get(a.cal, CONDS)
    od.mkdir(exist_ok=True)
    jobs = []
    for f in sorted((S / "d56").glob("p*_SRC.wav")):
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        base = [BIN, "artic", "-i", f, "--enroll", work / f"en_{ti:03d}", "--operator-enroll", work / f"op_{pi:04d}", "--mix", "0", "--per-p75", "--med-from-operator",
                "--warp", "--alpha", "1.3", "--env-bias", cal / f"bias_{pi:04d}_t{ti:03d}.f32"] + EXTRA
        for c, args in conds.items():
            jobs.append(base + args + ["-o", od / f"{k}_{c}.wav"])
        if (S / "d114" / f"{k}_C07S05B.wav").exists():
            shutil.copy(S / "d114" / f"{k}_C07S05B.wav", od / f"{k}_C07S05B.wav")
        shutil.copy(f, od / f.name)
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(job, jobs))
    print("done", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

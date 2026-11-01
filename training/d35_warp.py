"""D35: 製品の条件(D29/D31/D33 と同じ組・同じ元の文・同じ目標の登録・D33 の操作者の生の登録)で、話者差の元の声の動きを周波数伸縮してから足す(Rust `artic --warp --alpha`)。
包絡 = Σp T_目標 + W_α(元の包絡 − Σp T_操作者)。α は物理のつまみとして固定値で掃引(表どうしの当てはめは相関が平らで決まらない)。W100 は D31 の DLT と同じ式(検算)。
出力 = d4b の形式: D33 の cen・SRC・TAB・M35・DLT を写し、W100・W110・W120・W130 を加える(全て混ぜ 0)。
    uv run python d35_warp.py --d29 <scratchpad>/d29 --d33 <scratchpad>/d33 --out_dir <scratchpad>/d35 --n_tg 10
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
ALPHAS = (1.0, 1.1, 1.2, 1.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d29", required=True)
    ap.add_argument("--d33", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_tg", type=int, default=10)
    ap.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    d29, d33 = Path(a.d29), Path(a.d33)
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    males = sorted(p.name for p in (ROOT / "data/male_tts_corpus").iterdir() if p.is_dir())
    pi = 0
    for ti in range(a.n_tg):
        shutil.copy(d33 / f"cen_{ti:03d}.wav", od / f"cen_{ti:03d}.wav")
        endir = d29 / "_work" / f"enroll_{ti:03d}"
        for j in range(a.per):
            m = males[(ti * a.per + j) % len(males)]
            for c in ("SRC", "TAB", "M35", "DLT"):
                shutil.copy(d33 / f"p{pi:04d}_t{ti:03d}_{c}.wav", od / f"p{pi:04d}_t{ti:03d}_{c}.wav")
            srcp = d33 / f"p{pi:04d}_t{ti:03d}_SRC.wav"
            for al in ALPHAS:
                outp = od / f"p{pi:04d}_t{ti:03d}_W{round(al * 100)}.wav"
                cmd = [str(BIN), "artic", "-i", str(srcp), "-o", str(outp), "--mix", "0", "--enroll", str(endir),
                       "--operator-enroll", str(d33 / "_work" / f"en_{m}"), "--warp", "--alpha", str(al)]
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                if r.returncode != 0:
                    print("FAIL", cmd, r.stderr[-500:], flush=True)
                    return 1
            pi += 1
        print("target", ti, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

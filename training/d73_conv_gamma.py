"""D73 = D67 の派生(--conds で条件を替える): 例 --conds W10:1.0 で出力部はそのまま(--rvoc results/rvoc3hi/export)・動きを縮めない γ 1。
以下は D67 の説明。D67: 出力部の追加学習(表の包絡を条件に)の後の変換(男声 → 女声)を、Rust の製品経路のまま出力部の重みだけ替えて鳴らす。
元・登録 = D56(<s>/d56 の SRC・_work/op_PPPP・D53 の目標の登録)。共通 = --mix 0・操作者の登録・周期性 75 パーセンタイル・高さの基準 = 操作者の中央値・--warp。
  BTAB = D56 の TAB(rvoc3hi・γ 0)・BW07 = D56 の L4FULL(rvoc3hi・α 1.3・γ 0.7)・FTAB = 追加学習後・γ 0・FW07 = 追加学習後・α 1.3・γ 0.7。
出力 = d4b の形式(cen・pNNNN_tNNN_<cond>・SRC)。
    uv run python d67_conv_ft.py --s <scratchpad> --rvoc results/diag_rvtab/export --out_dir <scratchpad>/d67
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {cmd} {r.stderr[-500:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--rvoc", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--conds", default="FTAB:0,FW07:0.7")
    a = ap.parse_args()
    S = Path(a.s)
    src = S / "d56"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    com = ["--mix", "0", "--per-p75", "--med-from-operator", "--warp", "--alpha", "1.3", "--rvoc", Path(a.rvoc).resolve()]
    for f in sorted(src.glob("p*_SRC.wav"))[: a.max]:
        k = f.name[: -len("_SRC.wav")]
        m = re.match(r"p(\d+)_t(\d+)", k)
        pi, ti = int(m.group(1)), int(m.group(2))
        en = S / "d53_en" / f"en_{ti:03d}"
        op = src / "_work" / f"op_{pi:04d}"
        for name, g in (c.split(":") for c in a.conds.split(",")):
            run([BIN, "artic", "-i", f, "-o", od / f"{k}_{name}.wav", "--enroll", en, "--operator-enroll", op, "--resid", g] + com)
        shutil.copy(f, od / f.name)
        shutil.copy(src / f"{k}_TAB.wav", od / f"{k}_BTAB.wav")
        shutil.copy(src / f"{k}_L4FULL.wav", od / f"{k}_BW07.wav")
        print(k, flush=True)
    for c in sorted(src.glob("cen_*.wav")):
        shutil.copy(c, od / c.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

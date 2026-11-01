"""D33: 製品の条件(D29/D31 と同じ組・同じ元の文・同じ目標の登録・Rust の製品経路)で、操作者の基準の表の作り方を比べる。
男声 TTS 47 人を全員 Rust で登録(lightvc-app artic-enroll・参照 22s 以上・D31 の操作者は同じ参照の WAV)。
  DPOPM = 元の包絡 + Σp(T_目標 − 男声の母集団の平均の表(操作者を除く 46 人))・DSH<κ> = 操作者の表を観測数で縮める (n_u·T_op + κ·T_popM)/(n_u + κ)。
出力 = d4b の形式: D31 の cen・SRC・TAB・M35・DLT を写し、DPOPM・DSH10・DSH30・DSH100 を加える(全て混ぜ 0)。
    uv run python d33_operator_shrink.py --d29 <scratchpad>/d29 --d31 <scratchpad>/d31 --out_dir <scratchpad>/d33 --n_tg 10
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
KAPPAS = (10, 30, 100)


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {cmd} {r.stderr[-500:]}")


def read_en(d: Path) -> tuple:
    j = json.loads((d / "enroll.json").read_text())
    t = np.fromfile(d / "enroll.bin", dtype="<f4").reshape(j["D"], j["K"]).astype(np.float64)
    return t, np.asarray(j["counts"], np.float64), j


def write_en(d: Path, table: np.ndarray, j: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    table.astype("<f4").tofile(d / "enroll.bin")
    (d / "enroll.json").write_text(json.dumps({"D": j["D"], "K": j["K"], "mu": j["mu"], "per_mean": j["per_mean"], "counts": []}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d29", required=True)
    ap.add_argument("--d31", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_tg", type=int, default=10)
    ap.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    d29, d31 = Path(a.d29), Path(a.d31)
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    work = od / "_work"; work.mkdir(exist_ok=True)
    males = sorted(p.name for p in (ROOT / "data/male_tts_corpus").iterdir() if p.is_dir())
    tabs, cnts, js = {}, {}, {}
    for m in males:
        wav = d31 / "_work" / f"op_{m}.wav"
        if not wav.exists():
            wav = work / f"op_{m}.wav"
            if not wav.exists():
                xs, tot = [], 0.0
                for f in sorted((ROOT / "data/male_tts_corpus" / m).glob("t*_neutral.wav")):
                    if tot >= 22.0:
                        break
                    x = load48(f); xs.append(x); tot += len(x) / 48000
                sf.write(wav, np.concatenate(xs).astype(np.float32), 48000)
        en = work / f"en_{m}"
        if not (en / "enroll.json").exists():
            run([BIN, "artic-enroll", "-r", wav, "-o", en])
        tabs[m], cnts[m], js[m] = read_en(en)
        assert len(cnts[m]) == tabs[m].shape[1], f"{m} の登録に観測数がない"
    print("enrolled", len(males), "males; seen units median", float(np.median([(c > 0).sum() for c in cnts.values()])), flush=True)
    allT = np.stack([tabs[m] for m in males])
    pi = 0
    for ti in range(a.n_tg):
        shutil.copy(d31 / f"cen_{ti:03d}.wav", od / f"cen_{ti:03d}.wav")
        endir = d29 / "_work" / f"enroll_{ti:03d}"
        for j in range(a.per):
            mi = (ti * a.per + j) % len(males)
            m = males[mi]
            popm = np.delete(allT, mi, 0).mean(0)
            refs = {"DPOPM": popm}
            for kp in KAPPAS:
                n = cnts[m][None]
                refs[f"DSH{kp}"] = (n * tabs[m] + kp * popm) / (n + kp)
            for c in ("SRC", "TAB", "M35", "DLT"):
                shutil.copy(d31 / f"p{pi:04d}_t{ti:03d}_{c}.wav", od / f"p{pi:04d}_t{ti:03d}_{c}.wav")
            srcp = d31 / f"p{pi:04d}_t{ti:03d}_SRC.wav"
            for name, table in refs.items():
                rd = work / f"ref_{name}_{m}"
                write_en(rd, table, js[m])
                run([BIN, "artic", "-i", srcp, "-o", od / f"p{pi:04d}_t{ti:03d}_{name}.wav", "--mix", "0", "--enroll", endir, "--operator-enroll", rd])
            pi += 1
        print("target", ti, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

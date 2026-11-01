"""D704(paper_mf E-C1d・規則は results/conv_p0/d704/prereg.yaml): 関門の組の入力を外郎売と同じ型の mp3 で往復した組 H/<組>_mp3 を作る(学習なし・評価専用)。
入力(d56/_work/pre_<key>.wav = 操作者の参照 12 s + 元の声 8 s)・元の声(d56/<key>_SRC.wav)・操作者の参照(d56/_work/op_PPPP.wav)を ffmpeg libmp3lame(48 kHz・mono・80 kbps)で往復し、
操作者の登録(d85_work/op_PPPP)を mp3 の参照から artic-enroll で作り直す。目標の側(d37・rr_en・d85_work/en_TTT)は元の組への symlink。
--codec none --n 1: 同じ手順で符号化だけ外した組 H/<組>_id(描画が元の関門の出力とビット一致することの検査)。
    uv run python d704_mp3set.py --sets devvctkm:ME devff2
    uv run python d704_mp3set.py --sets devvctkm:ME devff2 --codec none --n 1
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
SR = 48000


def mp3_rt(src: Path, dst: Path, kbps: int) -> dict:
    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    assert sr == SR, (src, sr)
    x = x.mean(1)
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "a.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), "-c:a", "libmp3lame", "-b:a", f"{kbps}k", str(m)], check=True)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(m), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype="<f4").copy()
    c = correlate(y, x, mode="full", method="fft")
    W = 3000
    mid = len(x) - 1
    lag = int(np.argmax(c[mid - W:mid + W + 1])) - W
    z = np.zeros(len(x), np.float32)
    if lag >= 0:
        k = min(len(x), len(y) - lag)
        z[:k] = y[lag:lag + k]
    else:
        k = min(len(x) + lag, len(y))
        z[-lag:-lag + k] = y[:k]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), z, SR, subtype="FLOAT")
    e = x - z
    return {"lag": lag, "len_in": len(x), "len_dec": len(y), "snr_db": round(float(10 * np.log10(np.sum(x ** 2) / max(np.sum(e ** 2), 1e-12))), 2)}


def copy_wav(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return {"lag": 0, "snr_db": None}


def job(t: tuple) -> dict:
    src, dst, codec, kbps = t
    r = mp3_rt(src, dst, kbps) if codec == "mp3" else copy_wav(src, dst)
    return {"file": dst.name} | r


def build(base: str, kind: str | None, codec: str, kbps: int, n: int, workers: int) -> dict:
    S = H / base
    O = H / f"{base}_{'mp3' if codec == 'mp3' else 'id'}"
    assert not (O / "devset.json").exists(), f"{O} は既にある"
    d = json.loads((S / "devset.json").read_text())
    P = [r for r in d["pairs"] if kind is None or r["kind"] == kind]
    if n > 0:
        P = P[:n]
    (O / "d56" / "_work").mkdir(parents=True, exist_ok=True)
    (O / "d85_work").mkdir(parents=True, exist_ok=True)
    for sub in ("d37", "rr_en"):
        (O / sub).symlink_to(S / sub)
    for ti in sorted({r["ti"] for r in P}):
        (O / "d85_work" / f"en_{ti:03d}").symlink_to(S / "d85_work" / f"en_{ti:03d}")
    jobs, ops = [], {}
    for r in P:
        key = f"p{r['pi']:04d}_t{r['ti']:03d}"
        jobs.append((S / "d56" / "_work" / f"pre_{key}.wav", O / "d56" / "_work" / f"pre_{key}.wav", codec, kbps))
        jobs.append((S / "d56" / f"{key}_SRC.wav", O / "d56" / f"{key}_SRC.wav", codec, kbps))
        ops.setdefault((S / "d56" / "_work" / f"op_{r['pi']:04d}.wav").resolve(), []).append(r["pi"])
    for k, src in enumerate(ops):
        jobs.append((src, O / "d56" / "_work" / f"opsrc_{k:03d}.wav", codec, kbps))
    with Pool(workers) as pl:
        res = pl.map(job, jobs)
    for k, (src, pis) in enumerate(ops.items()):
        w = O / "d56" / "_work" / f"opsrc_{k:03d}.wav"
        spd = O / "d85_work" / f"opspk_{k:03d}"
        r_ = subprocess.run([str(BIN), "artic-enroll", "-r", str(w), "-o", str(spd)], cwd=ROOT, capture_output=True, text=True)
        assert r_.returncode == 0, r_.stderr[-500:]
        for pi in sorted(set(pis)):
            (O / "d56" / "_work" / f"op_{pi:04d}.wav").symlink_to(w.name)
            (O / "d85_work" / f"op_{pi:04d}").symlink_to(spd.name)
    (O / "devset.json").write_text(json.dumps({k: v for k, v in d.items() if k != "pairs"} | {
        "pairs": P, "source": f"{base} の入力・元の声・操作者の参照を {'mp3 ' + str(kbps) + ' kbps で往復' if codec == 'mp3' else '符号化なしで写し'}(D704)"}, ensure_ascii=False, indent=1))
    lags = [x["lag"] for x in res]
    snr = [x["snr_db"] for x in res if x["snr_db"] is not None]
    return {"set": O.name, "pairs": len(P), "operators": len(ops), "files": len(res), "lag_values": sorted(set(lags)),
            "snr_db_median": round(float(np.median(snr)), 2) if snr else None, "snr_db_min": round(float(np.min(snr)), 2) if snr else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="<組>[:<種類>]")
    ap.add_argument("--codec", default="mp3", choices=("mp3", "none"))
    ap.add_argument("--kbps", type=int, default=80)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    rep = []
    for s in a.sets:
        base, _, kind = s.partition(":")
        rep.append(build(base, kind or None, a.codec, a.kbps, a.n, a.workers))
        print(json.dumps(rep[-1], ensure_ascii=False), flush=True)
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

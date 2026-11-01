"""--causal-ct(Rust)と D253 の PQP(Python)の条件の一致検査: devff の入力を自分自身へ(--enroll op --operator-enroll op --mix 1 --med-from-operator)で Rust の条件を書き出し、Python の同じ計算と比べる。
    uv run python cct_parity.py --s <dev の親> --out_dir <scratchpad>/cct_parity --n 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    import f0est as FE
    import nvoc as N
    import pae as PA
    import pyworld
    F = Path(a.s) / "devff"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    fb_, dct_ = PA._np_consts()
    w = 0.25
    worst = 0.0
    for f in sorted((F / "d56").glob("p00*_SRC.wav"))[: a.n]:
        pi = int(f.name[1:5])
        op = F / "d85_work" / f"op_{pi:04d}"
        dump = od / f"{f.stem}.f32"
        subprocess.run([str(BIN), "artic", "--enroll", str(op), "--operator-enroll", str(op), "--mix", "1", "--med-from-operator", "--causal-ct", "--rvoc", "results/rvoc3hi/export",
                        "--f4", "results/f4_1/export", "--per-p75", "-i", str(f), "-o", str(od / f"{f.stem}_rust.wav")], cwd=ROOT, env={"PATH": "/usr/bin:/bin", "LIGHTVC_DUMP_COND": str(dump)}, check=True, capture_output=True)
        cr = np.fromfile(dump, "<f4").reshape(-1, 31)
        x, sr = sf.read(f, dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        with torch.no_grad():
            v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x)[None].to(dev)], 1), n))
            f0 = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
        f0 = np.pad(f0, (0, n - len(f0)), mode="edge")
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        T0 = 1.0 / np.where(f0 > 0, f0, 200.0)
        tau = np.maximum((PA.A + np.minimum(np.arange(n) * N.HOP, (np.arange(n) + 1) * N.HOP - 1.5 * T0 * N.SR)) / N.SR, PA.A / N.SR)
        sp_ = pyworld.cheaptrick(xa.astype(np.float64), np.where(f0 > 0, f0, 0.0).astype(np.float64), tau, N.SR, fft_size=PA.NFFT)
        ec = dct_[:25] @ np.log(np.maximum(np.sqrt(sp_) @ fb_.T, 1e-5)).T
        q = np.pad(ec, ((0, 0), (2, 0)), mode="edge")
        sm = w * q[:, :-2] + (1 - 2 * w) * q[:, 1:-1] + w * q[:, 2:]
        c0n, env = (sm[0] - PA.C0_SIL) / 10, sm[1:25] / 10
        m = min(len(cr), n)
        d0 = np.abs(cr[:m, 0] - c0n[:m])
        de = np.abs(cr[:m, 1:25] - env[:, :m].T)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1) / 200), 0)
        df = np.abs(cr[:m, 25] - lf[:m])
        print(f.name, "frames", m, "c0n max|d|", round(float(d0.max()), 5), "p99", round(float(np.percentile(d0, 99)), 5), "env max|d|", round(float(de.max()), 5), "p99", round(float(np.percentile(de, 99)), 5),
              "logf0 max|d|", round(float(df.max()), 4), "voicing mismatch", int(((cr[:m, 26] > 0.5) != (f0[:m] > 0)).sum()), flush=True)
        worst = max(worst, float(np.percentile(de, 99)), float(np.percentile(d0, 99)))
    print("PARITY", "PASS" if worst < 1e-2 else "CHECK", round(worst, 5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

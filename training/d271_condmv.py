"""D271(学習なし・results/conv_p0/d271_prereg.yaml): 男声 → 女声の条件の行ごとの分布(有声 / 無声別の平均・標準偏差)を、操作者の参照を同じ変換に通した値 → 目標の参照の写し合成の値へ写し(登録の定数)、Python の rvoc3hi で鳴らす。
    uv run python d271_condmv.py --s <dev の親> --n 30
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
COMMON = ["--causal-ct", "--rvoc", "results/rvoc3hi/export", "--per-p75", "--med-from-operator", "--f4", "results/f4_1/export"]
CONV = ["--floor-apply", "--warp", "--resid", "0.7", "--color", "--mix", "0", "--alpha", "1.3"]
ROWS = {"ENV": list(range(1, 25)), "PER": list(range(27, 31)), "C0": [0]}


def job(j: tuple) -> None:
    cmd, dump = j
    if not Path(dump).exists():
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env={"PATH": "/usr/bin:/bin", "LIGHTVC_DUMP_COND": str(dump)}, check=True, capture_output=True)


def stats(c: np.ndarray) -> dict:
    v = c[:, 26] > 0.5
    return {m: (c[sel].mean(0), c[sel].std(0) + 1e-6) for m, sel in (("v", v), ("u", ~v)) if sel.sum() > 10}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    import nvoc as N
    import rvoc as R
    import train_rvoc as TR
    D = Path(a.s) / "dev"
    od = D / "d271"
    od.mkdir(exist_ok=True)
    srcs = sorted((D / "d56").glob("p00*_SRC.wav"))[: a.n]
    jobs = []
    for f in srcs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        en, op = D / "d85_work" / f"en_{ti:03d}", D / "d85_work" / f"op_{pi:04d}"
        jobs.append(([BIN, "artic", "--enroll", en, "--operator-enroll", op] + COMMON + CONV + ["-i", f, "-o", od / f"_o_{f.stem}.wav"], od / f"cv_{f.stem}.f32"))
        jobs.append(([BIN, "artic", "--enroll", en, "--operator-enroll", op] + COMMON + CONV + ["-i", D / "d56/_work" / f"op_{pi:04d}.wav", "-o", od / f"_o_op{pi:04d}_t{ti:03d}.wav"], od / f"opcv_{pi:04d}_t{ti:03d}.f32"))
        jobs.append(([BIN, "artic", "--enroll", en, "--operator-enroll", en, "--mix", "1"] + COMMON + ["-i", D / "d37/_work" / f"ref_{ti:03d}.wav", "-o", od / f"_o_ref{ti:03d}.wav"], od / f"ref_{ti:03d}.f32"))
    uniq = {str(d): (c, d) for c, d in jobs}
    with Pool(10) as pl:
        list(pl.imap_unordered(job, list(uniq.values())))
    print("dumped", len(uniq), flush=True)
    s_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=s_["cfg"]["ch"], kernels=tuple(s_["cfg"]["kernels"]), dils=tuple(s_["cfg"]["dils"]), d_cond=s_["cfg"]["d_cond"]).to("cuda")
    gen.load_state_dict(s_["ema"]); gen.eval().requires_grad_(False)

    @torch.no_grad()
    def render(c: np.ndarray) -> np.ndarray:
        cond = torch.from_numpy(c.T.astype(np.float32))[None].to("cuda")
        lf, v = c[:, 25], c[:, 26]
        f0 = np.where(v > 0.5, 200.0 * np.exp(lf), 0.0).astype(np.float32)
        y = gen(cond, TR.excitation(torch.from_numpy(f0)[None].to("cuda"), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    for f in srcs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        c = np.fromfile(od / f"cv_{f.stem}.f32", "<f4").reshape(-1, 31).astype(np.float64)
        so = stats(np.fromfile(od / f"opcv_{pi:04d}_t{ti:03d}.f32", "<f4").reshape(-1, 31).astype(np.float64))
        st = stats(np.fromfile(od / f"ref_{ti:03d}.f32", "<f4").reshape(-1, 31).astype(np.float64))
        v = c[:, 26] > 0.5
        outs = {"B": c.copy()}
        for name, rows in (("MV", ROWS["ENV"] + ROWS["PER"] + ROWS["C0"]), ("MV_ENV", ROWS["ENV"]), ("MV_PER", ROWS["PER"])):
            y = c.copy()
            for m, sel in (("v", v), ("u", ~v)):
                if m in so and m in st:
                    for r in rows:
                        if r >= 27 and m == "u":
                            continue
                        y[sel, r] = (c[sel, r] - so[m][0][r]) / so[m][1][r] * st[m][1][r] + st[m][0][r]
            outs[name] = y
        for name, y in outs.items():
            sf.write(od / f"{k}_{name}.wav", render(y), N.SR)
    print("rendered", len(srcs) * 4, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""D214(学習なし・R を使わない・results/conv_p0/d214_prereg.yaml): d139 の経路の shimmer の床の帰属。
RUSTB = 製品の経路 G03S10Z + 較正 B を条件の書き出しつきで鳴らし直す・PYR = その条件を Python の出力部で鳴らす・
PYTB = d139 の経路の R なしの形(表 + 男声の c0f + Z・較正 B)・PYX = PYTB の c0 の輪郭だけを RUSTB の条件の c0 に替える(有声平均は PYTB に合わせる)。等化 H はどれにも掛けない。
    uv run python d214_path.py --s <dev>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm
from d114_env_bias_calib import BIN, CFG, EXTRA

ROOT = Path(__file__).resolve().parent.parent


def rust(j: tuple) -> None:
    cmd, dump = j
    env = dict(__import__("os").environ, LIGHTVC_DUMP_COND=dump)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--rvoc", default="results/rvtab_x/last.pt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--swap", action="store_true", help="D214 の OTHER の次(記述): PYTB の周期性の行だけ(PYP)・包絡の行だけ(PYE)を RUSTB の条件に替えた形だけを書く")
    a = ap.parse_args()
    import conv_c0 as C0
    import nvoc as N
    import rvoc as R
    import train_rvoc as TR
    S = Path(a.s)
    od = S / "d214"; od.mkdir(exist_ok=True)
    FR = S / "rr_real"
    work = S / "d85_work"
    cal = S / "d114/_cal_G03S10Z"
    shift = json.loads((S / "d114/_c0shift_G03S10Z.json").read_text())
    pairs = sorted((S / "d56").glob("p*_SRC.wav"))
    jobs = []
    for f in pairs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        if (od / f"{k}_RUSTB.wav").exists() and (od / f"{k}_RUSTB.f32").exists():
            continue
        cmd = [BIN, "artic", "--enroll", work / f"en_{ti:03d}", "--operator-enroll", work / f"op_{pi:04d}", "--mix", "0", "--per-p75", "--med-from-operator", "--warp", "--alpha", "1.3"]
        cmd += EXTRA + CFG["G03S10Z"] + ["--c0-shift", f"{shift[f'{pi}_{ti}']:.4f}", "-i", f, "-o", od / f"{k}_RUSTB.wav", "--env-bias", cal / f"bias_{pi:04d}_t{ti:03d}.f32"]
        jobs.append((cmd, str(od / f"{k}_RUSTB.f32")))
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(rust, jobs))
    same = []
    for f in pairs:
        k = f.stem[:-4]
        x, _ = sf.read(od / f"{k}_RUSTB.wav", dtype="float32"); y, _ = sf.read(S / "d114" / f"{k}_G03S10ZB.wav", dtype="float32")
        n = min(len(x), len(y))
        same.append(float(np.abs(x[:n] - y[:n]).max()))
    print("RUSTB vs G03S10ZB max|diff|", max(same), flush=True)

    dev = a.device
    sr_ = torch.load(ROOT / a.rvoc, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()

    @torch.no_grad()
    def render(c0n, env24, f0, per_t) -> np.ndarray:
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24 / 10, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def ld(corpus: str, stem: str) -> dict:
        man = {it["stem"]: it for it in json.loads((FR / corpus / "manifest.json").read_text())["items"]}
        d = torch.load(FR / man[stem]["pt"], weights_only=False)
        return {"p": RRm.dense(d["pk"], d["pv"]), "u": d["pk"][:, 0].long().numpy(), "e3": d["e3"].float().numpy(), "c0f": d["c0f"].numpy(), "f1": d["f1"].numpy(),
                "env": d["env"].float().numpy(), "ft": d["ft"].numpy(), "p4": d["p4"].float().numpy()}

    tgt = {}
    for t in range(30):
        r = ld("real_ref", f"ref_{t:03d}")
        vt = r["ft"] > 0
        tgt[t] = {"T": C0.table(r["env"][1:25], r["u"], RRm.K, Cb).astype(np.float32), "mu": float(np.median(np.log(r["ft"][vt]))), "p4": r["p4"][:, r["f1"] > 0].mean(1),
                  "c0T": float(((r["env"][0, vt] - RRm.C0_SIL) / 10).mean()), "eT": r["env"][1:25, vt].mean(1).astype(np.float64)}
    jit = []
    for f in pairs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        c = np.fromfile(od / f"{k}_RUSTB.f32", "<f4").reshape(-1, 31).astype(np.float64)
        vr = c[:, 26] > 0.5
        f0r = np.where(vr, 200.0 * np.exp(c[:, 25]), 0.0)
        sf.write(od / f"{k}_PYR.wav", render(c[:, 0], c[:, 1:25].T * 10, f0r, c[:, 27:31].T), N.SR)
        b = ld("d56_src", f.stem); o = ld("d56_op", f"op_{pi:04d}")
        tg = tgt[ti]
        vo, vM = o["f1"] > 0, b["f1"] > 0
        lfo = float(np.median(np.log(o["f1"][vo])))
        eop = tg["T"] @ o["p"]
        z = tg["c0T"] - float(o["c0f"][vo].mean())
        bias = tg["eT"] - eop[:, vo].mean(1)
        env = (tg["T"] @ b["p"]).astype(np.float64)
        env[:, vM] += bias[:, None]
        c0 = b["c0f"].astype(np.float64) + z
        f0o = np.where(vM, b["f1"] * np.exp(tg["mu"] - lfo), 0.0).astype(np.float64)
        per = (b["p4"] - o["p4"][:, vo].mean(1)[:, None] + tg["p4"][:, None]).astype(np.float64)
        n = min(len(c0), len(c))
        if a.swap:
            pp = per.copy(); pp[:, :n] = c[:n, 27:31].T
            ee = env.copy(); ee[:, :n] = c[:n, 1:25].T * 10
            sf.write(od / f"{k}_PYP.wav", render(c0, env, f0o, pp), N.SR)
            sf.write(od / f"{k}_PYE.wav", render(c0, ee, f0o, per), N.SR)
            continue
        sf.write(od / f"{k}_PYTB.wav", render(c0, env, f0o, per), N.SR)
        cx = c0.copy()
        vb = vM[:n] & vr[:n]
        cx[:n] = c[:n, 0] - c[:n, 0][vb].mean() + c0[:n][vb].mean()
        sf.write(od / f"{k}_PYX.wav", render(cx, env, f0o, per), N.SR)
        dj = lambda x, v: float(np.median(np.abs(np.diff(x))[v[1:] & v[:-1]]))
        jit.append({"key": k, "c0_rust": dj(c[:, 0], vr), "c0_py": dj(c0, vM), "lf_rust": dj(c[:, 25], vr), "lf_py": dj(np.log(np.maximum(f0o, 1.0)), vM),
                    "per_rust": dj(c[:, 27], vr), "per_py": dj(per[0], vM)})
        print(k, {kk: round(vv, 4) for kk, vv in jit[-1].items() if kk != "key"}, flush=True)
    if a.swap:
        return 0
    summ = {kk: round(float(np.median([j[kk] for j in jit])), 4) for kk in jit[0] if kk != "key"}
    (od / "jitter.json").write_text(json.dumps({"median": summ, "rust_vs_d114_maxdiff": max(same), "per_pair": jit}, indent=1))
    print("median", summ, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

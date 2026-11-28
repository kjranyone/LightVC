import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from p96_lag_probe import env_lags

KERNELS = {"K_now": "0.25,0.5,0.25,0", "K_half": "0.5,0.5,0,0", "K_q": "0.75,0.25,0,0", "K_zero": "0.5,0.75,0,-0.25", "K_none": "1,0,0,0"}
if os.environ.get("P96_KERNELS"):
    KERNELS = dict(kv.split("=") for kv in os.environ["P96_KERNELS"].split(";"))


def zz(w: np.ndarray, x: np.ndarray) -> list[float]:
    f, _, s = signal.stft(w, 48000, nperseg=480, noverlap=240, nfft=512, boundary=None, padded=False)
    p = np.abs(s) ** 2
    px = np.abs(signal.stft(x, 48000, nperseg=480, noverlap=240, nfft=512, boundary=None, padded=False)[2]) ** 2
    edges = np.geomspace(100, 8000, 17)
    b = np.stack([p[(f >= a) & (f < c)].sum(0) for a, c in zip(edges[:-1], edges[1:])])
    lv = 10 * np.log10(px.sum(0) + 1e-12)
    n = min(b.shape[1], len(lv))
    l, m = 10 * np.log10(b[:, :n] + 1e-10), (lv > lv.max() - 35)[:n]
    d2 = np.abs(l[:, 2:] - 2 * l[:, 1:-1] + l[:, :-2]).mean(0)
    m2 = m[2:] & m[1:-1] & m[:-2]
    return [round(float(d2[m2].mean()), 2), round(float(np.percentile(d2[m2], 95)), 2)]


def run(root: Path, d: Path, tg: Path, tag: str, env: dict[str, str]) -> np.ndarray:
    out = d / f"p96k_{tag}.f32"
    if not out.exists():
        e = dict(os.environ)
        e.update(env)
        subprocess.run([str(root / "target/release/examples/snv_trace"), str(d / "in.f32"), str(out), str(d / f"p96k_{tag}.csv"), "0", str(d / "op"), str(tg)], check=True, cwd=root, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.fromfile(out, dtype=np.float32)


def one(root: Path, work: Path, sp: str, name: str, taps: str) -> tuple[str, str, dict]:
    d = work / sp
    x = np.fromfile(d / "in.f32", dtype=np.float32)
    y = run(root, d, work / "tg", f"{name}_conv", {"LVC_EK": taps})
    yr = run(root, d, work / "tg", f"{name}_convraw", {"LVC_EK": taps, "LVC_NO_GUARD": "1"})
    yc = run(root, d, work / "tg", f"{name}_copy", {"LVC_EK": taps, "LVC_COPY": "1", "LVC_NO_GUARD": "1"})
    n = min(len(x), len(y))
    return sp, name, {"zz_conv": zz(y[:n], x[:n]), "zz_copy": zz(yc[:n], x[:n]), "zz_src": zz(x[:n], x[:n]), "lag_conv": env_lags(x[:n], y[:n])["all"], "lag_copy": env_lags(x[:n], yc[:n])["all"], "peak_conv_raw": round(float(np.abs(yr).max()), 3), "peak_copy_raw": round(float(np.abs(yc).max()), 3)}


def main() -> None:
    root, work, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    sps = sys.argv[4:]
    res: dict = {s: {} for s in sps}
    with ThreadPoolExecutor(4) as ex:
        for sp, name, r in ex.map(lambda a: one(root, work, *a), [(s, k, v) for k, v in KERNELS.items() for s in sps]):
            res[sp][name] = r
            print(sp, name, json.dumps(r), flush=True)
    summ = {}
    for k in KERNELS:
        rows = [res[s][k] for s in sps]
        summ[k] = {"zz95_conv": [q["zz_conv"][1] for q in rows], "zz95_copy_median": float(np.median([q["zz_copy"][1] for q in rows])), "zz95_src": [q["zz_src"][1] for q in rows], "lag_conv_median": float(np.median([q["lag_conv"][0] for q in rows])), "lag_copy_median": float(np.median([q["lag_copy"][0] for q in rows])), "peak_conv_raw_max": max(q["peak_conv_raw"] for q in rows)}
    res["summary"] = summ
    old = json.load(open(dst)) if dst.exists() else {}
    for s in sps:
        old.setdefault(s, {}).update(res[s])
    old.setdefault("summary", {}).update(summ)
    json.dump(old, open(dst, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

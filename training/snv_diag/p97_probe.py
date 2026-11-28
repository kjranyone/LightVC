import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from p96_f0_lag import f0 as pyin_f0
from p96_f0_lag import lag as f0_lag
from p96_kernel_probe import zz
from p96_lag_centroid import lag as cen_lag
from p96_lag_probe import env_lags, onset_lag

ROOT = Path(__file__).resolve().parent.parent.parent
IN = ROOT / "results/snv_p96/probe_in"
SPS = ["ca91ecce", "cb87e9f4", "cc5576e7", "cd094b9e"]


def trace(voc: str, sp: str, dst: Path, env: dict[str, str]) -> np.ndarray:
    if not dst.exists():
        e = dict(os.environ)
        e.update({"LVC_ESM": "1", **({"LVC_VOC": voc} if voc else {}), **env})
        subprocess.run([str(ROOT / "target/release/examples/snv_trace"), str(IN / sp / "in.f32"), str(dst), str(dst.with_suffix(".csv")), "0", str(IN / sp / "op"), str(IN / "tg")], check=True, cwd=ROOT, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.fromfile(dst, dtype=np.float32)


def phrase_at() -> tuple[str, int, int]:
    ph, sr = sf.read(ROOT / "results/snv_p95/ear3/phrase_esm.wav", dtype="float32")
    assert sr == 48000
    best = None
    for sp in SPS:
        y = np.fromfile(IN / sp / "product_conv.f32", np.float32)
        c = signal.fftconvolve(y, ph[::-1], mode="valid")
        j = int(np.abs(c).argmax())
        seg = y[j:j + len(ph)]
        r = float((seg * ph).sum() / (np.linalg.norm(seg) * np.linalg.norm(ph) + 1e-9))
        if best is None or r > best[0]:
            best = (r, sp, j)
    return best[1], best[2], len(ph)


def main() -> None:
    name, voc = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
    out = ROOT / "results/snv_p97/probe" / name
    out.mkdir(parents=True, exist_ok=True)
    psp, pj, pn = phrase_at()
    res: dict = {"name": name, "voc": voc, "speakers": {}}
    for sp in SPS:
        x = np.fromfile(IN / sp / "in.f32", dtype=np.float32)
        y = trace(voc, sp, out / f"{sp}_conv.f32", {})
        yr = trace(voc, sp, out / f"{sp}_convraw.f32", {"LVC_NO_GUARD": "1"})
        yc = trace(voc, sp, out / f"{sp}_copy.f32", {"LVC_COPY": "1", "LVC_NO_GUARD": "1"})
        n = min(len(x), len(y))
        d = np.concatenate([np.zeros(480, np.float32), x])[:len(x)]
        cal = cen_lag(x, d)[0]
        fx = pyin_f0(x, ROOT / "results/snv_p96/pyin" / f"{sp}_in.npy")
        r = {
            "onset": onset_lag(x[:n], y[:n])[0],
            "env_peak": env_lags(x[:n], y[:n])["all"][0],
            "env_centroid": round(cen_lag(x[:n], y[:n])[0] - cal + 10.0, 2),
            "env_centroid_copy": round(cen_lag(x[:n], yc[:n])[0] - cal + 10.0, 2),
            "f0": f0_lag(fx, pyin_f0(y[:n], out / f"{sp}_conv_pyin.npy"), False),
            "zz_conv": zz(y[:n], x[:n]),
            "zz_copy": zz(yc[:n], x[:n]),
            "peak_conv_raw": round(float(np.abs(yr).max()), 3),
        }
        if sp == psp:
            r["zz_phrase"] = zz(y[pj:pj + pn], y[pj:pj + pn])
            sf.write(out / "phrase.wav", y[pj:pj + pn], 48000, subtype="PCM_16")
        sf.write(out / f"{sp}.wav", y, 48000, subtype="PCM_16")
        res["speakers"][sp] = r
        print(sp, json.dumps(r), flush=True)
    rows = [res["speakers"][s] for s in SPS]
    res["summary"] = {
        "onset_median": float(np.median([q["onset"] for q in rows])),
        "env_peak_median": float(np.median([q["env_peak"] for q in rows])),
        "env_centroid_median": float(np.median([q["env_centroid"] for q in rows])),
        "env_centroid_copy_median": float(np.median([q["env_centroid_copy"] for q in rows])),
        "f0_median": float(np.median([q["f0"][0] for q in rows])),
        "zz95_conv": [q["zz_conv"][1] for q in rows],
        "zz95_copy": [q["zz_copy"][1] for q in rows],
        "zz95_phrase": res["speakers"][psp]["zz_phrase"][1],
        "peak_conv_raw_max": max(q["peak_conv_raw"] for q in rows),
    }
    json.dump(res, open(out / "probe.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

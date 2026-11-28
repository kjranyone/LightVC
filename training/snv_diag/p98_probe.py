import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from p96_f0_lag import f0 as pyin_f0
from p96_f0_lag import lag as f0_lag
from p96_kernel_probe import zz
from p96_lag_centroid import lag as cen_lag
from p96_lag_probe import env_lags, onset_lag
from p97_probe import IN, ROOT, SPS, phrase_at

import snv_out_guard as OG
import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_p1 as T

K3G = os.environ.get("SNV_PROBE_SYS", "ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch")


def utt(x: np.ndarray) -> dict:
    x = x[: P.n_frames(len(x)) * P.H].astype(np.float32)
    d = P.analyze(x)
    f0, env, per = d["f0"].astype(np.float32), d["env"].astype(np.float32), d["per"].astype(np.float32)
    s, n = T.voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    y = T.to_y(env, per, f0)
    m = y.shape[1]
    yc = np.concatenate([T.to_y(P.fixed_env(x, m), np.zeros((4, m), np.float32), np.ones(m, np.float32))[:25], y[25:]], 0).astype(np.float32)
    return {"y": y, "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0, "vs": s.astype(np.float32), "vn": n, "yc": yc, "wav": x}


def main() -> None:
    name = sys.argv[1]
    out = ROOT / "results/snv_p98/probe" / name
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda p: np.fromfile(p, np.float32)
    gen = EV.load_gen()
    S = EV.Ens(K3G, gen)
    sw = (P.WIN_CAP_MS, P.W_YIN, P.TAU_UV_HZ)
    if os.environ.get("SNV_PROBE_TGV3"):
        P.WIN_CAP_MS, P.W_YIN, P.TAU_UV_HZ = 1000.0, 1200, 200.0
    tg = [utt(rd(p)) for p in sorted((IN / "tg").glob("*.f32"))]
    P.WIN_CAP_MS, P.W_YIN, P.TAU_UV_HZ = sw
    st = S.structure("t", tg)
    p_out = OG.speech_level([u["wav"] for u in tg])
    psp, pj, pn = phrase_at()
    res: dict = {"name": name, "ck": os.environ.get("SNV_RCK", "rvoc3hi"), "speakers": {}}
    for sp in SPS:
        x = rd(IN / sp / "in.f32")
        u = utt(x)
        enr = [utt(rd(p)) for p in sorted((IN / sp / "op").glob("*.f32"))]
        w = S.convert(u, enr, st, 0)
        w = OG.apply(u["wav"], w, OG.speech_level([e["wav"] for e in enr]), p_out)
        y = np.concatenate([np.zeros(gen.delay, np.float32), w])[: len(x)]
        y = np.concatenate([y, np.zeros(len(x) - len(y), np.float32)])
        y.tofile(out / f"{sp}_conv.f32")
        n = min(len(x), len(y))
        d = np.concatenate([np.zeros(480, np.float32), x])[: len(x)]
        cal = cen_lag(x, d)[0]
        fx = pyin_f0(x, ROOT / "results/snv_p96/pyin" / f"{sp}_in.npy")
        r = {
            "onset": onset_lag(x[:n], y[:n])[0],
            "env_peak": env_lags(x[:n], y[:n])["all"][0],
            "env_centroid": round(cen_lag(x[:n], y[:n])[0] - cal + 10.0, 2),
            "f0": f0_lag(fx, pyin_f0(y[:n], out / f"{sp}_conv_pyin.npy"), False),
            "zz_conv": zz(y[:n], x[:n]),
        }
        if sp == psp:
            r["zz_phrase"] = zz(y[pj:pj + pn], y[pj:pj + pn])
            sf.write(out / "phrase.wav", y[pj:pj + pn], 48000, subtype="PCM_16")
        sf.write(out / f"{sp}.wav", y, 48000, subtype="PCM_16")
        res["speakers"][sp] = r
        print(sp, json.dumps(r), flush=True)
        for p in S.parts:
            p.pcache.clear()
    rows = [res["speakers"][s] for s in SPS]
    res["summary"] = {
        "onset_median": float(np.median([q["onset"] for q in rows])),
        "env_peak_median": float(np.median([q["env_peak"] for q in rows])),
        "env_centroid_median": float(np.median([q["env_centroid"] for q in rows])),
        "f0_median": float(np.median([q["f0"][0] for q in rows])),
        "zz95_conv": [q["zz_conv"][1] for q in rows],
        "zz95_phrase": res["speakers"][psp]["zz_phrase"][1],
    }
    json.dump(res, open(out / "probe.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res["summary"], ensure_ascii=False))


if __name__ == "__main__":
    with torch.no_grad():
        main()

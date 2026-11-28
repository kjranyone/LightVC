import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import p100_ct as CT
import pae as PA
import snv_dev_eval as DE
import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_p1 as T
from snv_a0_measure import to16

ROOT = Path(__file__).resolve().parent.parent.parent
FORMS = {"v3": (200.0, 3.0, 1000.0), "uv": (500.0, 3.0, 1000.0), "w25": (200.0, 2.5, 1000.0), "w2": (200.0, 2.0, 1000.0), "w15": (200.0, 1.5, 1000.0), "cap10": (200.0, 3.0, 10.0), "cap75": (200.0, 3.0, 7.5), "cap6": (200.0, 3.0, 6.0), "s2": ("ct", "sine2", 1e9), "s2lo200": ("ct", "sine2", 200.0), "s2lo150": ("ct", "sine2", 150.0), "t15lo200": ("ct", "trap150", 200.0), "t125lo200": ("ct", "trap125", 200.0), "t2lo200": ("ct", "trap200", 200.0), "k2lo200": ("ct", "tuk200", 200.0), "k175lo200": ("ct", "tuk175", 200.0), "k15lo200": ("ct", "tuk150", 200.0), "k125lo200": ("ct", "tuk125", 200.0), "kc75": ("ct", "tuk150", (200.0, 7.5)), "kc6": ("ct", "tuk150", (200.0, 6.0)), "kc5": ("ct", "tuk150", (200.0, 5.0)), "kc6all": ("ct", "tuk150", (1e9, 6.0))}
LAGS = np.arange(-3, 8)


def ref_env(x: np.ndarray, f0: np.ndarray) -> np.ndarray:
    fb, _ = PA._np_consts()
    xa = np.concatenate([np.zeros(P.PAD_L), x.astype(np.float64), np.zeros(PA.A + 4096)])
    t = np.arange(len(f0))
    sp = pyworld.cheaptrick(xa, np.where(f0 > 0, f0, 0.0).astype(np.float64), (P.PAD_L + t * P.H) / P.SR, P.SR, fft_size=PA.NFFT)
    return P.envelope_from_amp(np.sqrt(sp) @ fb.T)


def lag_ms(a: np.ndarray, r: np.ndarray, m: np.ndarray) -> tuple[float, int]:
    idx = np.nonzero(m)[0]
    idx = idx[(idx + LAGS.max() < a.shape[-1]) & (idx + LAGS.min() >= 0)]
    c = []
    for l in LAGS:
        u, v = a[..., idx + l], r[..., idx]
        u, v = u - u.mean(-1, keepdims=True), v - v.mean(-1, keepdims=True)
        c.append(float(((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12)).mean()))
    c = np.array(c)
    j = int(np.clip(c.argmax(), 1, len(c) - 2))
    den = c[j - 1] - 2 * c[j] + c[j + 1]
    return float((LAGS[j] + (0.5 * (c[j - 1] - c[j + 1]) / den if den < 0 else 0.0)) * 5.0), int(LAGS[c.argmax()])


def rough(e: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    k = (v[1:] > 0) & (v[:-1] > 0)
    d = np.abs(np.diff(e, axis=1))[:, k]
    return float(d[0].mean()), float(d[1:].mean())


def sm3(e: np.ndarray) -> np.ndarray:
    q = np.pad(e, ((0, 0), (2, 0)), mode="edge")
    return 0.25 * q[:, :-2] + 0.5 * q[:, 1:-1] + 0.25 * q[:, 2:]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    forms = {k: FORMS[k] for k in (sys.argv[2].split(",") if len(sys.argv) > 2 else FORMS)}
    draw = len(sys.argv) <= 3 or sys.argv[3] != "nodraw"
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    sets = {"devA": [it["u"]["wav_path"] for it in DE.cv_items("devA", n)]}
    vs = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"]
    sets["e1val"] = [str(ROOT / f"data/snv_e1/{v}/t03.wav") for v in vs]
    if draw:
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(EV.DEV).eval()
        mos = lambda w: float(utmos(torch.from_numpy(to16(w.astype(np.float64), EV.SR).astype(np.float32))[None].to(EV.DEV), 16000).item())
        gen = EV.load_gen()
    res: dict = {}
    for st, wavs in sets.items():
        acc = {k: {q: [] for q in ("lag_c0", "lag_c0_v", "lag_c0_uv", "lag_sh", "err", "r_c0", "r_sh", "r_c0_s", "r_sh_s", "utmos")} for k in forms}
        for wp in wavs:
            x = librosa.load(wp, sr=P.SR, mono=True)[0].astype(np.float32)
            x = x[: P.n_frames(len(x)) * P.H]
            d = P.analyze(x)
            f0 = d["f0"].astype(np.float64)
            v = (f0 > 0).astype(np.float32)
            ref = ref_env(x, f0)
            act = ref[0] > np.percentile(ref[0], 30)
            for k, (uvhz, wk, cap) in forms.items():
                if uvhz == "ct":
                    env = CT.envelope(x, f0, wk, *cap) if isinstance(cap, tuple) else CT.envelope(x, f0, wk, cap)
                else:
                    P.TAU_UV_HZ, P.WIN_K, P.WIN_CAP_MS = uvhz, wk, cap
                    env = P.envelope(x, f0)
                    P.TAU_UV_HZ, P.WIN_K, P.WIN_CAP_MS = 200.0, 3.0, 1000.0
                a = acc[k]
                l0, j0 = lag_ms(env[:1], ref[:1], act)
                a["lag_c0"].append(l0)
                a["lag_c0_v"].append(lag_ms(env[:1], ref[:1], act & (v > 0))[0])
                a["lag_c0_uv"].append(lag_ms(env[:1], ref[:1], act & (v == 0))[0] if (act & (v == 0)).sum() > 30 else np.nan)
                a["lag_sh"].append(lag_ms(env[1:9], ref[1:9], act)[0])
                j = max(j0, 0)
                m = env.shape[1] - j
                a["err"].append(float(np.abs(env[:, j:] - ref[:, :m])[:, act[:m]].mean()))
                r0, r1 = rough(env, v)
                s0, s1 = rough(sm3(env), v)
                a["r_c0"].append(r0)
                a["r_sh"].append(r1)
                a["r_c0_s"].append(s0)
                a["r_sh_s"].append(s1)
                if not draw:
                    a["utmos"].append(0.0)
                    continue
                y = T.to_y(env.astype(np.float32), d["per"].astype(np.float32), d["f0"].astype(np.float32))
                lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
                cond = torch.from_numpy(np.concatenate([y[:25], lf[None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0))[None].to(EV.DEV)
                cond = torch.cat([EV.smooth_env(gen, cond[:, :25]), cond[:, 25:]], 1)
                a["utmos"].append(mos(EV.draw(gen, cond, torch.from_numpy(d["f0"].astype(np.float32))[None].to(EV.DEV), None)))
        out = {}
        for k in forms:
            a, b = acc[k], acc["v3"]
            du = np.array(a["utmos"]) - np.array(b["utmos"])
            out[k] = {q: round(float(np.nanmean(a[q])), 3) for q in ("lag_c0", "lag_c0_v", "lag_c0_uv", "lag_sh", "err")}
            out[k] |= {q + "_ratio": round(float(np.mean(a[q]) / np.mean(b[q])), 3) for q in ("r_c0", "r_sh", "r_c0_s", "r_sh_s")}
            out[k] |= {"utmos": round(float(np.mean(a["utmos"])), 4), "utmos_vs_v3": [round(float(du.mean()), 4), round(float(1.96 * du.std() / np.sqrt(len(du))), 4)]}
            print(st, k, json.dumps(out[k]), flush=True)
        res[st] = out
    json.dump(res, open(ROOT / f"results/snv_p100/front{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    with torch.no_grad():
        main()

from __future__ import annotations

import os

os.environ.setdefault("SNV_WIN_CAP_MS", "7.5")
os.environ.setdefault("SNV_W_YIN", "600")
os.environ.setdefault("SNV_TAU_UV", "500")
os.environ.setdefault("SNV_ENVC", "1")
os.environ.setdefault("SNV_ESM", "1")
os.environ.setdefault("SNV_WF16", "1")
os.environ.setdefault("SNV_KV16", "1")

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from pesq import pesq
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import p119_copy as C9
import snv_out_guard as OG
import snv_p0_front as P
import snv_p125_feat as F125
import snv_p1_eval as EV
import train_snv_p1 as T
from p96_f0_lag import f0 as pyin_f0
from p96_f0_lag import lag as f0_lag
from p96_kernel_probe import zz
from p96_lag_centroid import lag as cen_lag
from p96_lag_probe import env_lags, onset_lag
from p97_probe import IN, SPS, phrase_at

ROOT = EV.ROOT
VOICES = ("ef100", "ef032", "ef300")
MEM_P125 = ("snv_p125a_c", "snv_p125b_c", "snv_p125e_c")
G_BASE = "+mask:snv_p45_gate_t80+fb:anch"
G_C0BG = "+mask:snv_p45_gate_t80+fb:c0bg0.45_0.15,anch"
SYS_P125_ANCH = "ens:" + "|".join(f"{t}{G_BASE}" for t in MEM_P125)
SYS_P125_C0BG = "ens:" + "|".join(f"{t}{G_C0BG}" for t in MEM_P125)


def raw_utt(x: np.ndarray, use_v4: bool = True) -> dict:
    x = x[: P.n_frames(len(x)) * P.H].astype(np.float32)
    d = F125.analyze_v4(x) if use_v4 else P.analyze(x)
    f0, env, per = d["f0"].astype(np.float32), d["env"].astype(np.float32), d["per"].astype(np.float32)
    s, n = T.voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    y = T.to_y(env, per, f0)
    m = y.shape[1]
    yc = np.concatenate([T.to_y(P.fixed_env(x, m), np.zeros((4, m), np.float32), np.ones(m, np.float32))[:25], y[25:]], 0).astype(np.float32)
    return {"y": y, "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0, "vs": s.astype(np.float32), "vn": n, "yc": yc, "wav": x}


def wb_pesq_opt(x: np.ndarray, y: np.ndarray) -> float:
    x16 = resample_poly(x.astype(np.float64), 1, 3)
    best = -9.0
    for sh in (-240, -180, -120, -60, 0, 60, 120, 180, 240, 300):
        if sh >= 0:
            n = min(len(x), len(y) - sh)
            xs, ys = x[:n], y[sh : sh + n]
        else:
            s = -sh
            n = min(len(x) - s, len(y))
            xs, ys = x[s : s + n], y[:n]
        y16 = resample_poly(ys.astype(np.float64), 1, 3)
        m = min(len(x16), len(y16))
        best = max(best, float(pesq(16000, x16[:m], y16[:m], "wb")))
    return round(best, 4)


def d2_val(z: torch.Tensor) -> float:
    return float((z[..., 2:] - 2.0 * z[..., 1:-1] + z[..., :-2]).abs().mean())


@torch.no_grad()
def run_p120_arm(arm_name: str, rck: str, sys_tag: str) -> dict:
    os.environ["SNV_RCK"] = rck
    os.environ["SNV_F0V4"] = "1"
    gen = EV.load_gen()
    S = EV.Ens(sys_tag, gen)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    pesq_rows: dict[str, list[float]] = {k: [] for k in ("T", "K", "Kc0", "Kenv", "Kper")}
    lm_rows: dict[str, list[float]] = {k: [] for k in ("T", "K", "Kc0", "Kenv", "Kper")}
    vstats = {}
    for v in VOICES:
        refs = [raw_utt(lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav"), use_v4=True) for j in (0, 1, 2)]
        st = S.structure("t" + v, refs)
        x_tgt = lw(ROOT / f"data/snv_e1/{v}/t05.wav")
        d_h = np.load(ROOT / f"data/snv_p122_feat/e1/{v}/t05.npz")
        fh = d_h["f0"].astype(np.float32)
        if getattr(gen, "delay", 240) == 480:
            fh = np.concatenate([fh[:1], fh[:-1]])
            per_t = np.concatenate([d_h["per"][:, :1], d_h["per"][:, :-1]], 1)
        else:
            per_t = d_h["per"]
        y_tgt_np = T.to_y(d_h["env"], per_t, fh)
        u = raw_utt(lw(ROOT / f"data/snv_p_wav_x16/e1/{v}/t05.wav"), use_v4=True)
        enr = [raw_utt(lw(ROOT / f"data/snv_p_wav_x16/e1/{v}/t0{j}.wav"), use_v4=True) for j in (0, 1, 2)]
        med = lambda us: float(np.median(np.concatenate([q["f0"][q["f0"] > 0] for q in us])))
        pu_enr = [S.parts[0].prep(q) for q in enr]
        k_st = float(12.0 * np.log2(med(refs) / med(pu_enr)))
        _, yo = S.convert(u, enr, st, k_st, with_y=True)
        n = min(yo.shape[-1], y_tgt_np.shape[1], len(fh))
        Km = yo[:, :29, :n].float()
        Tm = torch.from_numpy(y_tgt_np[:29, :n])[None].to(EV.DEV)
        f0 = fh[:n]
        vv = torch.from_numpy((f0 > 0).astype(np.float32))[None].to(EV.DEV)
        lf = torch.from_numpy(np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32))[None].to(EV.DEV)
        sm_k = EV.smooth_env(gen, Km[:, :25])
        sm_t = EV.smooth_env(gen, Tm[:, :25])
        vstats[v] = {
            "c0_l1_raw": round(float((Km[0, 0] - Tm[0, 0]).abs().mean()), 4),
            "c0_l1_sm": round(float((sm_k[0, 0] - sm_t[0, 0]).abs().mean()), 4),
            "c0_d2_raw_K_T": [round(d2_val(Km[0, :1]), 4), round(d2_val(Tm[0, :1]), 4)],
            "c0_d2_sm_K_T": [round(d2_val(sm_k[0, :1]), 4), round(d2_val(sm_t[0, :1]), 4)],
            "c0_d2_sm_ratio": round(d2_val(sm_k[0, :1]) / max(d2_val(sm_t[0, :1]), 1e-9), 3),
        }
        mix = {
            "T": Tm,
            "K": Km,
            "Kc0": torch.cat([Km[:, :1], Tm[:, 1:]], 1),
            "Kenv": torch.cat([Tm[:, :1], Km[:, 1:25], Tm[:, 25:]], 1),
            "Kper": torch.cat([Tm[:, :25], Km[:, 25:]], 1),
        }
        for mk, ym in mix.items():
            cond = torch.cat([EV.smooth_env(gen, ym[:, :25]), lf[:, None], vv[:, None], ym[:, 25:29].clamp(0, 1.5) * vv[:, None]], 1)
            w = np.asarray(EV.draw(gen, cond, torch.from_numpy(f0)[None].to(EV.DEV), None), np.float32)
            pesq_rows[mk].append(wb_pesq_opt(x_tgt[: n * P.H], w))
            lm_rows[mk].append(round(C9.lm(x_tgt[: n * P.H], w), 4))
    return {
        "arm": arm_name,
        "pesq": {k: {"per_voice": v, "mean": round(float(np.mean(v)), 4)} for k, v in pesq_rows.items()},
        "logmel": {k: {"per_voice": v, "mean": round(float(np.mean(v)), 4)} for k, v in lm_rows.items()},
        "stats": vstats,
        "d2_sm_ratio_max": max(s["c0_d2_sm_ratio"] for s in vstats.values()),
    }


@torch.no_grad()
def run_probe4_arm(arm_name: str, rck: str, sys_tag: str, out_dir: Path) -> dict:
    os.environ["SNV_RCK"] = rck
    os.environ["SNV_F0V4"] = "1"
    sub = out_dir / arm_name
    sub.mkdir(parents=True, exist_ok=True)
    rd = lambda p: np.fromfile(p, np.float32)
    gen = EV.load_gen()
    S = EV.Ens(sys_tag, gen)
    tg = [raw_utt(rd(p), use_v4=True) for p in sorted((IN / "tg").glob("*.f32"))]
    st = S.structure("t", tg)
    p_out = OG.speech_level([u["wav"] for u in tg])
    psp, pj, pn = phrase_at()
    sp_res = {}
    for sp in SPS:
        x = rd(IN / sp / "in.f32")
        u = raw_utt(x, use_v4=True)
        enr = [raw_utt(rd(p), use_v4=True) for p in sorted((IN / sp / "op").glob("*.f32"))]
        w = S.convert(u, enr, st, 0)
        w = OG.apply(u["wav"], w, OG.speech_level([e["wav"] for e in enr]), p_out)
        lag_pad = 480 if getattr(gen, "delay", 240) == 480 else int(gen.delay)
        y = np.concatenate([np.zeros(lag_pad, np.float32), w])[: len(x)]
        y = np.concatenate([y, np.zeros(max(0, len(x) - len(y)), np.float32)])
        n = min(len(x), len(y))
        d = np.concatenate([np.zeros(480, np.float32), x])[: len(x)]
        cal = cen_lag(x, d)[0]
        fx = pyin_f0(x, ROOT / "results/snv_p96/pyin" / f"{sp}_in.npy")
        r = {
            "onset": onset_lag(x[:n], y[:n])[0],
            "env_peak": env_lags(x[:n], y[:n])["all"][0],
            "env_centroid": round(cen_lag(x[:n], y[:n])[0] - cal + 10.0, 2),
            "f0": f0_lag(fx, pyin_f0(y[:n], sub / f"{sp}_conv_pyin.npy"), False),
            "zz_conv": zz(y[:n], x[:n]),
        }
        if sp == psp:
            r["zz_phrase"] = zz(y[pj : pj + pn], y[pj : pj + pn])
        sp_res[sp] = r
        for p in S.parts:
            p.pcache.clear()
    rows = [sp_res[s] for s in SPS]
    return {
        "onset_median": float(np.median([q["onset"] for q in rows])),
        "env_peak_median": float(np.median([q["env_peak"] for q in rows])),
        "env_centroid_median": float(np.median([q["env_centroid"] for q in rows])),
        "f0_median": float(np.median([q["f0"][0] for q in rows])),
        "zz95_conv": [round(float(q["zz_conv"][1]), 3) for q in rows],
        "zz95_phrase": round(float(sp_res[psp]["zz_phrase"][1]), 3),
    }


def main() -> None:
    out_dir = ROOT / "results/snv_p125"
    out_dir.mkdir(parents=True, exist_ok=True)
    rck = "results/snv_p123_s/snap/ema_20k.pt"
    arms = [
        ("P125_c0bg", rck, SYS_P125_C0BG),
        ("P125_anch", rck, SYS_P125_ANCH),
    ]
    res: dict = {"p120": {}, "probe4": {}}
    prev = ROOT / "results/snv_p124/step_b_probes.json"
    if prev.exists():
        pr = json.loads(prev.read_text())
        for k in ("B0_oldA", "R3_v4_c0bg_p123s"):
            if k in pr.get("p120", {}):
                res["p120"][k] = pr["p120"][k]
            if k in pr.get("probe4", {}):
                res["probe4"][k] = pr["probe4"][k]
    for name, ck, stg in arms:
        r = run_p120_arm(name, ck, stg)
        res["p120"][name] = r
        print("P120", name, "PESQ", {k: r["pesq"][k]["mean"] for k in r["pesq"]}, "d2_ratio_max", r["d2_sm_ratio_max"], "c0_l1_sm", [r["stats"][v]["c0_l1_sm"] for v in VOICES], flush=True)
    for name, ck, stg in arms:
        p = run_probe4_arm(name, ck, stg, out_dir / "probe")
        res["probe4"][name] = p
        print("PROBE4", name, json.dumps(p, ensure_ascii=False), flush=True)
    (out_dir / "eval_probes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""D637: 構音の包絡の内容の網(train_envnet.py・diag_envnet1)を製品の Rust の条件で鳴らす(開発の組 dev2 / dev3・PT と同じ描画)。
Rust を LIGHTVC_DUMP_RCOND つきで鳴らし、文脈の符号器の前の生の条件(c0n・表の引き + H1 EQ・log f0・有声・周期性・事後 K・C2)と符号器の後の包絡を書き出す。
EN = 表の引き + 網・ENR = EN + 遅い揺らぎ + R の ΔS(D634 の実効の行 − 符号器の後の行)・TBR = 表の引き + 遅い揺らぎ + R の ΔS(網も符号器も無し)。規則は results/conv_p0/d637/prereg.yaml。
    uv run python d637_envnet_eval.py --stage rust
    uv run python d637_envnet_eval.py --stage render
    uv run python d637_envnet_eval.py --stage rough
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d632_f0env_swap import fit, lag, shift, warp
from d634_artic_render import BIN, FQ, PRE, env_from_rows, synth, vdir
from d635_split import norm

OUT = ROOT / "results/conv_p0/d637"
NET = ROOT / "results/diag_envnet1/last.pt"
ARMS = ("EN", "ENR", "TBR")
DC = 295


def rdir(s: str, k: int) -> Path:
    return H / s / "d637_rcond" / f"k{k}"


def rust(job: tuple) -> str:
    s, k, key = job
    S = H / s
    pi, ti = int(key[1:5]), int(key[7:10])
    rf = rdir(s, k) / f"{key}.rcond16"
    if rf.exists():
        return "skip"
    opw = S / "d56" / "_work" / f"op_{pi:04d}.wav"
    cmd = [str(BIN), "artic", "--enroll", str(S / "d85_work" / f"en_{ti:03d}"), "--operator-enroll", str(S / "d85_work" / f"op_{pi:04d}"),
           "--mix", "1", "--causal-ct", "--per-p75", "--f4", "results/f4_1/export", "--rvoc", "results/diag_ppgvoc58/export", "--c2", "results/c2_1/export",
           "--opslow", "4,0", "--c0-ppg-ref", str(opw), "--semitones", str(k), "--rr", "results/d587_rbar/export", "--rr-mem", str(S / "rr_mem" / f"en_{ti:03d}"),
           "--rr-op-ref", str(opw), "--rr-stride", "2"] + (["--h1eq", "3"] if k in (5, 7) else [])
    rf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp, rc = Path(td) / "y.wav", Path(td) / "x.rcond"
        r = subprocess.run(cmd + ["-i", str(S / "d467" / f"{key}.wav"), "-o", str(tmp)], cwd=ROOT, capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "LIGHTVC_DUMP_RCOND": str(rc), "LVC_SPIN_US": "0", "LVC_NO_PIN": "1"})
        if r.returncode != 0:
            return f"FAIL {s} {k} {key} {r.stderr[-200:]}"
        x = np.fromfile(rc, "<f4").reshape(-1, DC + 24)
    x.astype(np.float16).tofile(rf)
    return "ok"


_NET = None


def net():
    global _NET
    if _NET is None:
        torch.set_num_threads(1)
        from train_envnet import EnvNet
        st = torch.load(NET, map_location="cpu", weights_only=False)
        _NET = EnvNet(st["cfg"]["ch"], tuple(st["cfg"]["dils"]), st["cfg"]["d_c2"]).eval()
        _NET.load_state_dict(st["net"])
    return _NET


def env_rows(s: str, k: int, key: str) -> dict:
    rc = np.fromfile(rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, DC + 24).astype(np.float32)
    vc = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)
    assert len(rc) == len(vc), (len(rc), len(vc))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T))[None]
    with torch.no_grad():
        y = net()(t(rc[:, 31:31 + 200]), t(rc[:, 0:1]), t(rc[:, 25:26]), t(rc[:, 26:27]), t(rc[:, 1:25]), t(rc[:, 231:DC]))[0].numpy().T
    post = vc[:, 1:25] - rc[:, DC:]
    sl = slice(PRE, PRE + 1600)
    return {"EN": y[sl], "ENR": (y + post)[sl], "TBR": (rc[:, 1:25] + post)[sl], "PT": vc[sl, 1:25], "raw": rc[sl, 1:25]}


def render(it: tuple) -> dict:
    s, k, key = it
    d = H / s / "d570"
    from d6_unit_cov import load48
    v = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)[PRE:PRE + 1600].astype(np.float64)
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    lg = lag(xc, xs)
    _, sps, aps = (shift(m, lg) for m in analyze(xs))
    n = len(v)
    apg = fit(warp(aps, G[k][1]), n)
    pin = fit(sps, n).mean(1, keepdims=True)
    f0 = np.where(v[:, 26] > 0.5, 200.0 * np.exp(v[:, 25]), 0.0)
    er = env_rows(s, k, key)
    rec = {"key": f"{s}|{k}|{key}", "pt_match": float(np.abs(er["PT"] - v[:, 1:25]).max())}
    vo = f0 > 0
    rec["delta_rms_EN_raw"] = float(np.sqrt(((er["EN"] - er["raw"])[vo] ** 2).mean()))
    rec["delta_rms_PT_raw"] = float(np.sqrt(((er["PT"] - er["raw"])[vo] ** 2).mean()))
    for a in ARMS:
        o = d / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            sf.write(o, synth(f0, norm(env_from_rows(er[a].astype(np.float64)), pin), apg), 48000, subtype="PCM_16")
    if s == "dev2" and k == 7 and key in keys("dev2")[:12]:
        r = er["EN"].astype(np.float64)
        yt = synth(f0, norm(env_from_rows(r), pin), apg)
        ya = synth(f0, norm(env_from_rows(r, 1.1), pin), apg)
        rec["alpha1_identity"] = bool(np.array_equal(env_from_rows(r, 1.0), env_from_rows(r)))
        _, s0, _ = analyze(yt.astype(np.float64))
        _, s1, _ = analyze(ya.astype(np.float64))
        lf = np.linspace(np.log(300.0), np.log(5000.0), 400)
        a0 = np.log(np.stack([np.interp(np.exp(lf), FQ, q) for q in s0]) + 1e-20)
        a1 = np.log(np.stack([np.interp(np.exp(lf), FQ, q) for q in s1]) + 1e-20)
        sv = np.flatnonzero(f0[: min(len(a0), len(a1))] > 0)
        step = lf[1] - lf[0]
        sh = []
        for t in sv[::4]:
            x0, x1 = a0[t] - a0[t].mean(), a1[t] - a1[t].mean()
            cc = [float(np.dot(x0[max(0, -m):len(x0) - max(0, m)], x1[max(0, m):len(x1) - max(0, -m)])) for m in range(-30, 31)]
            sh.append(np.exp((int(np.argmax(cc)) - 30) * step))
        rec["alpha11_ratio_med"] = float(np.median(sh)) if sh else float("nan")
    return rec


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["rust", "render", "rough"])
    a = ap_.parse_args()
    npr = int(os.environ.get("LVC_NPROC", "12"))
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    if a.stage == "rust":
        with Pool(int(os.environ.get("LVC_RUST_NPROC", "8"))) as pl:
            res = list(pl.imap_unordered(rust, its))
        print({x: sum(r.startswith(x) for r in res) for x in ("ok", "skip", "FAIL")}, [r for r in res if r.startswith("FAIL")][:3], flush=True)
        return 0
    if a.stage == "render":
        with Pool(npr) as pl:
            recs = pl.map(render, its, chunksize=2)
        (OUT / "render_rec.json").write_text(json.dumps(recs, indent=0))
        kn = [r for r in recs if "alpha11_ratio_med" in r]
        print("rendered", len(recs), "PT 再現の最大の差(float16)", round(max(r["pt_match"] for r in recs), 4),
              "行のずれの RMS 中央値 EN − 表", round(float(np.median([r["delta_rms_EN_raw"] for r in recs])), 4), "PT − 表", round(float(np.median([r["delta_rms_PT_raw"] for r in recs])), 4),
              "knob alpha1 identity", all(r["alpha1_identity"] for r in kn), "alpha1.1 ratio", [round(r["alpha11_ratio_med"], 3) for r in kn], flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0], "PT": f"PT_{k}"} | {x: f"{x}_{k}" for x in ARMS}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d634/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("PT",) + ARMS:
        rep[arm] = {}
        for k in C:
            rep[arm][k] = {}
            nm = sysn(k)
            for f, (_, hi) in FEATS.items():
                xs = np.array([res[str(H / s / "d570" / nm[arm] / f"{key}_{nm[arm]}.wav")][f] for s, kk, key in its if kk == k])
                gs = np.array([res[str(H / s / "d570" / nm["G"] / f"{key}_{nm['G']}.wav")][f] for s, kk, key in its if kk == k])
                m = np.isfinite(xs) & np.isfinite(gs)
                gb = float(np.mean(gs[m] > xs[m])) if hi else float(np.mean(gs[m] < xs[m]))
                rep[arm][k][f] = {"G_better": round(gb, 3), "median_X": round(float(np.median(xs[m])), 4), "median_G": round(float(np.median(gs[m])), 4)}
            print(arm, k, {f: (v["G_better"], v["median_X"], v["median_G"]) for f, v in rep[arm][k].items()}, flush=True)
        rep[arm]["flutter_guard"] = bool(all(rep[arm][k]["pflutter"]["G_better"] <= 0.60 for k in C))
        rep[arm]["parity4"] = bool(all(rep[arm][k][f]["G_better"] <= 0.60 for k in C for f in FEATS))
    (OUT / "rough.json").write_text(json.dumps(rep, indent=1))
    print("flutter guard", {arm: rep[arm]["flutter_guard"] for arm in rep}, "parity4", {arm: rep[arm]["parity4"] for arm in rep}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

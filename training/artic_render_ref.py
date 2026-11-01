"""構音の描画(基準 PR5 + 大きさの音節の動きのつまみ γ・D669)の Python の参照(Rust crates/lightvc-core/src/artic_render.rs と同じ式・同じ流れ)と、Rust との出力の一致の検査。
Rust を --render-artic と LIGHTVC_DUMP_RCOND つきで鳴らし、書き出した生の条件 [295] + 遅い揺らぎ + R の ΔS を足した行 [24] から Python で同じ出力を作って比べる。
雑音は splitmix64 + Box-Muller の共通の乱数(種 0・Rust の rvoc.rs splitmix_u と同じ)。
    uv run python artic_render_ref.py --set dev2 --k 7 --n 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import pae as PA
from causal_synth import CausalSynth
from d634_artic_render import BIN, DCT, FC, FQ
from d636_traj_gain import onepole
from d639_knob_hf import ap_from_per
from d631_world_swap import H, ROOT, keys
from train_envnet import EnvNet
from train_ppgvoc import splitmix_normals

EXPORT = ROOT / "results/artic_render2/export"


class SplitmixRng:
    def __init__(self, seed: int):
        self.seed, self.q = seed, 0

    def standard_normal(self, n: int) -> np.ndarray:
        out = splitmix_normals(self.seed, n, self.q)
        self.q += n
        return out


def level(c0n: np.ndarray, c30: np.ndarray, cfg: dict, gamma: float) -> np.ndarray:
    al = float(np.exp(-2 * np.pi * cfg.get("level_hz", 2.0) * cfg["hop"] / cfg["sr"]))
    out = c30.copy()
    s = None
    for t in range(len(c30)):
        if c0n[t] <= cfg.get("level_thr", 0.05):
            continue
        s = c30[t] if s is None else al * s + (1 - al) * c30[t]
        out[t] = c30[t] if gamma >= 1.0 else s + gamma * (c30[t] - s)
    return out


def load_export() -> tuple:
    man = json.loads((EXPORT / "manifest.json").read_text())
    w = np.fromfile(EXPORT / "weights.bin", "<f4")
    get = lambda n: w[man["tensors"][n][0]:man["tensors"][n][0] + int(np.prod(man["tensors"][n][1]))].reshape(man["tensors"][n][1])
    st = torch.load(man["cfg"]["net"], map_location="cpu", weights_only=False)
    net = EnvNet(st["cfg"]["ch"], tuple(st["cfg"]["dils"]), st["cfg"]["d_c2"]).eval()
    net.load_state_dict(st["net"])
    return net, get("curve").astype(np.float64), man["cfg"]


def render_ref(rc: np.ndarray, alpha: float = 1.0, gamma: float | None = None) -> np.ndarray:
    net, curve, cfg = load_export()
    k, cv = cfg["k"], cfg["cv"]
    raw, after = rc[:, :31 + k + cv].astype(np.float64), rc[:, 31 + k + cv:].astype(np.float64)
    post = after - raw[:, 1:25]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T.astype(np.float32)))[None]
    with torch.no_grad():
        y = net(t(raw[:, 31:31 + k]), t(raw[:, 0:1]), t(raw[:, 25:26]), t(raw[:, 26:27]), t(raw[:, 1:25]), t(raw[:, 31 + k:]))[0].numpy().T.astype(np.float64)
    tab = raw[:, 1:25]
    d = y - tab
    hop, sr = cfg["hop"], cfg["sr"]
    a_hp = float(np.exp(-hop / sr / cfg["hp_tau_s"]))
    a = lambda hz: float(np.exp(-2 * np.pi * hz * hop / sr))
    rows = tab + d - onepole(d, a_hp) + post
    c0s = onepole((10.0 * raw[:, 0] + cfg["c0_sil"])[:, None], a(cfg["c0_hz"]))[:, 0]
    c0s = level(rc[:, 0].astype(np.float32), c0s, cfg, cfg.get("level_gamma", 1.0) if gamma is None else gamma)
    v = raw[:, 26] > 0.5
    lf = np.where(v, raw[:, 25], 0.0)
    lfs = lf.copy()
    al = a(cfg["lf_hz"])
    for i in range(1, len(lf)):
        if v[i] and v[i - 1]:
            lfs[i] = al * lfs[i - 1] + (1.0 - al) * lf[i]
    f0 = np.where(v, 200.0 * np.exp(lfs), 0.0)
    lm = (rows * 10.0) @ DCT[1:25]
    e = np.exp(2.0 * np.stack([np.interp(FQ, FC * alpha, r) for r in lm]))
    s = e.mean(1)
    sb = np.exp(onepole(np.log(s)[:, None], a(cfg["shape_hz"]))[:, 0])
    sp = e / s[:, None] * sb[:, None] * np.exp(2.0 * c0s / np.sqrt(128.0))[:, None] * curve
    ap = ap_from_per(raw[:, 27:31], f0)
    cs = CausalSynth(seed=0)
    cs.rng = SplitmixRng(0)
    return np.concatenate([cs.block(float(f0[i]), sp[i], ap[i]) for i in range(len(f0))])


def run_rust(s: str, k: int, key: str, td: Path) -> tuple:
    S = H / s
    pi, ti = int(key[1:5]), int(key[7:10])
    opw = S / "d56" / "_work" / f"op_{pi:04d}.wav"
    cmd = [str(BIN), "artic", "--enroll", str(S / "d85_work" / f"en_{ti:03d}"), "--operator-enroll", str(S / "d85_work" / f"op_{pi:04d}"),
           "--mix", "1", "--causal-ct", "--per-p75", "--f4", "results/f4_1/export", "--rvoc", "results/diag_ppgvoc58/export", "--c2", "results/c2_1/export",
           "--opslow", "4,0", "--c0-ppg-ref", str(opw), "--semitones", str(k), "--rr", "results/d587_rbar/export", "--rr-mem", str(S / "rr_mem" / f"en_{ti:03d}"),
           "--rr-op-ref", str(opw), "--rr-stride", "2", "--render-artic", str(EXPORT), "-i", str(S / "d467" / f"{key}.wav"), "-o", str(td / "y.wav")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "LIGHTVC_DUMP_RCOND": str(td / "x.rcond"), "LVC_SPIN_US": "0", "LVC_NO_PIN": "1"})
    assert r.returncode == 0, r.stderr[-500:]
    y, _ = sf.read(td / "y.wav", dtype="float64")
    return y, np.fromfile(td / "x.rcond", "<f4").reshape(-1, 295 + 24)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev2")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results/artic_render1/parity.json"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    rep = []
    for key in keys(a.set)[: a.n]:
        with tempfile.TemporaryDirectory() as td:
            yr, rc = run_rust(a.set, a.k, key, Path(td))
        yp = render_ref(rc)
        n = min(len(yr), len(yp))
        yr, yp = yr[:n], yp[:n]
        g = float(np.dot(yr, yp) / (np.dot(yp, yp) + 1e-20))
        err = yr - g * yp
        snr = float(10 * np.log10(np.dot(yr, yr) / (np.dot(err, err) + 1e-30)))
        rep.append({"key": key, "k": a.k, "frames": int(len(rc)), "scale": round(g, 6), "snr_db": round(snr, 2), "max_abs_err": float(np.abs(err).max()), "rms": float(np.sqrt((yr ** 2).mean()))})
        print(rep[-1], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

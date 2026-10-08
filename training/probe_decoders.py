"""ステップ1(0学習): decoder 候補 c32(基準)/A(因果AA)/B(A2思想) の出荷予算プローブ。

測定: ストリーミング一致(full vs 1frame step)・未来不変性(frame t 以降の入力改変で t 以前の出力が
ビット一致)・追加遅延(A: c32出力との相互相関)・パラメータ数・MAC/frame・CPU 1thread の
decode_step 時間(warm-up後 N step・p50/p95/p99/max・mean RTF)。PyTorch eager 値は Rust より重いので
c32 の同条件値を較正用に併記する(c32 Rust 実測 p95 3.09ms)。

    CUDA_VISIBLE_DEVICES= uv run python probe_decoders.py --steps 10000
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, HOP_LENGTH, SAMPLE_RATE
from decoder_aa import DecoderAA
from decoder_a2 import DecoderA2, A2_DILS, A2_KS
from train_d1 import build_index
from train_cfmys import F0FIX, F0_FPS, LAT_FPS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/ys1_dec2"


def f0_frames(f0_raw: torch.Tensor, T: int) -> torch.Tensor:
    i_f = ((torch.arange(T, dtype=torch.float64) + 1.0) * F0_FPS / LAT_FPS - 1.0).floor()
    return f0_raw.float()[i_f.clamp(0, f0_raw.shape[0] - 1).long()]


def time_steps(step_fn, n: int, warm: int) -> dict:
    for _ in range(warm):
        step_fn()
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        step_fn()
        ts[i] = (time.perf_counter() - t0) * 1000
    return {"p50_ms": round(float(np.percentile(ts, 50)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3),
            "p99_ms": round(float(np.percentile(ts, 99)), 3), "max_ms": round(float(ts.max()), 3),
            "mean_rtf": round(float(ts.mean()) / 10.0, 4), "steps": n, "warmup": warm}


def macs_b(d: DecoderA2) -> dict:
    C = d.C
    per_sample = sum(C * C * k + C * C for k in A2_KS) + 2 * C + C * 16
    c = d.ctrl
    W = c.inp.out_channels
    per_frame_ctrl = c.inp.in_channels * W * 3 + len(c.convs) * (W * W * 3 + W * W) + W * c.out.out_channels
    return {"stack_mac_per_frame": per_sample * HOP_LENGTH, "ctrl_mac_per_frame": per_frame_ctrl,
            "total_mac_per_frame": per_sample * HOP_LENGTH + per_frame_ctrl}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--b-channels", type=int, default=16)
    ap.add_argument("--a-taps", type=int, default=12)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    ck = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location="cpu", weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512))
    codec.load_state_dict(ck["ema"])
    codec.eval()
    decA = DecoderAA(taps=a.a_taps)
    missing = decA.load_state_dict(codec.decoder.state_dict(), strict=True)
    decA.eval()
    decB = DecoderA2(channels=a.b_channels).eval()

    pairs, lats, held = build_index(0)
    f = sorted(p for p in pairs if p.parent.name not in set(held))[0]
    d = torch.load(f, map_location="cpu", weights_only=False)
    import librosa
    x, _ = librosa.load(d["path"], sr=SAMPLE_RATE, mono=True)
    T = min(len(x) // HOP_LENGTH, 300)
    x = torch.from_numpy(x[:T * HOP_LENGTH]).float()[None, None]
    f0 = f0_frames(torch.load(F0FIX / f.parent.name / f.name, map_location="cpu", weights_only=False)["f0"], T)[None]
    rep: dict = {"utt": f.stem, "frames": T, "cpu": platform.processor() or platform.machine(),
                 "torch_threads": 1, "note": "PyTorch eager・CPU 1thread。c32の同条件値でRust換算を較正する"}
    with torch.no_grad():
        z = codec.encode(x)
        y32 = codec.decoder(z)
        yA = decA(z)
        noise = torch.randn(1, 1, T * HOP_LENGTH)
        yB = decB(z, f0, noise)
        rep["parity"] = {
            "c32": float((y32 - codec.decoder.stream().decode_chunk(z)).abs().max()),
            "A": float((yA - decA.stream().decode_chunk(z)).abs().max()),
            "B": float((yB - decB.stream().decode_chunk(z, f0, noise)).abs().max())}
        inv = {"A": [], "B": []}
        for t0 in (60, 150, 240):
            z2 = z.clone()
            z2[..., t0:] = torch.randn_like(z2[..., t0:])
            f02 = f0.clone()
            f02[:, t0:] = 400.0
            inv["A"].append(float((decA(z2)[..., :t0 * HOP_LENGTH] - yA[..., :t0 * HOP_LENGTH]).abs().max()))
            inv["B"].append(float((decB(z2, f02, noise)[..., :t0 * HOP_LENGTH]
                                   - yB[..., :t0 * HOP_LENGTH]).abs().max()))
            sens_a = float((decA(z2)[..., t0 * HOP_LENGTH:] - yA[..., t0 * HOP_LENGTH:]).abs().max())
            sens_b = float((decB(z2, f02, noise)[..., t0 * HOP_LENGTH:] - yB[..., t0 * HOP_LENGTH:]).abs().max())
            inv.setdefault("sensitivity_after_edit", []).append((round(sens_a, 4), round(sens_b, 4)))
        rep["future_invariance_max_abs_before_edit"] = inv
        a32, aA = y32[0, 0].numpy(), yA[0, 0].numpy()
        lags = range(-20, 80)
        cc = [float(np.dot(a32[100:-100], np.roll(aA, -L)[100:-100])) for L in lags]
        rep["A_delay"] = {"measured_samples": int(list(lags)[int(np.argmax(cc))]),
                          "analytic_samples": decA.added_delay_samples(),
                          "act_delay_base_samples": decA.d,
                          "ms": round(decA.added_delay_samples() / SAMPLE_RATE * 1000, 3)}
    rep["params"] = {"c32_decoder": sum(p.numel() for p in codec.decoder.parameters()),
                     "A": sum(p.numel() for p in decA.parameters()),
                     "B": sum(p.numel() for p in decB.parameters())}
    from causal_codec import architecture_stats
    rep["macs"] = {"c32_mac_per_frame": architecture_stats(codec).decoder_macs_per_second // 100,
                   "B": macs_b(decB)}
    print(json.dumps({k: v for k, v in rep.items() if k != "rtf"}, ensure_ascii=False, indent=1), flush=True)

    torch.set_num_threads(1)
    zs = z[..., :1].contiguous()
    fs = f0[:, :1].contiguous()
    rtf = {}
    with torch.no_grad():
        s32 = codec.decoder.stream()
        rtf["c32"] = time_steps(lambda: s32.decode_step(zs), a.steps, a.warm)
        print("c32", rtf["c32"], flush=True)
        sA = decA.stream()
        rtf["A"] = time_steps(lambda: sA.decode_step(zs), a.steps, a.warm)
        print("A", rtf["A"], flush=True)
        sB = decB.stream()
        rtf["B"] = time_steps(lambda: sB.decode_step(zs, fs), a.steps, a.warm)
        print("B", rtf["B"], flush=True)
    rep["rtf_pytorch_cpu1"] = rtf
    rep["rust_calibration"] = {"c32_rust_p95_ms": 3.0879,
                               "ratio_pytorch_to_rust_c32": round(rtf["c32"]["p95_ms"] / 3.0879, 3)}
    OUT.mkdir(parents=True, exist_ok=True)
    rep["config"] = {"b_channels": a.b_channels, "a_taps": a.a_taps}
    (OUT / f"probe_step1{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("->", OUT / f"probe_step1{a.tag}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

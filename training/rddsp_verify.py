"""Unit verification for rddsp.py. Every component is checked against a signal
with a KNOWN answer before any audio is handed to the ear."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R

OK, BAD = "PASS", "FAIL"
fails = []


def check(name: str, cond: bool, detail: str) -> None:
    tag = OK if cond else BAD
    if not cond:
        fails.append(name)
    print(f"[{tag}] {name:34s} {detail}")


def tone(f0: float, n: int, harms, amps=None) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float64) / R.SR
    amps = amps or [1.0 / k for k in harms]
    y = sum(a * torch.sin(2 * math.pi * f0 * k * t) for k, a in zip(harms, amps))
    return (y / y.abs().max() * 0.5).float()


def main() -> None:
    n = R.SR * 2

    # ---- f0: the whole point of §9-4 is the MISSING FUNDAMENTAL case
    for label, harms, want in [("full series", [1, 2, 3, 4, 5, 6], 220.0),
                               ("missing f0", [2, 3, 4, 5, 6], 220.0),
                               ("weak f0 x0.05", [1, 2, 3, 4, 5, 6], 220.0)]:
        a = None
        if label.startswith("weak"):
            a = [0.05, 1.0, 0.8, 0.6, 0.4, 0.3]
        x = tone(220.0, n, harms, a)
        f0, voi = R.harmonic_sum_f0(x)
        v = f0[f0 > 50]
        med = float(v.median()) if v.numel() else 0.0
        err = abs(med - want) / want
        check(f"f0 {label}", err < 0.05, f"median {med:6.1f}Hz (want {want}) err {100*err:4.1f}%")

    # ---- MVF: harmonic below B, noise above B -> estimator must find B
    for B in (2500.0, 6000.0):
        t = torch.arange(n, dtype=torch.float64) / R.SR
        f0 = 220.0
        h = sum(torch.sin(2 * math.pi * f0 * k * t) / k
                for k in range(1, int(B / f0) + 1))
        g = torch.Generator().manual_seed(0)
        nz = torch.randn(n, generator=g).double()
        Nz = R.stft(nz.float())
        fb = torch.arange(R.NB)[:, None] * R.SR / R.NFFT
        Nz = Nz * (fb > B).float()
        nzf = R.istft(Nz, n)
        x = (h / h.abs().max() * 0.4).float() + nzf / nzf.abs().max() * 0.12
        mvf = R.estimate_mvf(x, R.harmonic_sum_f0(x)[0])
        med = float(mvf.median())
        check(f"MVF boundary {B:.0f}Hz", abs(med - B) < 0.30 * B, f"median {med:7.1f}Hz")

    # ---- band aperiodicity: pure tone -> low ap, pure noise -> ap ~1
    x = tone(220.0, n, [1, 2, 3, 4, 5, 6])
    ap_t, bands = R.estimate_bap(x, R.harmonic_sum_f0(x)[0])
    g = torch.Generator().manual_seed(1)
    xn = torch.randn(n, generator=g) * 0.2
    ap_n, _ = R.estimate_bap(xn, R.harmonic_sum_f0(xn)[0])
    check("bap tone vs noise", float(ap_t[0].median()) < float(ap_n[0].median()),
          f"tone {float(ap_t[0].median()):.3f} < noise {float(ap_n[0].median()):.3f} | {len(bands)} bands")

    # ---- FIR: magnitude response must match the requested log-magnitude
    T = 8
    fb = torch.arange(R.NB, dtype=torch.float32)[:, None].expand(R.NB, T)
    want = 1.5 - 4.0 * (fb / R.NB) - 2.5 * ((fb - 60.0) / 300.0) ** 2   # realistic log-mag, ~[-8,1.5]
    want = want.clamp(min=-8.0).contiguous()
    ir_min = R._min_phase_ir(want, R.NFFT)
    H = torch.fft.rfft(ir_min, n=R.NFFT, dim=-1).abs()[0]
    err = (torch.log(H + 1e-9) - want[:, 0]).abs().mean()
    check("FIR min-phase |H| match", float(err) < 0.05, f"mean |log|H| - target| = {float(err):.4f}")
    e_first = float((ir_min[0][: R.NFFT // 8] ** 2).sum() / (ir_min[0] ** 2).sum())
    check("FIR min-phase causality", e_first > 0.95, f"{100*e_first:.1f}% energy in first 1/8 (causal)")
    ir_lin = R._linear_phase_ir(want, R.NFFT)
    pk = int(ir_lin[0].abs().argmax())
    check("FIR linear-phase centered", abs(pk - R.NFFT // 2) < 4, f"peak at {pk} (n/2={R.NFFT//2})")

    # ---- mixed-phase must carry ANTICAUSAL energy that min-phase cannot
    import librosa
    from kansei_train import CACHE, DATA
    uid = sorted(p.stem for p in CACHE.glob("*.npz"))[-1]
    xr, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
    xr = torch.tensor(xr[: R.SR * 4])
    f0r, _ = R.harmonic_sum_f0(xr)
    fm = float(f0r[f0r > 50].median())
    gci, soe = R.zff_gci(xr, fm)
    dd = torch.diff(gci).float()
    dd = dd[(dd > R.SR / 600) & (dd < R.SR / 60)]
    gci_hz = R.SR / float(dd.median())
    check("ZFF GCI rate vs f0", abs(gci_hz - fm) / fm < 0.15,
          f"GCI {gci_hz:.1f}Hz vs f0 {fm:.1f}Hz ({len(gci)} instants, no f0 estimator used)")

    def cep_ratio(ir: torch.Tensor, f0v: float = 220.0) -> float:
        n2 = R.SR
        pulse = torch.zeros(n2)
        pulse[:: int(R.SR / f0v)] = 1.0
        y = torch.nn.functional.conv1d(pulse[None, None], ir.flip(0)[None, None],
                                       padding=ir.shape[-1] - 1)[0, 0][:n2]
        y = y / y.abs().max() * 0.5
        g, _ = R.zff_gci(y, f0v)
        c = R.gci_complex_cepstrum(y, g[:150], f0v)
        return float(c[:, -40:].abs().mean()) / (float(c[:, 1:41].abs().mean()) + 1e-12)

    mp = R._min_phase_ir(want, R.NFFT)[0][:512]
    r_min, r_max = cep_ratio(mp), cep_ratio(mp.flip(0))
    check("cepstrum separates min/max phase", r_min < 0.5 and r_max > 2.0,
          f"min-phase {r_min:.4f} < 0.5 < 2.0 < max-phase {r_max:.1f}")
    ccs = R.gci_complex_cepstrum(xr, gci[:400], fm)
    r_speech = float(ccs[:, -40:].abs().mean()) / (float(ccs[:, 1:41].abs().mean()) + 1e-12)
    check("speech has anticausal part", r_speech > 5.0 * r_min,
          f"speech neg/pos {r_speech:.3f} vs min-phase {r_min:.4f} (glottal open phase present)")

    # ---- full round trip on real speech
    t0 = time.time()
    y, h, nz, p = R.resynthesize(xr)
    dt = time.time() - t0
    lsd = R.logspec(xr, y) if hasattr(R, "logspec") else _lsd(xr, y)
    lsd_h = _lsd(xr, h)
    vfrac = float((p["f0"] > 50).float().mean())
    check("round trip finite", bool(torch.isfinite(y).all()), f"len {y.shape[-1]}")
    edges = [0, 500, 1000, 2000, 4000, 8000, 16000, 22050]
    A, B = R.stft(xr).abs() ** 2, R.stft(y).abs() ** 2
    Tn = min(A.shape[-1], B.shape[-1])
    binhz = R.SR / R.NFFT
    dbs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        i, j = int(lo / binhz), int(hi / binhz)
        ea, eb = A[i:j, :Tn].sum(), B[i:j, :Tn].sum()
        dbs.append(10 * math.log10(float(eb + 1e-12) / float(ea + 1e-12)))
    worst = max(abs(d) for d in dbs)
    check("round trip band energy", worst < 3.0,
          "per-band dB " + " ".join(f"{d:+.1f}" for d in dbs) + f" | worst {worst:.1f}dB")
    print(f"[info] LSD {lsd:.3f} (harm-only {lsd_h:.3f}) -- diagnostic only: LSD rewards an "
          f"all-harmonic solution and must NOT be used to tune MVF")
    print(f"[info] voiced {100*vfrac:.1f}%  f0 med {float(p['f0'][p['f0']>50].median()):.0f}Hz  "
          f"MVF med {float(p['mvf'].median()):.0f}Hz  ap[0] med {float(p['ap'][0].median()):.3f}")
    print(f"[info] render {dt:.2f}s for {xr.shape[-1]/R.SR:.1f}s audio = RTF {dt/(xr.shape[-1]/R.SR):.2f} (CPU, unoptimized)")

    print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
    sys.exit(1 if fails else 0)


def _lsd(a: torch.Tensor, b: torch.Tensor) -> float:
    A, B = R.stft(a).abs(), R.stft(b).abs()
    T = min(A.shape[-1], B.shape[-1])
    return float((torch.log(A[..., :T] + 1e-5) - torch.log(B[..., :T] + 1e-5)).abs().mean())


if __name__ == "__main__":
    main()

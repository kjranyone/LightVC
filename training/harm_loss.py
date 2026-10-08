"""調和エネルギー対比損失(検査L: 損失のpitch感度の機械化)。

voiced枠で条件f0の倍音位置のSTFT振幅を近傍(±eps)・半分位置と比で対比する。
推論経路に依存しない学習専用損失。

    CUDA_VISIBLE_DEVICES=0 uv run python harm_loss.py   # 感度検査(検査L)
"""
from __future__ import annotations

import torch

N_FFT = 2048
HOP = 480
N_HARM = 4
EPS_R = 0.03


def _interp_bins(S: torch.Tensor, freq_hz: torch.Tensor) -> torch.Tensor:
    """S [B,F,T], freq_hz [B,T,K] -> 線形補間振幅 [B,T,K]。"""
    bin_hz = 48000.0 / N_FFT
    b = freq_hz / bin_hz
    F_bins = S.shape[1]
    lo = b.floor().clamp(0, F_bins - 2).long()
    fr = (b - lo).clamp(0.0, 1.0)
    Sp = S.permute(0, 2, 1)                                # [B,T,F]
    v_lo = Sp.gather(2, lo)
    v_hi = Sp.gather(2, (lo + 1).clamp(max=F_bins - 1))
    return v_lo * (1 - fr) + v_hi * fr


def harm_contrast(wav: torch.Tensor, f0_hz: torch.Tensor,
                  n_harm: int = N_HARM, eps_r: float = EPS_R) -> torch.Tensor:
    """wav [B,N]@48k, f0_hz [B,T]@100fps(0=無声) -> scalar loss (voiced平均)。"""
    B, N = wav.shape
    w = torch.hann_window(N_FFT, device=wav.device)
    S = torch.stft(wav, N_FFT, HOP, N_FFT, w, return_complex=True).abs()
    T = min(f0_hz.shape[1], S.shape[2])
    S = S[:, :, :T]
    f0 = f0_hz[:, :T]
    voiced = f0 > 50.5
    f0v = f0 * voiced
    if voiced.sum() < 8:
        return wav.new_zeros(())
    ks = torch.arange(1, n_harm + 1, device=wav.device).view(1, 1, -1)
    f0k = f0v.unsqueeze(-1) * ks                            # [B,T,K]
    nyq = 24000.0 * 0.95
    valid = f0k < nyq
    f0k = f0k * valid
    E_h = _interp_bins(S, f0k).sum(-1)                      # [B,T]
    f_n = torch.stack([f0k * (1 + eps_r), f0k * (1 - eps_r)], -1).mean(-1)
    E_near = _interp_bins(S, f_n).sum(-1)
    E_half = _interp_bins(S, f0v.unsqueeze(-1) * 0.5)
    E_a = E_near + E_half.squeeze(-1) * 0.5
    ratio = E_a / (E_h + E_a + 1e-5)
    return (ratio * voiced).sum() / voiced.sum().clamp(min=1)


def sensitivity_check() -> int:
    import json
    import sys
    from pathlib import Path
    import librosa
    import numpy as np

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(Path(__file__).parent))
    clips = [("fecf5112354be881", "fecf5112354be881_00006519"),
             ("fe8c22d913075909", "fe8c22d913075909_00046177"),
             ("209c94d37412922a", "209c94d37412922a_00008741")]
    out = {}
    for spk, stem in clips:
        f0 = torch.load(ROOT / "data/female_real_f0fix" / spk / (stem + ".pt"),
                        map_location="cpu", weights_only=False)["f0"].float()
        y48, _ = librosa.load(str(ROOT / "female-dataset" / spk / (stem + ".wav")),
                              sr=48000, mono=True)
        y = torch.from_numpy(y48)[None]
        n = min(len(y48) // HOP, f0.shape[0])
        f0g = f0[:n][None]
        yg = y[:, :n * HOP]
        curve = {}
        for st in (0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 7.0, -7.0):
            fs = torch.where(f0g > 0, f0g * 2.0 ** (st / 12.0), f0g)
            curve[st] = round(float(harm_contrast(yg, fs)), 4)
        out[stem] = curve
        print(stem[:20], curve, flush=True)
    ok = True
    for stem, c in out.items():
        l0 = c[0.0]
        mono_up = all(c[s] >= l0 for s in (1.0, 2.0, 3.0, 7.0))
        mono_dn = all(c[s] >= l0 for s in (-1.0, -2.0, -3.0, -7.0))
        disc1 = c[1.0] >= 1.5 * l0 and c[-1.0] >= 1.5 * l0
        print(f"  {stem[:16]}: min@0={l0 <= min(c.values())} mono={mono_up and mono_dn} "
              f"disc@1st={disc1}")
        ok = ok and (l0 <= min(c.values())) and mono_up and mono_dn and disc1
    verdict = "PASS" if ok else "FAIL"
    print("検査L:", verdict)
    out_path = ROOT / "results/s18_cfm_harmloss/sensitivity.json"
    out_path.write_text(json.dumps({"curves": out, "verdict": verdict}, indent=1))
    return 0 if ok else 1



N_HARM_V2 = 0
W_ACF = 3072


def acf_contrast(wav: torch.Tensor, f0_hz: torch.Tensor) -> torch.Tensor:
    """自己相関対比損失v2: 周期ドメインで±1stを判別(分解能の壁なし)。

    voiced枠で AC(lag_c) を AC(lag_c*(1±5.9%))(±1st)と比で対比。
    倍音構造周期信号は AC(lag_c) が高く、条件周期が±1stずれれば急落する。
    """
    B, N = wav.shape
    T = min(f0_hz.shape[1], max(0, (N - W_ACF) // HOP + 1))
    if T < 4:
        return wav.new_zeros(())
    f0 = f0_hz[:, :T]
    voiced = f0 > 50.5
    f0v = f0 * voiced
    lag_c = (48000.0 / f0v.clamp(min=60.0)).round().long().clamp(8, W_ACF // 3)
    base = (torch.arange(T, device=wav.device) * HOP)[None, :].expand(B, -1)
    win = torch.arange(W_ACF, device=wav.device)[None, None, :]
    bidx = torch.arange(B, device=wav.device)[:, None, None]
    idx = (base[:, :, None] + win).clamp(0, N - 1)
    frames = wav[bidx, idx]

    def acf(lag):
        idx2 = (base[:, :, None] + win + lag[:, :, None]).clamp(0, N - 1)
        fr2 = wav[bidx, idx2]
        num = (frames * fr2).sum(-1)
        den = (frames * frames).sum(-1).clamp(min=1e-4)
        return num / den

    a_c = acf(lag_c)
    devs = (0.941, 1.059, 0.881, 1.122, 0.866, 1.126)
    a_anti = torch.stack([acf((lag_c.float() * d).round().long()) for d in devs]).mean(0)
    ratio = (1 - a_c) / (2 - a_c - a_anti + 1e-5)
    return (ratio * voiced).sum() / voiced.sum().clamp(min=1)


def sensitivity_check_v2() -> int:
    import json
    import sys
    from pathlib import Path
    import librosa

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(Path(__file__).parent))
    clips = [("fecf5112354be881", "fecf5112354be881_00006519"),
             ("fe8c22d913075909", "fe8c22d913075909_00046177"),
             ("209c94d37412922a", "209c94d37412922a_00008741")]
    out = {}
    for spk, stem in clips:
        f0 = torch.load(ROOT / "data/female_real_f0fix" / spk / (stem + ".pt"),
                        map_location="cpu", weights_only=False)["f0"].float()
        y48, _ = librosa.load(str(ROOT / "female-dataset" / spk / (stem + ".wav")),
                              sr=48000, mono=True)
        y = torch.from_numpy(y48)[None]
        n = min(len(y48) // HOP, f0.shape[0])
        f0g = f0[:n][None]
        yg = y[:, :n * HOP]
        curve = {}
        for st in (0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 7.0, -7.0):
            fs = torch.where(f0g > 0, f0g * 2.0 ** (st / 12.0), f0g)
            curve[st] = round(float(acf_contrast(yg, fs)), 4)
        out[stem] = curve
        print(stem[:20], curve, flush=True)
    ok = True
    for stem, c in out.items():
        l0 = c[0.0]
        mono_up = all(c[s] >= l0 for s in (1.0, 2.0, 3.0, 7.0))
        mono_dn = all(c[s] >= l0 for s in (-1.0, -2.0, -3.0, -7.0))
        disc1 = c[1.0] >= 1.5 * l0 and c[-1.0] >= 1.5 * l0
        print(f"  {stem[:16]}: min@0={l0 <= min(c.values())} mono={mono_up and mono_dn} "
              f"disc@1st={disc1}")
        ok = ok and (l0 <= min(c.values())) and mono_up and mono_dn and disc1
    verdict = "PASS" if ok else "FAIL"
    print("検査L v2(ACF):", verdict)
    out_path = ROOT / "results/s18_cfm_harmloss/sensitivity_v2.json"
    out_path.write_text(json.dumps({"curves": out, "verdict": verdict}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(sensitivity_check_v2())
